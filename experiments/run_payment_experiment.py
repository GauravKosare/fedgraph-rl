"""v0.3 experiment -- federated mule-account detection on the payment-flow model.

Same RL controller and baselines as `run_experiment.py`, but:
  * data  = `make_payment_graph` (victim -> mule -> layering -> cash-out)
  * split = one federated client per bank (`partition_by_bank`) -- non-IID by
            ownership, not a Dirichlet knob
  * reward / metric = **money-recall at a false-positive budget** (Δ money at
            risk on caught mules), not F1

Run:  python experiments/run_payment_experiment.py [n_seeds] [--scarce]

  --scarce   v0.3.1 scarce-coverage regime: only 8 federated rounds (2 of 6 banks
             each -> ~2.7 visits/bank, so random selection leaves some banks
             barely trained) and stragglers on (banks with old infra contribute
             partial updates). Writes results_payment_scarce*.{json,png}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedgraphrl.config import Config
from fedgraphrl.payment_data import make_payment_graph, partition_by_bank
from fedgraphrl.environment import FederatedEnv
from fedgraphrl.federated import FederatedServer
from fedgraphrl.gnn import GCN, SGD
from fedgraphrl.autograd import Tensor
from fedgraphrl.rl_controller import ReinforceController, HeuristicController

METRIC_KEYS = ("money_recall", "recall", "precision", "fp_rate", "f1", "auc", "threshold")


def make_env(cfg, data, shards, model_cfg, seed):
    return FederatedEnv(data, shards, model_cfg,
                        max_rounds=cfg.max_rounds,
                        clients_per_round=min(cfg.clients_per_round, len(shards)),
                        epoch_budget=cfg.epoch_budget,
                        cost_coeff=cfg.cost_coeff,
                        stragglers=cfg.stragglers,
                        straggler_frac=cfg.straggler_frac,
                        straggler_slowdown=cfg.straggler_slowdown,
                        deadline_slack=cfg.deadline_slack,
                        drop_penalty=cfg.drop_penalty,
                        reward_mode="money", fp_budget=cfg.fp_budget, seed=seed)


def centralised_money(data, model_cfg, fp_budget, epochs=150, patience=25):
    """Pooled-data reference (not a strict upper bound -- on this noisy graph the
    per-bank subgraph training + FedAvg regularisation can rank mules better than
    a single GCN over the whole graph). Early-stopped on validation AUC."""
    model = GCN(**model_cfg, seed=0)
    opt = SGD(model.params, lr=0.05)
    cw = np.array([1.0, 5.0])
    srv = FederatedServer(data, model_cfg)
    d = data
    best_val, best_w, since = -1.0, model.get_weights(), 0
    for ep in range(epochs):
        model.zero_grad()
        logits = model.forward(data.features, data.adj, training=True)
        sel = Tensor(np.eye(data.num_nodes)[data.train_mask]) @ logits
        loss = sel.softmax_cross_entropy(data.labels[data.train_mask], weight=cw)
        loss.backward()
        opt.step()
        if ep >= 30:                       # early-stop on smooth val AUC, not the noisy money-recall
            from fedgraphrl.metrics import roc_auc
            p = model.predict_proba(d.features, d.adj)[:, 1]
            v = roc_auc(d.labels[d.val_mask], p[d.val_mask])
            if v > best_val:
                best_val, best_w, since = v, model.get_weights(), 0
            else:
                since += 1
                if since >= patience:
                    break
    model.set_weights(best_w)
    srv.model = model
    thr = srv.tuned_threshold_money(fp_budget)
    return srv.evaluate_money("test", fp_budget=fp_budget, threshold=thr)


def run_once(cfg: Config, seed: int, verbose=True):
    np.random.seed(seed)
    data, episodes = make_payment_graph(
        n_accounts=cfg.num_accounts, n_banks=cfg.n_banks,
        n_scam_episodes=cfg.n_scam_episodes, num_features=cfg.num_features, seed=seed)
    shards = partition_by_bank(data)
    model_cfg = {**cfg.model_cfg(), "in_dim": data.num_features}

    if verbose:
        tot = data.amount_at_risk[data.labels == 1].sum()
        print(f"payment graph: {data.num_nodes} accounts, {len(episodes)} scam episodes, "
              f"mule rate {data.labels.mean():.3f}, £{tot:,.0f} at risk, "
              f"{cfg.n_banks} banks")
        for i, s in enumerate(shards):
            print(f"  bank {i}: {len(s)} accounts, mule rate {data.labels[s].mean():.3f}")

    env = make_env(cfg, data, shards, model_cfg, seed=seed)
    adv_floor = cfg.adv_std_floor if cfg.stragglers else 1e-8
    adv_clip = cfg.adv_clip if cfg.stragglers else 1e9
    agent = ReinforceController(env, lr=cfg.lr, entropy_coeff=cfg.entropy_coeff,
                                use_critic=cfg.use_critic, critic_lr=cfg.critic_lr,
                                adv_std_floor=adv_floor, adv_clip=adv_clip, seed=seed)
    history = []
    for ep in range(cfg.episodes):
        frac = 1.0 - ep / max(cfg.episodes - 1, 1)
        agent.entropy_coeff = cfg.entropy_coeff * (
            cfg.entropy_final_frac + (1.0 - cfg.entropy_final_frac) * frac)
        ret, info = agent.run_episode(train=True, rollouts=cfg.rollouts_per_update)
        history.append({"episode": ep, "return": ret, "val_signal": info["val_f1"]})
        if verbose and (ep % 5 == 0 or ep == cfg.episodes - 1):
            print(f"  ep {ep:3d}  return {ret:8.2f}  val money-recall {info['val_f1']:.3f}")

    def eval_policy(n=7):
        runs = [agent._rollout(greedy=False)[2]["test"] for _ in range(n)]
        return {k: float(np.mean([r[k] for r in runs])) for k in METRIC_KEYS}

    results = {"rl_reinforce": eval_policy()}
    for kind in ("random", "coverage", "fraud_greedy", "all"):
        env_b = make_env(cfg, data, shards, model_cfg, seed=seed + 1)
        _, info = HeuristicController(env_b, kind=kind, seed=seed).run_episode()
        results[kind] = info["test"]
    results["centralised_pooled"] = centralised_money(data, model_cfg, cfg.fp_budget)
    return results, history


def _fmt(name, m):
    return (f"  {name:24s}  money-recall={m['money_recall']:.3f}  "
            f"FP={m['fp_rate']:.3f}  recall={m['recall']:.3f}  AUC={m['auc']:.3f}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scarce = "--scarce" in sys.argv
    n_seeds = int(args[0]) if args else 1
    cfg = Config()
    cfg.data_model = "payment_flow"
    cfg.reward_mode = "money"
    cfg.clients_per_round = 2          # 2 of 6 banks -> which 2 genuinely matters
    if scarce:                        # v0.3.1: make coverage genuinely scarce
        cfg.max_rounds = 8
        cfg.stragglers = True
    if n_seeds > 1:
        cfg.episodes = 100
    scarce_tag = "_scarce" if scarce else ""
    t0 = time.time()

    per_seed = []
    for i in range(n_seeds):
        seed = cfg.seed + 100 * i
        if n_seeds > 1:
            print(f"\n----- seed {seed} ({i+1}/{n_seeds}) -----")
        res, hist = run_once(cfg, seed, verbose=(n_seeds == 1))
        per_seed.append({"seed": seed, "results": res, "history": hist})
        for name, m in res.items():
            print(_fmt(name, m))

    tag = ("" if n_seeds == 1 else "_sweep") + scarce_tag
    out = Path(__file__).resolve().parents[1] / "experiments" / f"results_payment{tag}.json"
    payload = {"config": cfg.to_dict(), "per_seed": per_seed}
    if n_seeds > 1:
        pol = list(per_seed[0]["results"])
        summary = {}
        print(f"\n=========== PAYMENT SWEEP ({n_seeds} seeds) ===========")
        for p in pol:
            v = np.array([s["results"][p]["money_recall"] for s in per_seed])
            summary[p] = [float(v.mean()), float(v.std())]
            print(f"  {p:24s} money-recall {v.mean():.3f} +/- {v.std():.3f}")
        rl = np.array([s["results"]["rl_reinforce"]["money_recall"] for s in per_seed])
        print("\nPaired (RL money-recall - baseline):")
        for p in pol[1:]:
            b = np.array([s["results"][p]["money_recall"] for s in per_seed])
            print(f"  vs {p:22s} mean d={ (rl-b).mean():+.3f}  RL wins {int(((rl-b)>0).sum())}/{n_seeds}")
        payload["summary"] = summary
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}   ({time.time()-t0:.1f}s)")

    try:
        _plot(per_seed, out.with_suffix(".png"), n_seeds)
        print(f"saved {out.with_suffix('.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot(per_seed, path, n_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pol = list(per_seed[0]["results"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    h = per_seed[0]["history"]
    ax[0].plot([x["episode"] for x in h], [x["return"] for x in h])
    ax[0].set_title("RL learning curve (seed %d)" % per_seed[0]["seed"])
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("return")
    x = np.arange(len(pol))
    means = [np.mean([s["results"][p]["money_recall"] for s in per_seed]) for p in pol]
    stds = [np.std([s["results"][p]["money_recall"] for s in per_seed]) for p in pol]
    ax[1].bar(x, means, yerr=stds, capsize=4)
    for i, p in enumerate(pol):
        pts = [s["results"][p]["money_recall"] for s in per_seed]
        ax[1].scatter(np.full(len(pts), i), pts, color="k", s=14, zorder=3)
    ax[1].set_xticks(x); ax[1].set_xticklabels([p.replace("_", "\n") for p in pol], fontsize=8)
    ax[1].set_ylim(0, 1); ax[1].set_title(f"money-recall @ FP budget ({n_seeds} seed%s)" % ("" if n_seeds == 1 else "s"))
    fig.tight_layout(); fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
