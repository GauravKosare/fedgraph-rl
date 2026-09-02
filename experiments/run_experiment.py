"""End-to-end experiment.

1. Build a synthetic transaction graph with planted fraud rings.
2. Shard it across simulated edge devices in a non-IID way.
3. Train the global GCN fraud detector with FedAvg, where an RL controller
   decides each round which clients participate and how to allocate the
   local-compute budget.
4. Compare the learned policy against random / fraud-greedy / all-clients
   baselines, and against a centralised (non-federated) upper bound.

Run:  python experiments/run_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedgraphrl.config import Config
from fedgraphrl.data import make_transaction_graph, partition_non_iid
from fedgraphrl.environment import FederatedEnv
from fedgraphrl.federated import FederatedServer
from fedgraphrl.gnn import GCN, SGD
from fedgraphrl.autograd import Tensor
from fedgraphrl.rl_controller import ReinforceController, HeuristicController


def centralised_upper_bound(data, model_cfg, epochs=120):
    model = GCN(**model_cfg, seed=0)
    opt = SGD(model.params, lr=0.05)
    cw = np.array([1.0, 5.0])
    for _ in range(epochs):
        model.zero_grad()
        logits = model.forward(data.features, data.adj, training=True)
        sel = Tensor(np.eye(data.num_nodes)[data.train_mask]) @ logits
        loss = sel.softmax_cross_entropy(data.labels[data.train_mask], weight=cw)
        loss.backward()
        opt.step()
    srv = FederatedServer(data, model_cfg)
    srv.model = model
    return srv.evaluate("test", threshold=srv.tuned_threshold())


def make_env(cfg, data, shards, model_cfg, seed):
    return FederatedEnv(data, shards, model_cfg,
                        max_rounds=cfg.max_rounds,
                        clients_per_round=cfg.clients_per_round,
                        epoch_budget=cfg.epoch_budget,
                        cost_coeff=cfg.cost_coeff,
                        stragglers=cfg.stragglers,
                        straggler_frac=cfg.straggler_frac,
                        straggler_slowdown=cfg.straggler_slowdown,
                        deadline_slack=cfg.deadline_slack,
                        drop_penalty=cfg.drop_penalty, seed=seed)


def run_once(cfg: Config, seed: int, verbose: bool = True):
    """Full pipeline for one graph seed. Returns (results, history)."""
    np.random.seed(seed)
    data, ring_of = make_transaction_graph(
        num_accounts=cfg.num_accounts, num_features=cfg.num_features,
        fraud_ring_count=cfg.fraud_ring_count, seed=seed)
    shards = partition_non_iid(data, ring_of, num_clients=cfg.num_clients,
                               dirichlet_alpha=cfg.dirichlet_alpha, seed=seed)
    model_cfg = {**cfg.model_cfg(), "in_dim": data.num_features}

    if verbose:
        print(f"graph: {data.num_nodes} accounts, fraud rate {data.labels.mean():.3f}, "
              f"{int((data.adj > 0).sum())} directed adj entries")
        for i, s in enumerate(shards):
            print(f"  client {i}: {len(s)} nodes, shard fraud rate {data.labels[s].mean():.3f}")

    env = make_env(cfg, data, shards, model_cfg, seed=seed)
    # v0.2.1 advantage guards apply only on the straggler path so the v0.1
    # canonical numbers stay reproducible (defaults below are no-ops).
    adv_floor = cfg.adv_std_floor if cfg.stragglers else 1e-8
    adv_clip = cfg.adv_clip if cfg.stragglers else 1e9
    agent = ReinforceController(env, lr=cfg.lr, entropy_coeff=cfg.entropy_coeff,
                                use_critic=cfg.use_critic, critic_lr=cfg.critic_lr,
                                adv_std_floor=adv_floor, adv_clip=adv_clip,
                                seed=seed)
    history = []
    for ep in range(cfg.episodes):
        # v0.2.1 stabilisation (straggler path only -- keeps the v0.1 canonical
        # run byte-identical): linearly anneal exploration, high early / clean
        # exploitation late.
        if cfg.stragglers:
            frac = 1.0 - ep / max(cfg.episodes - 1, 1)
            agent.entropy_coeff = cfg.entropy_coeff * (
                cfg.entropy_final_frac + (1.0 - cfg.entropy_final_frac) * frac)
        ret, info = agent.run_episode(train=True, rollouts=cfg.rollouts_per_update)
        history.append({"episode": ep, "return": ret, "val_f1": info["val_f1"]})
        if verbose and (ep % 5 == 0 or ep == cfg.episodes - 1):
            print(f"  ep {ep:3d}  return {ret:8.2f}  val_f1 {info['val_f1']:.3f}")

    def eval_policy(n=7):
        keys = ("f1", "precision", "recall", "auc", "threshold")
        runs = [agent._rollout(greedy=False)[2]["test"] for _ in range(n)]
        return {k: float(np.mean([r[k] for r in runs])) for k in keys}

    results = {"rl_reinforce": eval_policy()}
    for kind in ("random", "fraud_greedy", "all"):
        env_b = make_env(cfg, data, shards, model_cfg, seed=seed + 1)
        _, info = HeuristicController(env_b, kind=kind, seed=seed).run_episode()
        results[kind] = info["test"]
    results["centralised_upper_bound"] = centralised_upper_bound(data, model_cfg)
    return results, history


def main():
    cfg = Config()
    t0 = time.time()
    results, history = run_once(cfg, cfg.seed, verbose=True)

    # -- report -----------------------------------------------------
    print("\n=== TEST-set fraud detection (F1 / precision / recall / AUC) ===")
    for name, m in results.items():
        print(f"  {name:24s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  AUC={m['auc']:.3f}  thr={m.get('threshold', 0.5):.3f}")

    out = Path(__file__).resolve().parents[1] / "experiments" / "results.json"
    out.write_text(json.dumps(
        {"config": cfg.to_dict(), "history": history, "results": results},
        indent=2))
    print(f"\nsaved {out}   ({time.time() - t0:.1f}s)")

    try:
        _plot(history, results, out.with_name("results.png"))
        print(f"saved {out.with_name('results.png')}")
    except Exception as e:                      # matplotlib optional
        print(f"(plot skipped: {e})")


def _plot(history, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    eps = [h["episode"] for h in history]
    ax[0].plot(eps, [h["return"] for h in history], label="episode return")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("return"); ax[0].legend()
    ax[0].set_title("RL controller learning curve")

    names = list(results); f1s = [results[n]["f1"] for n in names]
    ax[1].barh(names, f1s)
    ax[1].set_xlabel("test F1"); ax[1].set_title("Fraud detection: policy vs baselines")
    fig.tight_layout(); fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
