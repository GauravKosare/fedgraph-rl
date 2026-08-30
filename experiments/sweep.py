"""Multi-seed sweep: run the full pipeline over several graph seeds and report
mean +/- std per policy, so the comparison isn't hostage to one partition.

Run:  python experiments/sweep.py [n_seeds] [--stragglers]

  --stragglers   enable the v0.2 straggler/deadline environment and write to
                 experiments/sweep_results_stragglers.{json,png} instead of the
                 canonical v0.1 sweep_results.{json,png}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedgraphrl.config import Config
from run_experiment import run_once

POLICIES = ["rl_reinforce", "random", "fraud_greedy", "all", "centralised_upper_bound"]
METRICS = ["f1", "precision", "recall", "auc"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stragglers = "--stragglers" in sys.argv
    n_seeds = int(args[0]) if args else 5
    cfg = Config()
    cfg.episodes = 100          # RL updates per seed
    cfg.stragglers = stragglers
    tag = "_stragglers" if stragglers else ""
    base_seed = cfg.seed

    per_seed = []
    t0 = time.time()
    for i in range(n_seeds):
        seed = base_seed + 100 * i
        print(f"\n----- seed {seed}  ({i + 1}/{n_seeds}) -----")
        res, _ = run_once(cfg, seed, verbose=False)
        row = {p: {m: res[p][m] for m in METRICS} for p in POLICIES}
        per_seed.append({"seed": seed, "results": row})
        for p in POLICIES:
            print(f"  {p:24s} F1={row[p]['f1']:.3f}  AUC={row[p]['auc']:.3f}")

    # -- aggregate -------------------------------------------------------
    print(f"\n================ SWEEP SUMMARY ({n_seeds} seeds) ================")
    print(f"{'policy':24s} {'F1 mean+/-std':>16s} {'AUC mean+/-std':>16s} "
          f"{'precision':>12s} {'recall':>12s}")
    summary = {}
    for p in POLICIES:
        agg = {m: np.array([s['results'][p][m] for s in per_seed]) for m in METRICS}
        summary[p] = {m: [float(agg[m].mean()), float(agg[m].std())] for m in METRICS}
        print(f"{p:24s} "
              f"{agg['f1'].mean():.3f}+/-{agg['f1'].std():.3f}   "
              f"{agg['auc'].mean():.3f}+/-{agg['auc'].std():.3f}   "
              f"{agg['precision'].mean():.3f}+/-{agg['precision'].std():.3f} "
              f"{agg['recall'].mean():.3f}+/-{agg['recall'].std():.3f}")

    # rank RL vs each baseline on F1, per seed (paired)
    print("\nPaired comparison (RL F1 - baseline F1, per seed):")
    rl = np.array([s['results']['rl_reinforce']['f1'] for s in per_seed])
    for p in POLICIES[1:]:
        b = np.array([s['results'][p]['f1'] for s in per_seed])
        d = rl - b
        wins = int((d > 0).sum())
        print(f"  vs {p:24s} mean d={d.mean():+.3f}  RL wins {wins}/{n_seeds}")

    out = Path(__file__).resolve().parents[1] / "experiments" / f"sweep_results{tag}.json"
    out.write_text(json.dumps({"n_seeds": n_seeds, "config": cfg.to_dict(),
                               "per_seed": per_seed, "summary": summary}, indent=2))
    print(f"\nsaved {out}   ({time.time() - t0:.1f}s)")

    try:
        _plot(per_seed, summary, out.with_name("sweep_results.png"))
        print(f"saved {out.with_name('sweep_results.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot(per_seed, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(POLICIES))
    for j, metric in enumerate(("f1", "auc")):
        means = [summary[p][metric][0] for p in POLICIES]
        stds = [summary[p][metric][1] for p in POLICIES]
        ax[j].bar(x, means, yerr=stds, capsize=4)
        for i, p in enumerate(POLICIES):
            pts = [s['results'][p][metric] for s in per_seed]
            ax[j].scatter(np.full(len(pts), i), pts, color="k", s=14, zorder=3)
        ax[j].set_xticks(x)
        ax[j].set_xticklabels([p.replace("_", "\n") for p in POLICIES], fontsize=8)
        ax[j].set_title(f"test {metric.upper()} across {len(per_seed)} seeds")
        ax[j].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
