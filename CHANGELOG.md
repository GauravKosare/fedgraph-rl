# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.1] — unreleased

Fixes the v0.2 straggler-variant training collapse; adds the target-problem and
resource docs.

### Changed — straggler mechanics
- **Partial participation.** A selected client now trains the number of epochs it
  can *finish* before the deadline (`floor(deadline · speed / cost-per-epoch)`);
  its partial update is aggregated, compute-weighted by epochs done. It is
  *dropped* only if it can't complete even one epoch. (Was all-or-nothing.)
- Gentler defaults: `straggler_slowdown 4 → 3`, `deadline_slack 1.15 → 1.6`.
- Cost is now the deadline-capped compute actually burned, computed in the env.

### Added — REINFORCE stabilisation (straggler path only; v0.1 untouched)
- `drop_penalty` (0.6) — dense reward penalty per dropped update.
- Tuned-threshold reward: `ΔF1` at the best validation threshold, not 0.5, so a
  conservative model still gives signal. New `best_f1_threshold` (vectorised,
  ~40× faster than the old unique-value scan) and `FederatedServer.val_f1_tuned`.
- `adv_std_floor` (1.0) and `adv_clip` (±8) — guard the advantage normalisation.
- Entropy annealing to `entropy_final_frac` (10 %) of the start value.
- `info["partial"]` — count of partial (deadline-truncated) updates per round.

### Added — docs
- `docs/TARGET_PROBLEM.md` — recommendation to commit the project to *federated
  mule-account detection for APP scams*, with rationale and a reframing plan.
- `docs/PROBLEM_EXPLAINED.md` — the scam and the detection problem explained
  simply (incl. ELI5, why it matters, the fraud lifecycle).
- `docs/RESOURCES.md` — datasets, simulators (AMLSim, PaySim), benchmarks
  (GADBench, DGFraud), the PETs Prize Challenge, FL frameworks, papers.

### Results (5 seeds, `sweep.py 5 --stragglers`)
- **Instability fixed** — no seed collapsed (worst RL F1 0.387, was 0.196). Seed
  207 recovered to 0.387 and now beats random on that seed.
- RL vs uniform-random: **+0.051 F1, wins 3/5** (history: −0.017 → +0.037 → +0.051).
- RL vs fixed cohort: **+0.115 F1, wins 5/5**.
- RL 0.580 ± 0.154 · random 0.529 ± 0.188 · fraud_greedy 0.629 ± 0.098.
- RL still does **not** beat the fraud-rate heuristic (−0.049, 1/5) — a lean, not
  a clean win. Next levers are in `docs/TARGET_PROBLEM.md`, not more RL tuning.

---

## [0.1.0] — 2026-08-30

First complete version. Experiment frozen for write-up.

### Added
- `fedgraphrl.autograd` — ~150-line reverse-mode autograd engine (NumPy).
- `fedgraphrl.data` — synthetic transaction-graph generator with planted fraud
  rings and non-IID Dirichlet sharding across simulated clients.
- `fedgraphrl.gnn` — 2-layer GCN node classifier + SGD.
- `fedgraphrl.federated` — `FederatedClient`, `FederatedServer`, compute-weighted
  `fedavg`, validation-tuned decision threshold.
- `fedgraphrl.environment` — `FederatedEnv`: the federated training loop wrapped
  as an RL environment, with a soft per-round cost constraint.
- `fedgraphrl.rl_controller` — `PolicyNet`, `ValueNet`, `ReinforceController`
  (REINFORCE + multi-rollout averaging + optional actor–critic), and
  `HeuristicController` baselines.
- `experiments/run_experiment.py` — single-seed experiment + plot.
- `experiments/sweep.py` — multi-seed evaluation with paired statistics + plot.
- `tests/test_autograd.py` — finite-difference gradient check + federated-round
  smoke test.
- `docs/TRD.md`, `docs/WORKFLOW.md`, `docs/DESIGN_LOG.md`.

### Key findings (5 seeds, α = 0.10, actor–critic, 100 updates)
- RL controller beats a **fixed cohort** by +0.11 F1 on 5/5 seeds.
- RL controller **ties uniform-random** selection (Δ −0.017 F1, 2/5 seeds) and a
  **fraud-rate heuristic** (Δ +0.005 F1, 1/5 seeds).
- RL controller trails the **centralised upper bound** by ~0.10 F1.
- Multi-rollout averaging and the actor–critic baseline stabilise training but do
  not change the final ranking — gradient variance is not the bottleneck.

### Notes
- Pure NumPy (Matplotlib optional). No PyTorch — Python 3.14 lacked wheels, and a
  small autograd engine was sufficient.
