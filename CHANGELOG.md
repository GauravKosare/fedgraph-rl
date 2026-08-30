# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — unreleased

### Added
- **Straggler / deadline environment variant** (`FederatedEnv`, `Config`).
  Clients now have heterogeneous device speeds (`straggler_frac` of them run
  `straggler_slowdown`× slower). Each round has a wall-clock deadline
  (`deadline_slack` × a fast client's fair-share time). A selected client whose
  `epochs / speed` work time exceeds the deadline still burns compute (counted as
  cost) but its update is **dropped** from FedAvg. If every selected client is
  late, the round is wasted.
- Device speed added as a 5th per-client state feature; `feature_dim` 7 → 8.
  `ValueNet` input dimension now derived from `feature_dim` instead of hard-coded.
- `info["dropped"]` — count of late/dropped updates per round.
- `stragglers=False` recovers exact v0.1 behaviour.

### Rationale
v0.1 showed the RL controller could not beat uniform-random client selection
because the task had no timing structure to exploit. Stragglers + a deadline make
*when* and *how hard* each client trains genuinely consequential — the policy sees
device speed and can avoid wasting rounds on slow clients while still covering the
fraud rings they hold.

### Results (5 seeds, `sweep.py 5 --stragglers`)
- RL vs uniform‑random: **+0.037 F1, wins 3/5 seeds** (v0.1 was −0.017, 2/5) —
  timing pressure moves the result toward RL but stays within cross‑seed noise.
- RL 0.481 ± 0.190 · random 0.444 ± 0.124 · fraud_greedy 0.509 ± 0.057.
- New instability: one seed's RL training collapsed (F1 0.196). Tracked as v0.2.1.
- All FL policies lost ~0.15 F1 vs v0.1; centralised bound (no stragglers) unchanged.
See README §4.2 and `experiments/*_stragglers.*`.

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
