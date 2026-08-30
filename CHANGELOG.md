# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

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
