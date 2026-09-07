# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] — unreleased

The problem reframing (see `docs/DESIGN_LOG.md` §17). The RL algorithm is
unchanged — the *problem* is now a realistic model of federated mule-account
detection.

### Added
- **`fedgraphrl/payment_data.py`** — directed payment-flow graph generator:
  `victim → 1st-hop mule → layering mules → cash-out`. Two of `n_banks` are
  high-risk (receive ~75% of mule accounts); ~10% of legit accounts are
  high-throughput "merchant/payroll" decoys. Bank-ownership node features with
  heavy jitter. `partition_by_bank` — the non-IID split is ownership itself.
- **`GraphData.amount_at_risk` / `.bank_of`** — optional fields, `None` under the
  v0.1/v0.2 "rings" model; carried through `subgraph`.
- **`metrics.money_weighted_scores`** and **`best_threshold_at_fp`** — money-recall
  (£ at risk on caught mules ÷ total £ at risk) at a false-positive-rate budget.
- **`FederatedServer.evaluate_money` / `val_money_recall` / `tuned_threshold_money`**.
- **`FederatedEnv(reward_mode="money", fp_budget=…)`** — reward = Δ(val
  money-recall at the FP budget) × 100. `reward_mode="f1"` (default) is unchanged.
- **`experiments/run_payment_experiment.py`** — the v0.3 experiment + sweep.
- **`Config`**: `data_model`, `n_banks`, `n_scam_episodes`, `reward_mode`,
  `fp_budget`.
- **`docs/AI_MLOPS_CLOUD.md`** — plan (not built) for the GenAI layer, MLOps
  plumbing, and cloud deployment.

### Results (5 seeds, `run_payment_experiment.py 5`)
- **Exact tie** with uniform-random on money-recall: RL 0.639 ± 0.052 vs random
  0.639 ± 0.071 (Δ −0.000, RL wins 3/5; RL slightly more consistent).
- RL **+0.131 money-recall, wins 5/5** vs both fixed-cohort strategies
  (`fraud_greedy` / `all`, 0.508 ± 0.086) — "always pick the two high-risk banks"
  misses the layering mules spread across ordinary banks.
- Flat learning curve — 2-of-6 banks/round over 25 rounds is not scarce enough
  for scheduling to matter. `docs/TARGET_PROBLEM.md` lists v0.3.1 tighteners.
- `centralised_pooled` 0.262 ± 0.144 is **not an upper bound** — a single GCN over
  the whole noisy graph is a weaker ranker (AUC ~0.78) than per-bank training +
  FedAvg (AUC ~0.87); improving it is future work.

### Unchanged
- v0.1 / v0.2 / v0.2.1 code paths and their committed artifacts are untouched
  (`data_model="rings"`, `reward_mode="f1"`).

---

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
