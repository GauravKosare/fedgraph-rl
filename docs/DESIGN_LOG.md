# Design log — every decision, and why

Plain‑language record of how FedGraph‑RL was built and tuned, in the order it
happened. Each entry: **what** was decided, **why** (simple terms), **what it
changed**.

---

## 1. Why combine a GNN, federated learning, and RL at all?

**Decision.** Build one system where a **Graph Neural Network** is the model, it
is trained by **Federated Learning**, and a **Reinforcement Learning** agent
decides the FL schedule.

**Why.** Each paradigm covers a real gap the others leave:

- Fraud rings are a *graph* pattern (who transacts with whom). A per‑account model
  can't see the ring; a **GNN** can, by passing messages along edges.
- The transactions that reveal the ring are split across banks that *can't pool
  data*. **Federated learning** trains the shared GNN without moving raw data.
- Federated learning forces a choice every round — *which* banks train now, and
  *how hard*. That's a sequential decision under a cost budget, which is exactly
  what **RL** is for.

So the three aren't bolted together for show; the GNN needs FL to get enough
data, and FL needs a selection policy that RL can, in principle, learn.

**Changed.** Defined the whole project scope.

---

## 2. No PyTorch — a hand‑written autograd engine

**Decision.** Implement a ~150‑line reverse‑mode autograd engine over NumPy and
build the GCN on it. ([`fedgraphrl/autograd.py`](../fedgraphrl/autograd.py))

**Why.** The target machine has Python 3.14 and **no PyTorch wheels available**.
Rewriting later would be churn. A tiny autograd engine is enough for a 2‑layer
GCN, keeps the dependency list to "NumPy", and makes every gradient inspectable.

**Changed.** Everything downstream is NumPy. Runs anywhere in minutes; nothing is
a black box. Cost: no GPU, no big models — fine for a methods study.

**Guard.** A finite‑difference gradient check
(`tests/test_autograd.py::test_matmul_relu_gradcheck`) pins the engine to
correctness (agreement to 1e‑4).

---

## 3. Synthetic data with *planted* fraud rings

**Decision.** Generate the transaction graph procedurally: dense fraud rings +
mule bridges + a sparse legit background, with node features = a **weak, noisy**
fraud leak plus graph‑derived signal.

**Why.** No real transaction data (privacy, and none needed). "Planted" structure
means we *know* the ground truth pattern the GNN should find. The signal is
deliberately weak so the task isn't trivially solvable without the graph.

**Changed.** Gave a controllable difficulty dial. First attempt made the leak too
strong (feature ≈ label) → every method scored F1 ≈ 1.0 and nothing could be
compared. Weakened the leak (`signal[:,1] = label · N(0.45, 0.9)`); centralised
F1 dropped to ~0.70, leaving headroom in both directions.

---

## 4. Non‑IID sharding via a Dirichlet split

**Decision.** Assign each fraud ring's members to clients with a `Dirichlet(α)`
draw (low α ⇒ a ring lands mostly on one or two clients); spread legit accounts
uniformly. ([`partition_non_iid`](../fedgraphrl/data.py))

**Why.** If every client saw the same mix, any client could learn a good detector
alone and FL would be pointless. Concentrating rings makes clients *complementary*
— the whole point of federating — and gives a selection policy something to reason
about ("client 5 is the only one who has seen ring 3").

**Changed.** Created the core tension the RL agent is supposed to exploit.

---

## 5. Compute‑weighted FedAvg

**Decision.** The server averages client weights not just by sample count but also
by the compute the RL agent spent on each client this round
(`fedavg(reports, mix=compute_cost)`).

**Why.** If the controller decides to invest more local epochs in a client, the
server should trust that client's update more. It also gives the *budget* half of
the action a direct effect on the model, not just on cost.

**Changed.** Made "how many epochs per client" a meaningful lever rather than
cosmetic.

---

## 6. Framing FL as an RL environment

**Decision.** One **episode** = train a fresh global GCN over 25 federated rounds.
One **step** = one round: the agent picks `k = 4` clients and splits a 12‑epoch
budget among them. Reward = change in validation F1 × 100, minus a cost term.
([`FederatedEnv`](../fedgraphrl/environment.py))

**Why.** This is the smallest wrapper that turns "client selection" into a
standard RL problem with a clear state, action, and reward. Scaling F1 by 100
puts per‑round rewards on a sane numeric range for REINFORCE.

**Action mechanics.** `k` clients are drawn *without replacement* from a softmax
over per‑client "selection logits" (a Plackett–Luce sample — differentiable‑ish
and gives a usable log‑probability); the epoch budget is split by a softmax over
per‑client "budget logits". Both logits come from one shared 2‑layer MLP applied
per client row.

---

## 7. Multi‑rollout gradient averaging

**Problem.** With one trajectory per update, the learning curve swung wildly —
return jumping between −8 and +46 on adjacent updates, final policy quality
basically down to luck.

**Decision.** Collect **4 independent rollouts** per update, share one baseline
across the batch, average all gradients, then take a single step.
([`ReinforceController.run_episode`](../fedgraphrl/rl_controller.py))

**Why.** Standard variance reduction. A stochastic FL run is a very noisy reward
signal; averaging several cuts the noise by ~√4.

**Changed.** Learning curve got visibly smoother and training‑time val‑F1 rose
higher and more steadily. **But** the final *evaluated* policy was about the same.
First hint that variance wasn't the real problem.

---

## 8. Validation‑tuned decision threshold

**Problem.** One run reported RL test **F1 = 0.000** with **AUC = 0.73** — the
model *ranked* fraud reasonably but the fixed 0.5 cutoff put almost nothing over
the line. F1 was measuring the threshold, not the model.

**Decision.** After training, pick the probability threshold that maximises F1 on
the **validation** split, then score the test set at that threshold — for *every*
policy, including baselines and the centralised bound.
([`FederatedServer.tuned_threshold`](../fedgraphrl/federated.py))

**Why.** 0.5 is arbitrary for imbalanced classes. The fair question is "best F1
each model can reach at its best operating point", and that threshold must be
chosen without touching test data.

**Changed.** Every policy's F1 jumped ~0.1–0.5. The *ranking* between policies
stayed the same. Confirmed the earlier swings (e.g. a baseline going 0.25 → 0.79
between runs) were threshold artifacts, not method differences.

---

## 9. Multi‑seed sweep with paired statistics

**Decision.** Stop drawing conclusions from one graph. Run the full pipeline over
5 graph seeds `{7,107,207,307,407}` and report per‑policy **mean ± std** plus
**paired** per‑seed `RL − baseline` deltas and win counts.
([`experiments/sweep.py`](../experiments/sweep.py))

**Why.** Single‑seed numbers were bouncing ±0.15 F1. A method that "wins" on one
seed and loses on the next hasn't won. Paired comparison (same graph, RL vs
baseline) removes graph‑difficulty as a confounder.

**Changed.** This is where the honest picture appeared:

| | RL F1 | random F1 | fraud_greedy F1 |
|---|---|---|---|
| 5‑seed mean ± std | 0.64 ± 0.09 | 0.61 ± 0.08 | 0.64 ± 0.10 |

RL beat the **fixed cohort** 4–5/5, but tied **random** and **fraud_greedy**.

---

## 10. Harder non‑IID + bigger cost penalty — *this backfired*

**Decision (reverted).** Push `dirichlet_alpha` 0.15 → 0.05 (rings almost entirely
on single clients) and `cost_coeff` 0.02 → 0.06.

**Why (the hope).** More extreme skew + a cost that actually bites should make
smart scheduling pay off where naive selection can't.

**What happened.** RL got **worse** — lost to random on 4/5 seeds. The tripled
cost coefficient made the reward `ΔF1·100 − 0.06·cost` dominated by cost, so the
policy learned to be *stingy* (few epochs, cheap clients) at the expense of
accuracy. The heuristics ignore the cost term entirely, so they were unaffected
and pulled ahead.

**Lesson.** A cost penalty that scales with *total* spend punishes the agent for
working, not for overworking. Also: at α = 0.05, measured ring concentration was
75–100 % on one client — *when* you schedule that client is irrelevant, so there
was even less for a selection policy to exploit.

---

## 11. Soft cost constraint

**Decision.** Change the reward to
`ΔF1·100 − cost_coeff · max(0, round_cost − cost_budget)`, where `cost_budget` is
auto‑calibrated to the cost of an *average full round* (× 1.2 slack).
([`FederatedEnv.__init__` / `.step`](../fedgraphrl/environment.py))

**Why.** Cost should be a *constraint*, not an objective. Inside a reasonable
per‑round budget the agent should optimise accuracy freely; only genuine
overspend should be penalised.

**Changed.** Verified the overshoot penalty is **zero in ~100 % of rounds** at the
current settings — i.e. reward is now effectively pure ΔF1. This removed the
damage from §10 without reintroducing a bias. Ranking still unchanged (RL ≈
random), which now clearly points at the *task*, not the *reward*.

---

## 12. α = 0.10 sweet spot + 100 updates

**Decision.** Set `dirichlet_alpha = 0.10` (rings split across 2–3 clients, so
rotation genuinely helps) and train for **100** updates per seed instead of 30.

**Why.** A middle ground: skewed enough that scheduling matters, not so skewed
that it can't. More updates in case 30 was just undertrained.

**Changed.** Per‑seed std **rose** (±0.09 → ±0.15). More plain‑REINFORCE updates
drifted about as much as they learned. Still no win over random.

---

## 13. Actor–critic (learned value baseline)

**Decision.** Add [`ValueNet`](../fedgraphrl/rl_controller.py): a small MLP over a
permutation‑invariant pooling of the state (global features + mean/max/min of the
per‑client dynamic features → 15 dims), trained by MSE to the observed returns,
used as the advantage baseline instead of the scalar moving average.

**Why.** When REINFORCE‑with‑a‑scalar‑baseline underperforms, the standard next
step is a *state‑dependent* baseline (A2C). If gradient variance were the
bottleneck, this should help.

**Changed.** **Nothing measurable.** RL F1 0.603 ± 0.154 vs 0.605 ± 0.153 with the
old baseline; identical win counts. This is the decisive negative: variance
reduction — the one structural lever left — moved the needle zero.

---

## 14. Conclusion: stop, and report the null result

**Decision.** Freeze the experiment at `v0.1.0`, write everything up honestly, and
*not* keep tuning.

**Why.** Across §7–13 every plausible lever was pulled — variance reduction (twice),
reward shaping (twice), difficulty (three settings), training length. The result
is stable: **a learned scheduler reliably beats a fixed cohort but ties
uniform‑random selection and a fraud‑rate heuristic on this task.**

That is a legitimate finding. Random client sampling being hard to beat is
well‑documented in federated learning; this sandbox reproduces it cleanly on a
graph task and shows that neither an actor–critic nor reward tweaks change it.

**What would change the answer** is a different *environment*, not a different
*algorithm*: add client stragglers, wall‑clock deadlines, or concept drift so that
*timing* carries reward. Those variants are specced in [`TRD.md`](TRD.md) §9 as
`v0.2+`.

---

## 15. v0.2 — stragglers and a wall‑clock deadline

**Decision.** Add the first `v0.2` environment change: `straggler_frac = 0.35` of
clients run `4×` slower; each round has a deadline (`1.15 ×` a fast client's
fair‑share time); a selected client that can't finish its assigned epochs by the
deadline still burns compute but its **update is dropped** from FedAvg. Device
speed is added as a 5th per‑client state feature. Toggle: `stragglers` (default
**off** — the v0.1 result stays canonical).

**Why.** §14 concluded the missing ingredient was *timing*. If the agent is
punished (via dropped rounds) for dumping epochs on a slow client, and it can see
which clients are slow, then "route work around stragglers while still covering
their rings" is a real skill that random selection does not have.

**What happened (5 seeds).**

| | RL F1 | random F1 | fraud_greedy F1 | Δ(RL−random) |
|---|---|---|---|---|
| v0.1 | 0.60 ± 0.15 | 0.62 ± 0.17 | 0.60 ± 0.14 | −0.017 (2/5) |
| **v0.2 stragglers** | **0.48 ± 0.19** | 0.44 ± 0.12 | 0.51 ± 0.06 | **+0.037 (3/5)** |

- The result **moved in RL's favour** — from losing to random to edging it on the
  majority of seeds. On seed 7 it is decisive (0.73 vs 0.51). The device‑speed
  feature is doing something.
- But +0.037 is still **inside the ±0.19 noise**, so this is a lean, not a win.
- `fraud_greedy` became the strongest baseline by virtue of being *stable*
  (±0.06). RL's low mean is a variance problem, not a floor problem.
- **New failure mode:** seed 207 collapsed (F1 0.196, AUC 0.499). Dropped rounds
  make the reward sparser; plain REINFORCE sometimes never recovers.
- Stragglers cost every FL policy ~0.15 F1; the centralised bound (no stragglers)
  was unchanged — straggler *mitigation* matters more than selection cleverness.

**Lesson / next step.** The environment hypothesis from §14 is directionally
correct but under‑powered as built. Before `v0.2` can claim a positive RL result
it needs the instability fixed — candidates: a small explicit per‑drop reward
penalty (denser signal), entropy annealing, or more rollouts on the sparse‑reward
seeds. Tracked as `v0.2.1`.

---

## 16. v0.2.1 — fixing the straggler‑variant training collapse

**Problem.** Under stragglers (§15), seed 207 trained to a degenerate model
(F1 0.196, AUC ≈ 0.5 — no better than chance). Diagnosis had three parts:

1. **All‑or‑nothing drops made slow clients unusable.** With `slowdown = 4×` and
   `deadline_slack = 1.15`, a slow client could not finish *even one* epoch before
   the deadline — so it was always dropped. The only way to use its data was to
   never select it, which on seed 207 meant never seeing the fraud rings those
   clients held. The global model then collapsed to predicting "legit"
   everywhere.
2. **A collapsed model gives a zero reward signal.** Reward was `ΔF1` at the fixed
   0.5 threshold. An all‑legit model scores F1 = 0 *every round* → `ΔF1 = 0` →
   the policy gradient sees nothing but noise.
3. **Advantage normalisation amplified that noise.** Dividing near‑identical
   returns by a tiny standard deviation turns rounding noise into large spurious
   gradients — which is how a run *actively* diverges rather than just stalling.

**Fixes (all behind config knobs; `stragglers=False` keeps v0.1 untouched):**

| Fix | What it does |
|---|---|
| **Partial participation** | A selected client now trains the number of epochs it can *finish* by the deadline (`floor(deadline · speed / cost‑per‑epoch)`); it is only *dropped* if that is < 1. Its partial update is aggregated, compute‑weighted by epochs actually done. Assigning a slow client more epochs than it can finish just wastes budget — a lever the policy can learn, not a wall. |
| **Gentler straggler params** | `slowdown 4 → 3`, `deadline_slack 1.15 → 1.6`, so a slow client contributes ~1–2 real epochs instead of zero. |
| **Tuned‑threshold reward** (straggler mode only) | Reward is now `ΔF1` at the *best validation threshold*, not at 0.5. A conservative‑but‑ranking model scores > 0, so `ΔF1` stays informative. (Vectorised `best_f1_threshold` — also ~40× faster than the old unique‑value scan, which matters now that it runs every round.) |
| **Dense per‑drop penalty** | `reward −= drop_penalty · n_dropped` (`drop_penalty = 0.6`). Immediate feedback for wasting a slot on a straggler, even when F1 doesn't move. |
| **Advantage‑std floor + clip** | Normalisation denominator floored at `adv_std_floor = 1.0`; normalised advantages clipped to `± adv_clip = 8`. Stops the divide‑by‑tiny‑std blow‑up. |
| **Entropy annealing** | `entropy_coeff` decays linearly to 10 % of its start value over training — explore early, exploit cleanly late. |

**Result.** Seed 207 no longer collapses (train val‑F1 holds ~0.35 instead of
0.00; test F1 0.39 vs the old 0.20, and it now *beats* random on that seed). Full
5‑seed numbers: README §4.2.

**Trade‑off noted.** The tuned‑threshold reward and the stabilisation knobs are a
mild change to the learning dynamics, so the v0.1 canonical sweep was re‑run with
them too (reward still fixed‑0.5 on the no‑straggler path); the v0.1 conclusion is
unchanged.

---

## 17. v0.3 — the problem reframing (payment‑flow data + money‑weighted reward)

**Decision.** Stop iterating on the RL algorithm. Replace the *problem*:

- **`payment_data.py`** — a directed **payment‑flow** graph shaped like a real APP
  scam: `victim → 1st‑hop mule → layering mules → cash‑out`. Labels mark the
  mule/layering/cash‑out accounts (what the *receiving* bank must catch); the
  victim is not fraud. Two of six banks are "high‑risk" and receive ~75 % of mule
  accounts; ~10 % of legit accounts are high‑throughput "merchant/payroll"
  decoys that look mule‑like. Features are the ones a bank actually has (in/out
  amount & degree, distinct senders, flow‑through ratio, account age) with heavy
  jitter so no single feature separates.
- **`partition_by_bank`** — the federated split **is** account ownership. The
  victim and the first‑hop mule are at different banks ~90 % of the time, so the
  non‑IID structure is dictated, not tuned with a Dirichlet knob.
- **Money‑weighted reward & metric** (`money_weighted_scores`) — every mule node
  carries `amount_at_risk` (catching the first hop protects the whole episode;
  catching a downstream node protects only its slice). The metric is
  **money‑recall at a false‑positive budget**: £ at risk on caught mules ÷ total
  £ at risk, evaluated at the threshold that maximises catches while keeping the
  frozen‑legit‑account rate ≤ `fp_budget` (5 %). This is the number a bank's
  fraud‑ops lead and a regulator actually ask for.

**Why.** v0.1–v0.2.1 established that on a generic graph with an F1 reward, learned
client selection ties random. The open question was whether a *realistic* problem
— real non‑IID structure, a cost that is money not F1‑points, decoys that create
genuine false‑positive pressure — changes that.

**Result (5 seeds).** The headline did **not** change: learned orchestration is an
**exact tie with uniform‑random** (money‑recall 0.639 ± 0.052 vs 0.639 ± 0.071 —
RL slightly more consistent), and both beat the fixed‑cohort strategies
(`fraud_greedy` / `all`, 0.508) by **+0.13 money‑recall, 5/5 seeds** — picking
only the two high‑risk banks every round misses the layering mules deliberately
spread to ordinary banks. The learning curve is flat: with 2‑of‑6 banks per round
over 25 rounds, coverage is not scarce enough for scheduling to matter.

Note: **the pooled‑data model is not an upper bound here** — per‑bank subgraph
training + FedAvg regularisation ranks mules *better* than one GCN over the whole
noisy graph. Renamed `centralised_pooled`.

**What v0.3 actually delivers:** a benchmark that is defensible as a model of the
real problem (`TARGET_PROBLEM.md`), with money‑weighted evaluation. The RL result
is the same honest tie — which is itself the finding: **a learned scheduler is
not the lever; the environment has to make timing/coverage genuinely scarce**.

---

## 18. v0.3.1 — the scarce‑coverage regime (`--scarce`)

**Decision.** Test the §17 hypothesis directly. Same payment‑flow task and
money‑weighted reward, but coverage is now genuinely scarce:

- **8 federated rounds** instead of 25. With 2‑of‑6 banks per round that is ~2.7
  visits per bank — random selection *will*, by chance, leave one or two banks
  barely trained, and the layering mules those banks hold go undetected.
- **Stragglers on** — some banks run on slower infrastructure and contribute only
  partial updates, so *which* banks you schedule, and *when*, matters more.

`python experiments/run_payment_experiment.py 5 --scarce` → `*_scarce` artifacts.

**Result (5 seeds).** The hypothesis **failed**. Even here — a hard 8‑round
budget, stragglers on, non‑IID by bank — REINFORCE client selection does **not**
beat uniform‑random:

| | money‑recall | vs random |
|---|---|---|
| RL controller | 0.552 ± 0.101 | −0.020, RL wins 3/5 |
| Uniform‑random | 0.573 ± 0.101 | — |
| `fraud_greedy` / `all` | 0.271 ± 0.134 | +0.282, RL wins 5/5 |

Scarcity mostly just **added variance** (both policies ±0.10, up from ±0.05–0.07):
RL crushes random on seed 207 (+0.18) and loses badly on seeds 307/407 (−0.12,
−0.19). The mean is a tie, slightly the wrong way. `fraud_greedy` / `all` collapse
as expected — 8 rounds on 2 banks trains almost nothing.

Why random holds: money‑recall is dominated by catching the high‑value first‑hop
mules, which live at banks 0–1; random hits those two banks ~2.7 times each in 8
rounds, which is enough. Leaving an *ordinary* bank untrained costs only the
lower‑value layering mules. RL never finds a schedule that reliably beats "hit
the big banks often, which random already does."

**This is the strongest form of the project's null result:** the regime was
hand‑built to be the one where scheduling matters most, and REINFORCE client
selection *still* ties random. The lever is not a cleverer environment knob —
it is a different method (a policy with memory of per‑bank coverage; or a
non‑learned coverage‑guaranteeing heuristic, which would likely beat both).
