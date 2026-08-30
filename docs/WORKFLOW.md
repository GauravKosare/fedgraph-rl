# Workflow — FedGraph‑RL

How the pieces run, end to end, with diagrams. Read alongside
[`TRD.md`](TRD.md) (what/why) and [`DESIGN_LOG.md`](DESIGN_LOG.md) (decisions).

---

## 1. The big picture

```mermaid
flowchart TD
    A["1 · Generate synthetic transaction graph<br/>1,200 accounts · 12 planted fraud rings"] --> B
    B["2 · Split graph across 8 edge devices<br/>non-IID (Dirichlet α = 0.10)"] --> C
    C["3 · Build FederatedEnv<br/>wraps the whole FL training loop as one RL episode"] --> D
    D["4 · Train RL controller<br/>100 updates × 4 rollouts (actor–critic)"] --> E
    E["5 · Evaluate trained policy<br/>mean over 7 sampled rollouts, val-tuned threshold"] --> F
    F["6 · Run baselines on the same graph<br/>random · fraud_greedy · all · centralised"] --> G
    G["7 · Aggregate over 5 graph seeds<br/>mean ± std + paired per-seed deltas"] --> H
    H["8 · Write results.json / sweep_results.json + PNG plots"]
```

Entry points: `experiments/run_experiment.py` (steps 1–6, one seed),
`experiments/sweep.py` (steps 1–8, all seeds).

---

## 2. One federated round = one RL step

```mermaid
sequenceDiagram
    participant PI as RL controller π
    participant ENV as FederatedEnv
    participant C as Selected clients (k of 8)
    participant SRV as Server (FedAvg)

    ENV->>PI: state  (8 clients × 7 features)
    Note right of PI: global: round #, last F1, progress<br/>per client: since-selected, last loss,<br/>shard fraud rate, shard size
    PI->>PI: selection logits + budget logits (2-layer MLP)
    PI->>ENV: action = (k client ids, epochs per client)
    Note right of PI: k picked without replacement (Plackett–Luce)<br/>epoch_budget split by softmax

    loop for each selected client
        ENV->>C: global weights + local epoch count
        C->>C: e epochs, class-weighted cross-entropy on local shard
        C->>ENV: updated weights + compute_cost + comm_cost
    end

    ENV->>SRV: client reports
    Note right of ENV: v0.2: a client whose epochs / speed<br/>exceeds the round deadline is DROPPED<br/>(compute still charged as cost)
    SRV->>SRV: compute-weighted FedAvg (on-time updates only)
    SRV->>ENV: new global GCN
    ENV->>ENV: val F1 → reward = ΔF1·100 − cost_coeff·max(0, cost − budget)
    ENV->>PI: (next state, reward, done)
    Note over PI: after 25 rounds: done.<br/>tune threshold on val, score test set
```

---

## 3. How the RL controller learns (one update)

```mermaid
flowchart TD
    S["Collect 4 independent rollouts<br/>(each = 25 rounds, stochastic)"] --> R
    R["For every step: discounted return-to-go G_t"] --> V
    V{"use_critic ?"}
    V -->|"yes (default)"| VC["V(s_t) = ValueNet(pooled state)<br/>advantage A_t = G_t − V(s_t)<br/>fit V to G by MSE"]
    V -->|"no"| VB["advantage A_t = G_t − moving-average baseline"]
    VC --> N
    VB --> N
    N["Normalise advantages across the batch"] --> P
    P["Policy gradient:<br/>∇ log π(a_t | s_t) · A_t  + entropy bonus"] --> AVG
    AVG["Average gradients over all 4 rollouts × 25 steps"] --> STEP
    STEP["One clipped SGD step on PolicyNet<br/>(and one on ValueNet)"]
```

**Why this shape:** a single REINFORCE trajectory through a stochastic FL run is
extremely noisy. Averaging 4 rollouts and subtracting a learned value baseline are
the two textbook variance‑reduction moves. (Result: they stabilise the *learning
curve* but do not change the *final ranking* — see DESIGN_LOG §7–8.)

---

## 4. Evaluation & aggregation

```mermaid
flowchart LR
    subgraph PERSEED["for seed in {7,107,207,307,407}"]
        G1["graph + shards"] --> T1["train RL controller"]
        T1 --> P1["policy test F1/AUC<br/>(mean of 7 sampled rollouts)"]
        G1 --> B1["random / fraud_greedy / all<br/>(1 episode each)"]
        G1 --> U1["centralised upper bound"]
    end
    P1 --> AGG
    B1 --> AGG
    U1 --> AGG
    AGG["per-policy mean ± std<br/>paired: RL − baseline per seed, win counts"] --> OUT["sweep_results.json + .png"]
```

Primary metric **F1** (imbalanced classes); secondary **AUC** (threshold‑free).
All test scores use the threshold that maximises **validation** F1 for that model.

---

## 5. Developer workflow

```bash
# 1. install
python -m pip install -r requirements.txt

# 2. verify the autograd + a federated round
python tests/test_autograd.py            # prints "ok"

# 3. iterate on one seed (fast feedback, ~70 s)
python experiments/run_experiment.py
#    -> experiments/results.json, experiments/results.png

# 4. change hyper-parameters
$EDITOR fedgraphrl/config.py             # alpha, cost_coeff, episodes, use_critic, ...

# 5. full evaluation before committing a claim (~12 min)
python experiments/sweep.py 5
#    -> experiments/sweep_results.json, experiments/sweep_results.png

# 6. commit code + refreshed artifacts together
git add -A && git commit -m "..."
```

### Which knob does what (`fedgraphrl/config.py`)

| Knob | Meaning | Effect of increasing |
|---|---|---|
| `dirichlet_alpha` | non‑IID severity | ↑ = shards more IID (easier for FL, less for RL to exploit) |
| `clients_per_round` | `k` in the action | ↑ = more coverage per round, more cost |
| `epoch_budget` | total local epochs to split each round | ↑ = more local work, faster convergence, more cost |
| `cost_coeff` | overshoot penalty weight | ↑ = agent punished harder for exceeding the round budget |
| `episodes` | RL updates per seed | ↑ = more training (also more drift with plain REINFORCE) |
| `rollouts_per_update` | trajectories averaged per update | ↑ = lower‑variance gradient, linearly more compute |
| `use_critic` | learned `V(s)` vs moving‑average baseline | actor–critic on/off |
| `stragglers` *(v0.2)* | heterogeneous device speeds + deadline; late updates dropped | on = timing matters; the policy sees device speed |
| `straggler_frac` / `straggler_slowdown` | how many clients are slow, and by how much | ↑ = more/worse stragglers to route around |
| `deadline_slack` | round deadline ÷ a fast client's fair‑share time | ↓ = tighter deadline, more drops |
