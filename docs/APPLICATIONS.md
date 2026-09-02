# Real‑world applications

Where the ideas in FedGraph‑RL show up in industry, how you would actually deploy
them, and — just as important — when you should *not*.

FedGraph‑RL is a teaching/experiment sandbox with synthetic data. But it is a
small model of three things that are genuinely used in production, at different
levels of maturity:

| Layer | Maturity in industry | Who uses it |
|---|---|---|
| **GNN on a transaction / entity graph** for fraud & anomaly detection | **Production‑proven**, many vendors and in‑house systems | Payments, banking, marketplaces, insurance, telco, ad‑tech |
| **Federated learning across institutions** (train a shared model, never move raw data) | **Real but early** — mostly consortium pilots and regulated‑industry deployments | Bank anti‑fraud consortia, healthcare, telco |
| **Learned (RL) orchestration of *which* clients train each round** | **Research** — this repo's own 5‑seed result says the payoff is marginal unless the environment has hard timing constraints | FL researchers; a handful of cross‑device FL teams |

---

## 1. The whole pattern: federated graph anomaly detection across organisations

### The recurring real‑world problem

A criminal network (a money‑laundering ring, a bust‑out fraud ring, a bot farm)
operates **across several organisations at once**. Each organisation sees only its
own slice:

- **No single party sees the whole ring** — so per‑party models miss it.
- **The parties cannot pool raw data** — privacy law (GDPR, GLBA, HIPAA),
  competition concerns, and data residency all forbid it.
- The signal that reveals the ring is **relational** — who paid whom, who shares a
  device/address/beneficiary — i.e. a graph.

That is exactly the shape FedGraph‑RL models: a global graph, sharded non‑IID
across clients, a GNN that needs message passing to see the ring, and FedAvg so no
raw data leaves a client.

### Concrete industry scenarios

| Industry | The graph | The ring you're chasing | Why federated |
|---|---|---|---|
| **Anti‑money‑laundering (AML) between banks** | Accounts + inter‑bank transfers + shared beneficiaries | Layering chains, mule networks, trade‑based laundering that hops banks | Banks legally cannot share customer transaction data; regulators (FinCEN, FCA, FATF) now actively encourage privacy‑preserving information sharing |
| **Card‑payment fraud (issuer ↔ acquirer ↔ network)** | Cards, merchants, terminals, devices | Testing/enumeration attacks, collusive merchants, coordinated CNP fraud | Issuers, acquirers and the network each hold a different projection of the same fraud |
| **Authorised Push Payment (APP) / scam networks** | Payer accounts → beneficiary "receiving" accounts across banks | Scam beneficiary accounts that receive from victims at many banks | The receiving bank often has the strongest signal about a mule account it didn't onboard |
| **Telecom fraud between operators** | Subscribers, calls, SIMs, IMEIs | SIM‑box / interconnect bypass, subscription‑fraud rings, IRSF | Fraud spans operators and countries; CDRs are sensitive and jurisdiction‑bound |
| **Insurance fraud rings** | Claimants, providers, vehicles, addresses, adjusters | Staged‑accident rings, provider‑collusion, identity farms reused across insurers | Insurers compete but share fraud exposure; bureaus exist but are limited |
| **Marketplace / e‑commerce fraud across platforms** | Buyers, sellers, payment instruments, shipping addresses, devices | Account‑takeover rings, refund abuse, seller collusion, triangulation fraud | Platforms won't share user data but face the same actors |
| **Ad fraud / bot networks across exchanges** | Devices, IPs, publishers, campaigns | Coordinated invalid traffic, domain spoofing, install farms | Each exchange sees part of the botnet's activity |
| **Healthcare billing fraud / rare‑disease cohorts across hospitals** | Patients, providers, procedures, referrals | Upcoding/kickback provider rings; also non‑fraud: rare‑disease patient discovery | Patient data cannot leave the hospital; FL is already standard here |

### Named reference points (as of early 2026)

- **Graph analytics for AML without FL:** Quantexa, Palantir Foundry, TigerGraph,
  Neo4j, Amazon Neptune + GraphStorm, Google Cloud's AML AI. These prove the
  *graph* half at scale — FedGraph‑RL's GNN is a miniature of this.
- **Cross‑institution federated learning for financial crime:** SWIFT's federated
  learning pilots for cross‑border fraud; consortium efforts under the UK Economic
  Crime Plan and Singapore's COSMIC platform; vendors like Consilient and
  Lucinity. These prove the *federation* half.
- **FL infrastructure you would actually build on:** Flower, NVIDIA FLARE,
  TensorFlow Federated, OpenFL, Owkin / Rhino Health (healthcare).

---

## 2. The individual pieces, used on their own

You rarely need all three layers. Most real deployments use one or two:

### 2a. GNN transaction‑graph fraud detection — *inside a single company*

No federation at all. A bank or PSP builds one big graph of its own
accounts/cards/devices/merchants and runs a GNN (usually GraphSAGE or a
heterogeneous GNN, not a plain GCN) to score entities. This is mainstream:
PayPal, Stripe, and fraud platforms such as Feedzai, Featurespace (Visa),
and DataVisor all use graph‑based features or full GNNs.

*What transfers from this repo:* the modelling frame (nodes = entities, edges =
interactions, label = known fraud, message passing surfaces ring structure) and
the evaluation discipline (class‑imbalanced F1, AUC, validation‑tuned threshold).

### 2b. Federated learning client selection — *cross‑device*

Google's Gboard next‑word prediction, Apple's on‑device models, and similar
mobile deployments train across millions of phones. The server must pick which
devices participate each round — but they select mainly on **availability**
(plugged in, on Wi‑Fi, idle), not on a learned policy. FedGraph‑RL's
`FederatedEnv` is a compact testbed for research into smarter selection:
staleness‑aware, fairness‑aware, or cost‑aware sampling.

*What transfers:* the environment abstraction — one round = one RL step, state =
per‑client descriptors, reward = global metric gain − communication/compute cost.

### 2c. The negative result itself is useful

FedGraph‑RL's headline finding — *a learned client‑selection policy ties
uniform‑random sampling unless the environment has hard deadlines/stragglers* —
is a real decision input. If you are scoping an FL platform, this says: **ship
random (or availability‑based) sampling first; only invest in a learned
orchestrator if you have measured straggler pain or a wall‑clock SLA per round**
(§4.2 shows that is exactly when it starts to pay off).

---

## 3. How you would take this to production

The sandbox → production gap, component by component:

| Sandbox (this repo) | Production replacement |
|---|---|
| Synthetic 1,200‑node graph regenerated from a seed | Real transaction stream landed in a graph store (Neptune / TigerGraph / Neo4j) or a feature pipeline (Spark GraphFrames, GraphStorm) |
| Dense `N×N` adjacency in NumPy | Neighbour‑sampling GNN (GraphSAGE / GAT / R‑GCN) over billions of edges; mini‑batch training |
| Hand‑written autograd | PyTorch Geometric / DGL, or JAX; GPU |
| `FederatedEnv` + `fedavg` | A real FL framework: **Flower**, **NVIDIA FLARE**, **TF‑Federated**, **OpenFL** |
| Plain weight averaging | **Secure aggregation** (server never sees individual updates) + **differential privacy** (DP‑SGD) — usually a regulatory precondition, not an optional extra |
| Labels known for all nodes | Weak/partial labels, delayed labels (chargebacks arrive weeks later), label noise; semi‑supervised or PU learning |
| One shot, offline | Continuous retraining, drift monitoring, champion/challenger, model registry |
| Score → F1 | Score → **case management**: alerts ranked, routed to investigators, with **explanations** (which neighbours/paths drove the score) because AML/fraud decisions must be auditable (SR 11‑7, EU AI Act, GDPR Art. 22) |
| RL orchestrator always on | Start with availability/staleness heuristics; add a learned policy only for the constrained regime where it measurably helps |

### A realistic phased rollout for a bank consortium

1. **Single‑bank GNN.** Each bank builds its own entity graph and a GNN fraud
   model. Immediate value, no federation risk. (Months.)
2. **Federated model, random selection, secure aggregation + DP.** Banks jointly
   train one GNN via Flower/FLARE. Governance, legal basis, and the aggregation
   protocol are the hard part, not the ML. (Quarters.)
3. **Shared entity resolution / graph schema.** Agree what a "node" is across
   banks (hashed identifiers, shared beneficiary keys) — often the real
   bottleneck.
4. **Orchestration.** Only if rounds are slow or clients heterogeneous: add
   staleness/cost‑aware selection; benchmark against random exactly as
   `sweep.py` does here before trusting it.

---

## 4. When *not* to use this

| Situation | Do this instead |
|---|---|
| One organisation already has enough labelled data | Train centrally. Federation adds cost and complexity for nothing. |
| The parties are one legal entity / can lawfully pool data | Pool it and train centrally; FL is a workaround for a constraint you don't have. |
| The fraud signal is not relational (pure tabular, per‑transaction) | A gradient‑boosted tree on transaction features will match or beat a GNN with far less engineering. |
| You want a learned client‑selection policy "because RL" | Per this repo's result, random sampling ties it. Use random/availability‑based selection unless you have deadline/straggler constraints. |
| You think FL alone makes you compliant | It doesn't. You still need secure aggregation, DP, a lawful basis, and explainability. FL reduces *raw‑data* movement, not regulatory scope. |
| Adversaries adapt quickly (they do) | Budget for continuous relabelling, drift detection, and red‑teaming; a static model decays fast in fraud. |

---

## 5. One‑line summary

The **graph half** of FedGraph‑RL is how modern fraud/AML detection already works;
the **federated half** is where cross‑bank financial‑crime collaboration is
heading and where healthcare ML already is; the **RL orchestration half** is a
research question this repo answers cautiously — worth it only under real
wall‑clock or straggler pressure.
