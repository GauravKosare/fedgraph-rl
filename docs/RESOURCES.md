# Resources for working on federated graph fraud / mule detection

Annotated. Grouped by what you'd use them for. Everything here is public.
Links are given as search terms / repo names rather than raw URLs so they don't
rot — all are findable in one search.

> **Honest caveat:** there is **no public dataset of real transactions split
> across real banks with mule labels** — that data does not leave banks. Every
> resource below is either (a) synthetic, (b) a single‑institution graph, or
> (c) a public blockchain graph. You simulate the federation, which is exactly
> what this repo does.

---

## 1. Closest to this project's exact scenario (federated + financial crime)

| Resource | What it is | Why it matters here |
|---|---|---|
| **PETs Prize Challenge** (UK/US, 2022–2023) — *financial crime track* (DrivenData) | A public competition on **privacy‑enhancing federated learning across simulated banks + a central network operator (SWIFT)** to detect anomalous / illicit payments. Synthetic payment‑network dataset, full task spec, and **winning solution write‑ups + code are published**. | This is almost the `TARGET_PROBLEM.md` brief, already scoped by regulators and industry. Start here for problem framing, evaluation design, and how teams handled secure aggregation + DP. |
| **Suzumura et al., "Towards Federated Graph Learning for Collaborative Financial Crimes Detection"** (IBM, 2019, arXiv) | Position + early experiments on exactly "train a GNN across banks without sharing data". | The intellectual origin of this repo's idea. |
| **FATF / Egmont Group** reports on **information sharing and privacy‑enhancing technologies in AML** | Regulator guidance on *why* and *how* institutions are being pushed toward pooling intelligence not data. | Grounds the "why federated" argument. |
| **Singapore MAS COSMIC** platform (2024) | A live regulator‑run platform for banks to share financial‑crime information. Public design docs. | Real‑world proof the collaboration model exists. |

---

## 2. Simulators — generate your own labelled payment graph

| Tool | Notes |
|---|---|
| **AMLSim** (IBM, GitHub) | Agent‑based simulator that emits a **labelled transaction graph** with configurable laundering **typologies** (fan‑in, fan‑out, cycle, stack, bipartite, gather‑scatter). Java + Python. **Best fit for the `v0.3` payment‑flow data model** — you can plant mule structures and assign accounts to "banks". |
| **PaySim** (Kaggle: *Synthetic Financial Datasets For Fraud Detection*, Lopez‑Rojas) | Mobile‑money transaction simulator. ~6M rows, `TRANSFER`/`CASH_OUT` with `isFraud`. The fraud pattern (drain an account, cash out via another) is mule‑flavoured. Tabular but you can build a graph from `nameOrig`/`nameDest`. |
| **AMLworld generator** (IBM, released with the NeurIPS 2023 dataset) | The generator behind the dataset below; lets you tune illicit ratio and typologies. |
| **This repo's `data.py`** | The current synthetic generator. `v0.3` plan: replace with an AMLSim‑style payment‑flow model + per‑bank ownership. |

---

## 3. Ready‑made datasets

### Transaction / money‑laundering graphs
| Dataset | Size / type | Labels |
|---|---|---|
| **IBM "Realistic Synthetic Financial Transactions for AML" (AMLworld)** — NeurIPS 2023, Altman et al. (Kaggle) | Millions of transactions, **HI** (high‑illicit) and **LI** (low‑illicit) variants | Per‑transaction laundering label + typology |
| **Elliptic** (Weber et al., 2019) | ~203k Bitcoin tx nodes, 234k edges, 166 features | licit / illicit / unknown |
| **Elliptic++** | Adds the actor (wallet) layer and more labels | licit / illicit |
| **DGraph‑Fin** (Finvolution, NeurIPS 2022 datasets track) | ~3M nodes, ~4M edges, **real** financial fraud, dynamic | fraud / benign node labels |
| **Rabobank / IBM transaction network** sets | Smaller academic transaction graphs | varies |

### Fraud‑on‑graph benchmarks (not payments, but standard for GNN‑fraud methods)
| Dataset | Domain |
|---|---|
| **YelpChi** | Fake‑review spammer detection (used by CARE‑GNN, PC‑GNN) |
| **Amazon (fraud)** | Fraudulent reviewer detection |
| **T‑Finance, T‑Social** (from BWGNN paper) | Financial + social anomaly nodes |

### Tabular fraud (for baselines / account‑opening fraud)
| Dataset | Notes |
|---|---|
| **IEEE‑CIS Fraud Detection** (Kaggle 2019) | E‑commerce card fraud, ~590k tx, heavy feature engineering |
| **Bank Account Fraud (BAF)** — NeurIPS 2022, Feedzai | **Account‑opening** fraud, 6 variants, designed for fairness/robustness testing |
| **Credit Card Fraud** (Kaggle, ULB) | 284k tx, 0.17% fraud, PCA features — the classic imbalance benchmark |

---

## 4. Benchmarks & model implementations

| Repo / benchmark | Use |
|---|---|
| **GADBench** (NeurIPS 2023) | Graph‑anomaly‑detection benchmark: 29 datasets + ~20 algorithms, unified runner. Fastest way to see how standard GNN‑fraud models compare on your data. |
| **DGFraud** / **DGFraud‑TF2** | Toolbox of GNN fraud detectors: CARE‑GNN, PC‑GNN, GraphConsis, Player2Vec, GEM, SemiGNN. |
| **antifraud** (GitHub) | Implementations of GTAN, STAN, and transaction‑fraud GNNs with the S‑FFSD simulated dataset. |
| **PyGOD** | Python library of graph outlier detectors (DOMINANT, CoLA, etc.). |
| **FedGraphNN** (FedML) | Federated GNN benchmark + starter code — the "federated" half in a reusable form. |

---

## 5. Frameworks

### Federated learning
| Framework | Notes |
|---|---|
| **Flower** (`flwr`) | Framework‑agnostic, simplest to prototype with; good docs. |
| **NVIDIA FLARE** | Production‑leaning; used in regulated deployments (healthcare, finance). |
| **FATE** (WeBank) | **Finance‑oriented**; has federated GNN (`FedGraphNN`‑style) and secure aggregation built in. |
| **TensorFlow Federated**, **OpenFL** (Intel) | Alternatives; TFF good for simulation, OpenFL for real deployments. |

### Graph ML
| Library | Notes |
|---|---|
| **PyTorch Geometric (PyG)** | Most common; `NeighborLoader` for scaling, many GNN layers. |
| **DGL** | Strong for heterogeneous graphs (accounts + merchants + devices) and large‑scale sampling. |
| **GraphStorm** (AWS) | Distributed GNN training on billion‑edge graphs; enterprise‑oriented. |

### Privacy add‑ons (deployment preconditions)
| Tool | For |
|---|---|
| **Opacus** | DP‑SGD (differentially private training) in PyTorch. |
| **TenSEAL**, **PySyft** (OpenMined) | Homomorphic encryption / secure aggregation primitives. |

---

## 6. Background reading (free)

**The scam & the numbers**
- **UK Finance — *Annual Fraud Report*** (APP fraud loss figures, mule stats).
- **Payment Systems Regulator (PSR)** — APP‑fraud **mandatory reimbursement**
  policy statements (in force 7 Oct 2024).
- **Europol** — money‑muling reports, **EMMA** operation summaries.
- **FATF** — money‑laundering typologies; trade‑based ML.

**Methods**
- Weber et al., *Anti‑Money Laundering in Bitcoin: Experimenting with GCN* (2019) — Elliptic.
- Cardoso et al., *LaundroGraph: Self‑Supervised Graph Representation Learning for AML* (Feedzai, ICAIF 2022).
- Tang et al., *Rethinking Graph Neural Networks for Anomaly Detection* (BWGNN, ICML 2022).
- Dou et al., *Enhancing Graph Neural Network‑based Fraud Detectors against Camouflaged Fraudsters* (CARE‑GNN, CIKM 2020).
- Liu et al., *Pick and Choose: A GNN‑based Imbalanced Learning Approach for Fraud Detection* (PC‑GNN, WWW 2021).
- Altman et al., *Realistic Synthetic Financial Transactions for AML Models* (NeurIPS 2023).

**Federated + finance**
- He et al., *FedGraphNN* (2021).
- PETs Prize Challenge — technical reports from the winning teams (published by NIST / DrivenData).

---

## 7. A suggested starting path

1. Read the **PETs Prize Challenge** financial‑crime track write‑ups — problem framing is done for you.
2. Generate a labelled graph with **AMLSim** (or start from **AMLworld HI**).
3. Reproduce a non‑federated baseline with **GADBench** or **PC‑GNN** to know the ceiling.
4. Split the graph by "bank", wrap it in **Flower**, and compare federated vs centralised — this repo's `federated.py` / `environment.py` are a minimal reference.
5. Add **DP‑SGD (Opacus)** and secure aggregation; measure the utility cost.
6. Only then revisit the **RL orchestrator** — and only for the constrained
   (deadline / straggler) regime where this repo's §4.2 shows it helps.
