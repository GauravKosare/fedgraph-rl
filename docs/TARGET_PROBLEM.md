# Recommended target problem

> **Pick one real problem, shape the sandbox around it, and the whole project
> stops being "three ML toys glued together" and becomes a credible prototype.**

## Recommendation: federated mule‑account detection for Authorised Push Payment (APP) scams

Reframe FedGraph‑RL as a prototype of **cross‑bank detection of *mule accounts* —
the "receiving" accounts that scam victims are tricked into paying** — trained
federatedly across banks, under a **real‑time screening deadline**.

### Why this problem, specifically

| Criterion | Why APP / mule detection fits better than the alternatives |
|---|---|
| **It is genuinely a graph problem** | A mule network is money in → mule → fan‑out to layering accounts → cash‑out. The structure *is* the signal. A per‑account model sees "an account received £4,000"; the graph sees "an account that suddenly received from 11 unrelated payers at 6 banks and forwarded 95 % within an hour". |
| **It is non‑IID *by construction*** | The **sending** bank sees the victim and the payment; the **receiving** bank sees the mule and the cash‑out. Neither sees the whole path. This isn't a synthetic Dirichlet split — the data partition is dictated by which bank onboarded which account. |
| **The deadline is real, not contrived** | Banks increasingly screen payments *in‑flight* and can hold or refuse within seconds‑to‑minutes. That is exactly `FederatedEnv`'s per‑round wall‑clock deadline — the v0.2 straggler/deadline environment becomes the *default*, not an add‑on. |
| **There is a hard regulatory and financial driver** | UK APP‑fraud reimbursement became mandatory in October 2024 — banks now bear the loss, split 50/50 between sending and receiving bank. Both sides have a direct incentive to share detection capability without sharing customer data. Comparable frameworks: Singapore COSMIC, EU instant‑payments regulation, Australia's Scam‑Safe Accord. |
| **Privacy‑preserving collaboration is already the stated direction** | Regulators explicitly want banks to pool *intelligence* not *data*. Federated learning is a textbook fit; several vendors and consortia are already piloting it. |
| **Orchestration actually matters here** | With an in‑flight screening budget (limited compute/latency per cycle) and heterogeneous bank infrastructure, "which partner models do we refresh this cycle, and how much" is a real operational question — the one the RL controller is built to answer. |

### Why not the other candidates

- **Broad cross‑bank AML / transaction‑monitoring** — bigger market, but the work
  is mostly *batch* (nightly, T+1). No wall‑clock pressure ⇒ the RL half has
  nothing to exploit ⇒ back to the v0.1 null result.
- **Card‑testing / BIN‑attack detection across issuers** — real‑time and real, but
  less obviously a *graph* (it's more velocity/rate patterns) and the federation
  story is weaker (the network already sits in the middle).
- **Telecom SIM‑box fraud** — good technical fit, but smaller market and no
  reimbursement‑style regulatory forcing function.
- **Healthcare billing‑fraud rings** — federation is mature here, but the
  real‑time deadline is absent and the graph is claims‑based, not payment‑flow.

## What "solution‑provider oriented" means in practice

Shape every layer around the mule‑detection story:

| Layer | Reframe |
|---|---|
| **Data generator** (`data.py`) | Generate *payment‑flow* graphs: victim payers, mule receiving accounts, layering accounts, cash‑out. Label = confirmed mule. Bank ownership assigned per account (the non‑IID split is now "which bank holds this node", not a Dirichlet knob). Add the features a bank actually has: account age, KYC risk band, inbound/outbound velocity, first‑party vs third‑party transfer, device/beneficiary reuse. |
| **GNN** (`gnn.py`) | Heterogeneous / directed message passing (money flows one way); the "mule score" is a node score the receiving bank acts on. |
| **Federation** (`federated.py`) | Two roles per bank — *sending‑side* and *receiving‑side* views of the same payment — plus secure aggregation and DP‑SGD as first‑class, since those are deployment preconditions. |
| **Environment** (`environment.py`) | The per‑round deadline = the **screening‑cycle SLA**. Stragglers = banks with slower/older infra. `stragglers=True` becomes the default config. Reward ties to *caught‑mule rate at fixed false‑positive budget* (banks care about not freezing legitimate customers). |
| **RL controller** | The decision: given a fixed screening‑compute budget per cycle, which partner banks' sub‑models to refresh and how hard — to maximise caught mules per unit latency. Benchmarked, as now, against "refresh everyone" and "refresh at random". |
| **Outputs** | Report **money‑at‑risk prevented** and **false‑positive (frozen legit account) rate**, not just F1 — those are the numbers a bank's fraud‑ops lead and regulator ask for. Every alert ships with the sub‑graph that triggered it (explainability is mandatory). |

## Honest scoping

- Still synthetic data — but synthetic *payment‑flow* data with a documented
  generative model beats a generic Dirichlet graph for credibility.
- This does **not** become production. It becomes a **defensible prototype +
  benchmark**: "here is the federated‑graph mule‑detection problem, here is an
  environment that models the real constraints (non‑IID by bank, in‑flight
  deadline, heterogeneous infra), and here is what learned orchestration buys you
  over the naive baselines."
- The v0.1 finding (learned selection ≈ random without timing pressure) stays in
  the write‑up as the *motivation* for why the deadline matters.

## Suggested next milestones

1. **v0.2.1** (in progress) — fix the straggler‑variant training instability so
   the deadline environment is a reliable benchmark.
2. **v0.3 — payment‑flow data model.** Replace the generic graph generator with
   the mule/victim/layering/cash‑out model above; bank‑ownership partition.
3. **v0.3 — money‑weighted metrics.** Reward and reporting in "money‑at‑risk
   prevented" and "false‑positive rate", not F1.
4. **v0.4 — deployment realism.** Secure aggregation + DP‑SGD in the FL loop;
   neighbour‑sampling GNN (PyG/DGL) so the graph can scale.
5. **v0.4 — explainability.** Emit the triggering sub‑graph per flagged account.
