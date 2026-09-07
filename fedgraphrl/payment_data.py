"""v0.3 payment-flow data model: federated mule-account detection.

Instead of generic "fraud rings", this generates a **directed payment graph**
shaped like real Authorised-Push-Payment (APP) scam money flow:

    victim  --£X-->  1st-hop mule  --split-->  layering mules  -->  cash-out

- Nodes are bank accounts, each owned by one of `n_banks` banks.
- The **non-IID split is dictated by bank ownership** (`partition_by_bank`), not a
  Dirichlet knob: the victim and the first-hop mule are almost always at
  different banks, so the receiving bank sees the mule but not the payment
  context, and vice-versa.
- Labels mark the **mule / layering / cash-out** accounts (what the *receiving*
  bank must detect). The victim is not labelled fraud -- they are a victim.
- Every flagged-worthy node carries `amount_at_risk`: the money that is protected
  if the model catches it. Catching the first hop protects the whole episode;
  catching a downstream node protects only its slice. This is what the
  money-weighted reward / metric optimise (see `metrics.money_weighted_scores`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import GraphData, _normalize_adj


@dataclass
class ScamEpisode:
    victim: int
    first_hop: int
    layering: list[int]
    cash_out: list[int]
    amount: float          # money the victim was tricked into sending


def make_payment_graph(
    n_accounts: int = 1500,
    n_banks: int = 6,
    n_scam_episodes: int = 55,
    num_features: int = 16,
    legit_payments_per_acct: float = 3.0,
    fanout_range: tuple = (2, 4),
    forward_ratio_range: tuple = (0.80, 0.97),
    n_highrisk_banks: int = 2,          # banks that receive most mule traffic
    highrisk_share: float = 0.75,       # fraction of mule accounts placed there
    legit_merchant_frac: float = 0.10,  # legit accounts that *look* mule-like (decoys)
    seed: int = 0,
) -> tuple[GraphData, list[ScamEpisode]]:
    rng = np.random.default_rng(seed)
    bank_of = rng.integers(0, n_banks, size=n_accounts)
    highrisk_banks = set(range(n_highrisk_banks))
    is_mule = np.zeros(n_accounts, dtype=np.int64)
    amount_at_risk = np.zeros(n_accounts, dtype=np.float64)

    hr_accounts = np.where(np.isin(bank_of, list(highrisk_banks)))[0]
    other_accounts = np.where(~np.isin(bank_of, list(highrisk_banks)))[0]

    edges: dict[tuple, float] = {}

    def add_edge(s, d, amt):
        if s != d:
            edges[(int(s), int(d))] = edges.get((int(s), int(d)), 0.0) + float(amt)

    # -- legit background: modest, roughly balanced payments -------------
    for a in range(n_accounts):
        for _ in range(rng.poisson(legit_payments_per_acct)):
            add_edge(a, int(rng.integers(0, n_accounts)), rng.lognormal(3.5, 0.8))

    # -- legit "merchant / payroll" decoys: high throughput, many senders --
    n_merch = int(legit_merchant_frac * n_accounts)
    merchants = rng.choice(n_accounts, size=n_merch, replace=False)
    for mrc in merchants:
        for _ in range(rng.integers(8, 25)):
            add_edge(int(rng.integers(0, n_accounts)), int(mrc), rng.lognormal(4.0, 0.7))
        for _ in range(rng.integers(6, 20)):
            add_edge(int(mrc), int(rng.integers(0, n_accounts)), rng.lognormal(4.0, 0.7))

    # -- scam episodes -------------------------------------------------
    episodes: list[ScamEpisode] = []

    def pick_mule_account(avoid_bank):
        pool = hr_accounts if rng.random() < highrisk_share else other_accounts
        for _ in range(20):
            c = int(rng.choice(pool))
            if bank_of[c] != avoid_bank:
                return c
        return int(rng.choice(pool))

    for _ in range(n_scam_episodes):
        victim = int(rng.choice(other_accounts))              # victims: ordinary banks
        amount = float(rng.lognormal(8.4, 0.9))               # ~ £1k-40k, heavy tail
        hop1 = pick_mule_account(bank_of[victim])
        is_mule[hop1] = 1
        amount_at_risk[hop1] += amount
        add_edge(victim, hop1, amount)

        n_layer = int(rng.integers(*fanout_range))
        fwd = rng.uniform(*forward_ratio_range)
        layering, cash_out = [], []
        per = amount * fwd / n_layer
        for _ in range(n_layer):
            m = pick_mule_account(bank_of[hop1])
            is_mule[m] = 1
            amount_at_risk[m] += per
            add_edge(hop1, m, per)
            layering.append(m)
            co = int(rng.integers(0, n_accounts))
            is_mule[co] = 1
            amount_at_risk[co] += per * rng.uniform(0.8, 0.98)
            add_edge(m, co, per * rng.uniform(0.8, 0.98))
            cash_out.append(co)
        episodes.append(ScamEpisode(victim, hop1, layering, cash_out, amount))

    # -- adjacency (symmetric-normalised for the GCN) -------------------
    A = np.zeros((n_accounts, n_accounts))
    in_amt = np.zeros(n_accounts)
    out_amt = np.zeros(n_accounts)
    in_deg = np.zeros(n_accounts)
    out_deg = np.zeros(n_accounts)
    senders = [set() for _ in range(n_accounts)]
    for (s, d), amt in edges.items():
        A[s, d] = A[d, s] = 1.0
        out_amt[s] += amt
        in_amt[d] += amt
        out_deg[s] += 1
        in_deg[d] += 1
        senders[d].add(s)

    # -- payment-flow node features (deliberately noisy / overlapping) ---
    n_senders = np.array([len(s) for s in senders], dtype=float)
    flow_through = out_amt / (in_amt + 1.0)          # mules ~1 fast, but so are merchants
    acct_age = rng.uniform(20, 3000, n_accounts)
    n_mule = int(is_mule.sum())
    acct_age[is_mule == 1] = rng.uniform(5, 1400, n_mule)     # heavy overlap with legit
    # jitter the two most discriminative signals so no single feature separates
    ft_obs = np.clip(flow_through, 0, 3) + rng.normal(0, 0.6, n_accounts)
    ns_obs = n_senders * rng.lognormal(0, 0.5, n_accounts)
    signal = np.stack([
        np.log1p(in_amt), np.log1p(out_amt),
        in_deg, out_deg, ns_obs,
        ft_obs,
        acct_age / 3000.0,
        (ns_obs > np.percentile(ns_obs, 90)).astype(float),
    ], axis=1)
    noise = rng.normal(0, 1, size=(n_accounts, max(num_features - signal.shape[1], 1)))
    features = np.concatenate([signal, noise], axis=1)
    features = (features - features.mean(0)) / (features.std(0) + 1e-9)

    # -- splits -------------------------------------------------------
    idx = rng.permutation(n_accounts)
    n_tr, n_va = int(0.6 * n_accounts), int(0.2 * n_accounts)
    train_mask = np.zeros(n_accounts, bool); train_mask[idx[:n_tr]] = True
    val_mask = np.zeros(n_accounts, bool); val_mask[idx[n_tr:n_tr + n_va]] = True
    test_mask = np.zeros(n_accounts, bool); test_mask[idx[n_tr + n_va:]] = True

    data = GraphData(
        features=features, labels=is_mule, adj=_normalize_adj(A),
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        amount_at_risk=amount_at_risk, bank_of=bank_of,
    )
    return data, episodes


def partition_by_bank(data: GraphData) -> list[np.ndarray]:
    """One shard per bank -- the non-IID split is the ownership itself."""
    if data.bank_of is None:
        raise ValueError("payment-flow GraphData required (bank_of is None)")
    banks = np.unique(data.bank_of)
    return [np.where(data.bank_of == b)[0] for b in banks]
