"""Synthetic transaction-graph generator for federated fraud detection.

A single global graph of `accounts` (nodes) connected by `transactions` (edges)
is generated with planted *fraud rings* -- densely connected communities of
fraudulent accounts that transact mostly among themselves and with a few
"mule" bridge accounts.  The graph is then partitioned across `num_clients`
simulated edge devices (e.g. banks / regional processors) in a **non-IID** way:
each client sees a biased slice of the fraud rings, so no single client can
learn a good detector alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphData:
    features: np.ndarray          # (N, F)
    labels: np.ndarray            # (N,) in {0, 1}
    adj: np.ndarray               # (N, N) symmetric normalized adjacency (with self loops)
    train_mask: np.ndarray        # (N,) bool
    val_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    def subgraph(self, node_idx: np.ndarray) -> "GraphData":
        node_idx = np.sort(node_idx)
        a = self.adj[np.ix_(node_idx, node_idx)]
        return GraphData(
            features=self.features[node_idx],
            labels=self.labels[node_idx],
            adj=_normalize_adj((a > 0).astype(np.float64)),
            train_mask=self.train_mask[node_idx],
            val_mask=self.val_mask[node_idx],
            test_mask=self.test_mask[node_idx],
        )


def _normalize_adj(a: np.ndarray) -> np.ndarray:
    a = a + np.eye(a.shape[0])
    deg = a.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return dinv[:, None] * a * dinv[None, :]


def make_transaction_graph(
    num_accounts: int = 1200,
    num_features: int = 16,
    fraud_ring_count: int = 12,
    ring_size_range: tuple = (8, 22),
    avg_legit_degree: float = 4.0,
    seed: int = 0,
) -> tuple[GraphData, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.zeros(num_accounts, dtype=np.int64)
    edges = set()

    # -- plant fraud rings ------------------------------------------------
    ring_of = -np.ones(num_accounts, dtype=np.int64)
    cursor = 0
    for ring in range(fraud_ring_count):
        size = rng.integers(*ring_size_range)
        members = np.arange(cursor, min(cursor + size, num_accounts))
        cursor += size
        if len(members) < 3:
            break
        labels[members] = 1
        ring_of[members] = ring
        for i in members:                      # dense intra-ring wiring
            for j in members:
                if i < j and rng.random() < 0.55:
                    edges.add((int(i), int(j)))
        # a couple of mule bridges to legit accounts
        for _ in range(rng.integers(1, 4)):
            mule = int(rng.integers(cursor, num_accounts))
            edges.add((int(rng.choice(members)), mule))

    # -- legit background graph (preferential-ish attachment) ------------
    legit = np.where(labels == 0)[0]
    for node in legit:
        k = rng.poisson(avg_legit_degree)
        for _ in range(k):
            other = int(rng.choice(legit))
            if other != node:
                edges.add((min(node, other), max(node, other)))

    adj_bin = np.zeros((num_accounts, num_accounts))
    for i, j in edges:
        adj_bin[i, j] = adj_bin[j, i] = 1.0

    # -- node features: behavioural signal + noise ----------------------
    deg = adj_bin.sum(axis=1)
    base = rng.normal(0, 1, size=(num_accounts, num_features))
    signal = np.zeros((num_accounts, num_features))
    signal[:, 0] = deg / (deg.max() + 1e-9)
    signal[:, 1] = labels * rng.normal(0.45, 0.9, num_accounts)     # weakly leaky, very noisy
    signal[:, 2] = (deg > np.percentile(deg, 90)).astype(float)
    features = base + signal
    features = (features - features.mean(0)) / (features.std(0) + 1e-9)

    # -- splits ---------------------------------------------------------
    idx = rng.permutation(num_accounts)
    n_tr, n_va = int(0.6 * num_accounts), int(0.2 * num_accounts)
    train_mask = np.zeros(num_accounts, bool)
    val_mask = np.zeros(num_accounts, bool)
    test_mask = np.zeros(num_accounts, bool)
    train_mask[idx[:n_tr]] = True
    val_mask[idx[n_tr:n_tr + n_va]] = True
    test_mask[idx[n_tr + n_va:]] = True

    data = GraphData(features, labels, _normalize_adj(adj_bin),
                     train_mask, val_mask, test_mask)
    return data, ring_of


def partition_non_iid(
    data: GraphData,
    ring_of: np.ndarray,
    num_clients: int = 8,
    dirichlet_alpha: float = 0.3,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return a list of node-index arrays, one per client.

    Fraud rings are assigned to clients via a Dirichlet draw (low alpha => each
    ring concentrated on few clients).  Legit accounts are spread ~uniformly.
    """
    rng = np.random.default_rng(seed)
    client_nodes: list[list[int]] = [[] for _ in range(num_clients)]

    rings = np.unique(ring_of[ring_of >= 0])
    for ring in rings:
        members = np.where(ring_of == ring)[0]
        props = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        owners = rng.choice(num_clients, size=len(members), p=props)
        for node, owner in zip(members, owners):
            client_nodes[owner].append(int(node))

    legit = np.where(ring_of < 0)[0]
    rng.shuffle(legit)
    for c, chunk in enumerate(np.array_split(legit, num_clients)):
        client_nodes[c].extend(int(x) for x in chunk)

    return [np.array(sorted(set(c))) for c in client_nodes]
