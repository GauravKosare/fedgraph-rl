"""Federated learning simulator: FedAvg over GCN clients on graph shards."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autograd import Tensor
from .data import GraphData
from .gnn import GCN, SGD
from .metrics import binary_scores


@dataclass
class ClientReport:
    client_id: int
    weights: dict
    num_train: int
    local_loss: float
    compute_cost: float          # simulated FLOP-ish cost of the local work
    comm_cost: float             # simulated bytes uploaded (param count)


class FederatedClient:
    def __init__(self, client_id: int, shard: GraphData, model_cfg: dict,
                 lr: float = 0.05, class_weight=(1.0, 5.0)):
        self.client_id = client_id
        self.shard = shard
        self.model_cfg = model_cfg
        self.lr = lr
        self.class_weight = np.array(class_weight)

    @property
    def num_train(self) -> int:
        return int(self.shard.train_mask.sum())

    @property
    def fraud_rate(self) -> float:
        m = self.shard.train_mask
        return float(self.shard.labels[m].mean()) if m.any() else 0.0

    def train(self, global_weights: dict, local_epochs: int, seed: int) -> ClientReport:
        model = GCN(**self.model_cfg, seed=seed)
        model.set_weights(global_weights)
        opt = SGD(model.params, lr=self.lr)

        feats, adj = self.shard.features, self.shard.adj
        y = self.shard.labels
        mask = self.shard.train_mask
        loss_val = 0.0
        if mask.any():
            for _ in range(max(1, local_epochs)):
                model.zero_grad()
                logits = model.forward(feats, adj, training=True)
                sel = Tensor(np.eye(self.shard.num_nodes)[mask]) @ logits
                loss = sel.softmax_cross_entropy(y[mask], weight=self.class_weight)
                loss.backward()
                opt.step()
                loss_val = float(loss.data)

        n_params = sum(w.size for w in global_weights.values())
        edges = float((self.shard.adj > 0).sum())
        compute_cost = local_epochs * (edges + self.shard.num_nodes) * self.model_cfg["hidden"] / 1e6
        return ClientReport(self.client_id, model.get_weights(), self.num_train,
                            loss_val, compute_cost, n_params / 1e6)


def fedavg(reports: list[ClientReport], mix: np.ndarray | None = None) -> dict:
    """Weighted average of client weights.

    `mix` (optional) is an extra per-client multiplier from the RL controller's
    budget allocation -- it lets the server up-weight clients it chose to invest
    more investigation resource in.
    """
    if mix is None:
        mix = np.ones(len(reports))
    w = np.array([r.num_train for r in reports], dtype=float) * mix
    w = w / w.sum() if w.sum() > 0 else np.ones(len(reports)) / len(reports)
    out = {k: np.zeros_like(v) for k, v in reports[0].weights.items()}
    for coeff, r in zip(w, reports):
        for k in out:
            out[k] += coeff * r.weights[k]
    return out


class FederatedServer:
    def __init__(self, global_data: GraphData, model_cfg: dict):
        self.global_data = global_data
        self.model_cfg = model_cfg
        self.model = GCN(**model_cfg, seed=1234)
        self.round = 0

    def global_weights(self) -> dict:
        return self.model.get_weights()

    def apply(self, new_weights: dict):
        self.model.set_weights(new_weights)
        self.round += 1

    def evaluate(self, split: str = "val", threshold: float | None = None) -> dict:
        d = self.global_data
        mask = {"train": d.train_mask, "val": d.val_mask, "test": d.test_mask}[split]
        proba = self.model.predict_proba(d.features, d.adj)[:, 1]
        if threshold is None:
            threshold = 0.5
        m = binary_scores(d.labels[mask], proba[mask], threshold)
        m["threshold"] = float(threshold)
        return m

    def tuned_threshold(self) -> float:
        """Threshold that maximises F1 on the validation split."""
        d = self.global_data
        proba = self.model.predict_proba(d.features, d.adj)[:, 1]
        yv, pv = d.labels[d.val_mask], proba[d.val_mask]
        cands = np.unique(np.concatenate([[0.0], np.sort(pv), [1.0]]))
        best_t, best_f1 = 0.5, -1.0
        for t in cands:
            f1 = binary_scores(yv, pv, t)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t)
