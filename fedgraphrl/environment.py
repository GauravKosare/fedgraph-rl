"""Federated orchestration as an RL environment.

One *episode* trains a global GCN fraud detector from scratch over
`max_rounds` federated rounds.  Each *step* the agent chooses which clients
participate and how to split a fixed local-compute budget among them.

    state   : global progress + per-client descriptors
    action  : (selected client ids, local epochs per selected client)
    reward  : gain in global validation F1  -  cost_coeff * max(0, cost - budget)
              (soft cost constraint: free within a per-round compute budget,
               linear penalty only on the overshoot)
"""
from __future__ import annotations

import numpy as np

from .federated import FederatedClient, FederatedServer, fedavg


class FederatedEnv:
    def __init__(self, global_data, shards, model_cfg, *,
                 max_rounds: int = 25, clients_per_round: int = 4,
                 epoch_budget: int = 12, cost_coeff: float = 0.02,
                 cost_slack: float = 1.2, seed: int = 0):
        self.global_data = global_data
        self.model_cfg = model_cfg
        self.max_rounds = max_rounds
        self.clients_per_round = clients_per_round
        self.epoch_budget = epoch_budget
        self.cost_coeff = cost_coeff
        self.rng = np.random.default_rng(seed)
        # `shards` may be node-index arrays (from partition_non_iid) or ready GraphData
        shard_graphs = [global_data.subgraph(s) if isinstance(s, np.ndarray) else s
                        for s in shards]
        self.clients = [FederatedClient(i, s, model_cfg) for i, s in enumerate(shard_graphs)]
        self.n_clients = len(self.clients)
        self.client_desc = np.array([
            [c.num_train, c.fraud_rate, float((c.shard.adj > 0).sum())]
            for c in self.clients
        ], dtype=float)
        self.client_desc[:, 0] /= self.client_desc[:, 0].max() + 1e-9
        self.client_desc[:, 2] /= self.client_desc[:, 2].max() + 1e-9
        self.feature_dim = 3 + 4  # global(3) + per-client dynamic(4)

        # -- soft cost budget: the cost of an "average" full round -----------
        h = model_cfg["hidden"]
        n_params = (model_cfg["in_dim"] * h + h + h * model_cfg["n_classes"]
                    + model_cfg["n_classes"])
        comm_per_client = n_params / 1e6
        cpe = np.array([(float((c.shard.adj > 0).sum()) + c.shard.num_nodes) * h / 1e6
                        for c in self.clients])           # compute per local epoch
        self.cost_budget = cost_slack * (
            clients_per_round * comm_per_client + epoch_budget * cpe.mean())
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.server = FederatedServer(self.global_data, self.model_cfg)
        self.t = 0
        self.last_f1 = self.server.evaluate("val")["f1"]
        self.since_selected = np.zeros(self.n_clients)
        self.last_loss = np.ones(self.n_clients)
        return self._state()

    def _state(self) -> np.ndarray:
        g = np.array([self.t / self.max_rounds, self.last_f1,
                      self.server.round / self.max_rounds])
        rows = []
        for i in range(self.n_clients):
            dyn = np.array([
                self.since_selected[i] / self.max_rounds,
                self.last_loss[i],
                self.client_desc[i, 1],           # fraud rate
                self.client_desc[i, 0],           # size
            ])
            rows.append(np.concatenate([g, dyn]))
        return np.stack(rows)                       # (n_clients, feature_dim)

    # ------------------------------------------------------------------
    def step(self, selected: np.ndarray, epochs: np.ndarray):
        selected = np.asarray(selected, dtype=int)
        epochs = np.asarray(epochs, dtype=int)
        gw = self.server.global_weights()
        reports = []
        for cid, ep in zip(selected, epochs):
            seed = int(self.rng.integers(0, 1 << 30))
            reports.append(self.clients[cid].train(gw, ep, seed))

        mix = np.array([max(r.compute_cost, 1e-6) for r in reports])
        self.server.apply(fedavg(reports, mix=mix))

        f1 = self.server.evaluate("val")["f1"]
        total_cost = sum(r.compute_cost + r.comm_cost for r in reports)
        # soft constraint: no penalty while within the per-round cost budget,
        # linear penalty only on the overshoot.
        overshoot = max(0.0, total_cost - self.cost_budget)
        reward = (f1 - self.last_f1) * 100.0 - self.cost_coeff * overshoot

        self.last_f1 = f1
        self.since_selected += 1
        self.since_selected[selected] = 0
        for r in reports:
            self.last_loss[r.client_id] = np.tanh(r.local_loss)
        self.t += 1

        done = self.t >= self.max_rounds
        test_metrics = None
        if done:
            t = self.server.tuned_threshold()
            test_metrics = self.server.evaluate("test", threshold=t)
        info = {"val_f1": f1, "cost": total_cost, "test": test_metrics}
        return self._state(), reward, done, info
