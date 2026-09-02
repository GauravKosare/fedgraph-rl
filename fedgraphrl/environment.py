"""Federated orchestration as an RL environment.

One *episode* trains a global GCN fraud detector from scratch over
`max_rounds` federated rounds.  Each *step* the agent chooses which clients
participate and how to split a fixed local-compute budget among them.

    state   : global progress + per-client descriptors (incl. device speed)
    action  : (selected client ids, local epochs per selected client)
    reward  : gain in global validation F1  -  cost_coeff * max(0, cost - budget)
              (soft cost constraint: free within a per-round compute budget,
               linear penalty only on the overshoot)

v0.2 -- stragglers & deadline
    Clients have heterogeneous device speeds; a fraction are "slow".  Each round
    has a wall-clock deadline.  A selected client whose (epochs / speed) work
    time exceeds the deadline is a *straggler*: it still consumes compute
    (counted as cost) but its update is DROPPED from FedAvg.  The agent must
    learn to give slow clients fewer epochs -- or skip them -- while still
    eventually covering the fraud rings they hold.  Set `stragglers=False` to
    recover the v0.1 behaviour.
"""
from __future__ import annotations

import numpy as np

from .federated import FederatedClient, FederatedServer, fedavg


class FederatedEnv:
    def __init__(self, global_data, shards, model_cfg, *,
                 max_rounds: int = 25, clients_per_round: int = 4,
                 epoch_budget: int = 12, cost_coeff: float = 0.02,
                 cost_slack: float = 1.2, stragglers: bool = False,
                 straggler_frac: float = 0.35, straggler_slowdown: float = 4.0,
                 deadline_slack: float = 1.15, drop_penalty: float = 0.6,
                 seed: int = 0):
        self.global_data = global_data
        self.model_cfg = model_cfg
        self.max_rounds = max_rounds
        self.clients_per_round = clients_per_round
        self.epoch_budget = epoch_budget
        self.cost_coeff = cost_coeff
        self.stragglers = stragglers
        self.drop_penalty = drop_penalty
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
        # global(3) + per-client dynamic(4); +1 (device speed) only under stragglers
        # so the v0.1 policy network is byte-identical to the frozen v0.1 run.
        self.feature_dim = 3 + 4 + (1 if stragglers else 0)

        # -- soft cost budget: the cost of an "average" full round -----------
        h = model_cfg["hidden"]
        n_params = (model_cfg["in_dim"] * h + h + h * model_cfg["n_classes"]
                    + model_cfg["n_classes"])
        self.comm_per_client = n_params / 1e6
        cpe = np.array([(float((c.shard.adj > 0).sum()) + c.shard.num_nodes) * h / 1e6
                        for c in self.clients])           # compute per local epoch
        self.cpe = cpe
        self.cost_budget = cost_slack * (
            clients_per_round * self.comm_per_client + epoch_budget * cpe.mean())

        # -- v0.2: device speeds + per-round wall-clock deadline ------------
        self.client_speed = np.ones(self.n_clients)
        if stragglers:
            n_slow = max(1, int(round(straggler_frac * self.n_clients)))
            slow = self.rng.choice(self.n_clients, size=n_slow, replace=False)
            self.client_speed[slow] = 1.0 / straggler_slowdown
        # work-time model: time ~= epochs * cpe / speed.  A client trains only the
        # epochs it can finish before the deadline; if that is < 1 it is dropped.
        fair_epochs = epoch_budget / clients_per_round
        self.round_deadline = deadline_slack * fair_epochs * cpe.mean()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.server = FederatedServer(self.global_data, self.model_cfg)
        self.t = 0
        self.last_f1 = self._val_f1()
        self.since_selected = np.zeros(self.n_clients)
        self.last_loss = np.ones(self.n_clients)
        return self._state()

    def _val_f1(self) -> float:
        # v0.2.1: under stragglers, reward on the *tuned-threshold* validation F1.
        # A conservative global model (common when stragglers hide the fraud-heavy
        # shards) scores 0 at the fixed 0.5 cutoff -> ΔF1 == 0 every round -> no
        # learning signal.  Scoring at the best validation threshold keeps the
        # reward informative as long as the model *ranks* fraud at all.
        # The v0.1 (no-straggler) path keeps the original fixed-0.5 reward so its
        # published numbers stay reproducible.
        if self.stragglers:
            return self.server.val_f1_tuned()
        return self.server.evaluate("val")["f1"]

    def _state(self) -> np.ndarray:
        g = np.array([self.t / self.max_rounds, self.last_f1,
                      self.server.round / self.max_rounds])
        rows = []
        for i in range(self.n_clients):
            dyn = [
                self.since_selected[i] / self.max_rounds,
                self.last_loss[i],
                self.client_desc[i, 1],           # fraud rate
                self.client_desc[i, 0],           # size
            ]
            if self.stragglers:
                dyn.append(self.client_speed[i])  # device speed (1.0 fast, <1 slow)
            rows.append(np.concatenate([g, np.array(dyn)]))
        return np.stack(rows)                       # (n_clients, feature_dim)

    # ------------------------------------------------------------------
    def step(self, selected: np.ndarray, epochs: np.ndarray):
        selected = np.asarray(selected, dtype=int)
        epochs = np.asarray(epochs, dtype=int)
        gw = self.server.global_weights()
        kept = []
        n_dropped = n_partial = 0
        total_cost = 0.0
        for cid, ep in zip(selected, epochs):
            if self.stragglers:
                max_by_deadline = self.round_deadline * self.client_speed[cid] / self.cpe[cid]
            else:
                max_by_deadline = float(ep)
            trained_ep = int(min(int(ep), np.floor(max_by_deadline)))   # epochs completed
            charged_ep = min(float(ep), max_by_deadline)                # compute burned (deadline-capped)
            total_cost += charged_ep * self.cpe[cid] + self.comm_per_client
            if trained_ep >= 1:
                seed = int(self.rng.integers(0, 1 << 30))
                rep = self.clients[cid].train(gw, trained_ep, seed)
                kept.append(rep)
                self.last_loss[cid] = np.tanh(rep.local_loss)
                n_partial += int(trained_ep < int(ep))
            else:
                n_dropped += 1                       # couldn't finish even one epoch

        if kept:
            mix = np.array([max(r.compute_cost, 1e-6) for r in kept])
            self.server.apply(fedavg(kept, mix=mix))
        else:
            self.server.round += 1                  # a wasted round: nothing aggregated

        f1 = self._val_f1()
        # soft constraint: no penalty while within the per-round cost budget,
        # linear penalty only on the overshoot.
        overshoot = max(0.0, total_cost - self.cost_budget)
        # v0.2.1: a dense penalty per dropped update -- gives the policy usable
        # signal ("you wasted a slot on a straggler") even in rounds where the
        # global F1 does not move, which is the sparse-reward cause of the
        # v0.2 training collapse.
        reward = ((f1 - self.last_f1) * 100.0
                  - self.cost_coeff * overshoot
                  - self.drop_penalty * n_dropped)

        self.last_f1 = f1
        self.since_selected += 1
        self.since_selected[selected] = 0
        self.t += 1

        done = self.t >= self.max_rounds
        test_metrics = None
        if done:
            t = self.server.tuned_threshold()
            test_metrics = self.server.evaluate("test", threshold=t)
        info = {"val_f1": f1, "cost": total_cost, "dropped": n_dropped,
                "partial": n_partial, "test": test_metrics}
        return self._state(), reward, done, info
