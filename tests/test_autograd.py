"""Sanity checks: autograd gradients vs finite differences, and a smoke test
of one federated round."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedgraphrl.autograd import Tensor
from fedgraphrl.data import make_transaction_graph, partition_non_iid
from fedgraphrl.environment import FederatedEnv


def test_matmul_relu_gradcheck():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 3))
    W = rng.normal(size=(3, 2))
    y = np.array([0, 1, 1, 0])

    def loss_of(Wmat):
        x = Tensor(A)
        w = Tensor(Wmat, requires_grad=True)
        logits = (x @ w).relu()
        return logits.softmax_cross_entropy(y), w

    l, w = loss_of(W)
    l.backward()
    ana = w.grad

    num = np.zeros_like(W)
    eps = 1e-6
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            d = np.zeros_like(W); d[i, j] = eps
            lp, _ = loss_of(W + d)
            lm, _ = loss_of(W - d)
            num[i, j] = (lp.data - lm.data) / (2 * eps)
    assert np.allclose(ana, num, atol=1e-4), (ana, num)


def test_federated_round_smoke():
    data, ring_of = make_transaction_graph(num_accounts=300, fraud_ring_count=4, seed=1)
    shards = partition_non_iid(data, ring_of, num_clients=4, seed=1)
    cfg = {"hidden": 16, "dropout": 0.5, "n_classes": 2, "in_dim": data.num_features}
    env = FederatedEnv(data, shards, cfg, max_rounds=3, clients_per_round=2,
                       epoch_budget=6, seed=1)
    s = env.reset()
    assert s.shape == (4, env.feature_dim)
    s, r, done, info = env.step(np.array([0, 1]), np.array([3, 3]))
    assert np.isfinite(r) and "val_f1" in info


def test_straggler_drop():
    data, ring_of = make_transaction_graph(num_accounts=300, fraud_ring_count=4, seed=2)
    shards = partition_non_iid(data, ring_of, num_clients=4, seed=2)
    cfg = {"hidden": 16, "dropout": 0.5, "n_classes": 2, "in_dim": data.num_features}
    env = FederatedEnv(data, shards, cfg, max_rounds=4, clients_per_round=3,
                       epoch_budget=9, stragglers=True, straggler_frac=0.5,
                       straggler_slowdown=6.0, deadline_slack=1.0, seed=2)
    assert (env.client_speed < 1.0).any()          # some clients are slow
    slow = int(np.argmin(env.client_speed))
    # dump the whole epoch budget on the slowest client -> it must be dropped
    _, _, _, info = env.step(np.array([slow, (slow + 1) % 4, (slow + 2) % 4]),
                             np.array([7, 1, 1]))
    assert info["dropped"] >= 1


def test_payment_flow_money_metric():
    from fedgraphrl.payment_data import make_payment_graph, partition_by_bank
    from fedgraphrl.environment import FederatedEnv
    from fedgraphrl.metrics import money_weighted_scores

    data, episodes = make_payment_graph(n_accounts=400, n_banks=4,
                                        n_scam_episodes=15, seed=3)
    assert data.amount_at_risk is not None and data.bank_of is not None
    assert data.labels.sum() > 0 and len(episodes) == 15
    shards = partition_by_bank(data)
    assert len(shards) == 4 and sum(len(s) for s in shards) == data.num_nodes
    # victim and first hop should usually be at different banks
    diff = [data.bank_of[e.victim] != data.bank_of[e.first_hop] for e in episodes]
    assert sum(diff) >= 0.7 * len(episodes)

    # money metric: perfect ranking -> money_recall 1.0 at any FP budget
    perfect = data.labels.astype(float)
    m = money_weighted_scores(data.labels, perfect, data.amount_at_risk, fp_budget=0.02)
    assert m["money_recall"] > 0.99 and m["fp_rate"] <= 0.02 + 1e-9

    cfg = {"hidden": 16, "dropout": 0.5, "n_classes": 2, "in_dim": data.num_features}
    env = FederatedEnv(data, shards, cfg, max_rounds=3, clients_per_round=2,
                       epoch_budget=6, reward_mode="money", fp_budget=0.05, seed=3)
    s = env.reset()
    _, r, done, info = env.step(np.array([0, 1]), np.array([3, 3]))
    assert np.isfinite(r)
    _, _, done, info = env.step(np.array([2, 3]), np.array([3, 3]))
    _, _, done, info = env.step(np.array([0, 2]), np.array([3, 3]))
    assert done and "money_recall" in info["test"]


def test_coverage_heuristic_visits_all():
    from fedgraphrl.payment_data import make_payment_graph, partition_by_bank
    from fedgraphrl.environment import FederatedEnv
    from fedgraphrl.rl_controller import HeuristicController

    data, _ = make_payment_graph(n_accounts=500, n_banks=5, n_scam_episodes=18, seed=4)
    shards = partition_by_bank(data)
    cfg = {"hidden": 16, "dropout": 0.5, "n_classes": 2, "in_dim": data.num_features}
    env = FederatedEnv(data, shards, cfg, max_rounds=8, clients_per_round=2,
                       epoch_budget=6, reward_mode="money", fp_budget=0.05, seed=4)
    # instrument: record which clients the coverage heuristic picks each round
    picked = []
    orig_step = env.step
    def spy(sel, ep):
        picked.append(tuple(int(x) for x in sel))
        return orig_step(sel, ep)
    env.step = spy
    HeuristicController(env, kind="coverage", seed=4).run_episode()
    visited = set(c for rnd in picked for c in rnd)
    assert visited == set(range(5)), visited          # every bank trained at least once
    # a bank is never revisited while another is still unvisited (first 2-3 rounds)
    seen = set()
    for rnd in picked[:3]:
        seen |= set(rnd)
    assert len(seen) >= 5 - 1


if __name__ == "__main__":
    test_matmul_relu_gradcheck()
    test_federated_round_smoke()
    test_straggler_drop()
    test_payment_flow_money_metric()
    test_coverage_heuristic_visits_all()
    print("ok")
