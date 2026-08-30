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


if __name__ == "__main__":
    test_matmul_relu_gradcheck()
    test_federated_round_smoke()
    print("ok")
