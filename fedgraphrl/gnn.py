"""A 2-layer GCN node classifier built on the tiny autograd engine.

forward:  H1 = ReLU(A_hat @ X @ W0 + b0);  logits = A_hat @ dropout(H1) @ W1 + b1
"""
from __future__ import annotations

import numpy as np

from .autograd import Tensor


def _glorot(rng, shape):
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return rng.uniform(-limit, limit, size=shape)


class GCN:
    def __init__(self, in_dim: int, hidden: int = 32, n_classes: int = 2,
                 dropout: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dropout = dropout
        self.params: dict[str, Tensor] = {
            "W0": Tensor(_glorot(rng, (in_dim, hidden)), requires_grad=True),
            "b0": Tensor(np.zeros((1, hidden)), requires_grad=True),
            "W1": Tensor(_glorot(rng, (hidden, n_classes)), requires_grad=True),
            "b1": Tensor(np.zeros((1, n_classes)), requires_grad=True),
        }
        self._rng = rng

    # -- weight vector helpers (used by federated averaging) -------------
    def get_weights(self) -> dict[str, np.ndarray]:
        return {k: v.data.copy() for k, v in self.params.items()}

    def set_weights(self, w: dict[str, np.ndarray]):
        for k, v in w.items():
            self.params[k].data = v.copy()

    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()

    # -- forward -------------------------------------------------------
    def forward(self, features: np.ndarray, adj: np.ndarray, training: bool) -> Tensor:
        X = Tensor(features)
        A = Tensor(adj)
        h = (A @ (X @ self.params["W0"])) + self.params["b0"]
        h = h.relu()
        h = h.dropout(self.dropout, training, self._rng)
        logits = (A @ (h @ self.params["W1"])) + self.params["b1"]
        return logits

    def predict_proba(self, features, adj) -> np.ndarray:
        logits = self.forward(features, adj, training=False).data
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


class SGD:
    """Plain SGD with optional weight decay -- one optimiser per local client."""

    def __init__(self, params: dict[str, Tensor], lr: float = 0.05, weight_decay: float = 5e-4):
        self.params = params
        self.lr = lr
        self.wd = weight_decay

    def step(self):
        for p in self.params.values():
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            p.data = p.data - self.lr * g
