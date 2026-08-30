"""A tiny reverse-mode autograd engine over NumPy arrays.

Only the operations required by the GCN classifier are implemented:
elementwise add/sub/mul, matmul, ReLU, dropout, row-softmax cross-entropy,
and reductions.  It is deliberately small (~150 lines) so the whole training
stack stays dependency-free (NumPy only).
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` so that it matches `shape` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_parents", "requires_grad")

    def __init__(self, data, requires_grad: bool = False, _parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._parents = _parents

    # -- construction helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    # -- ops ----------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._accum(-out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad * self.data, other.data.shape))

        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def matmul(self, other):
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def _backward():
            if self.requires_grad:
                self._accum(out.grad @ other.data.T)
            if other.requires_grad:
                other._accum(self.data.T @ out.grad)

        out._backward = _backward
        return out

    __matmul__ = matmul

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def dropout(self, p: float, training: bool, rng: np.random.Generator):
        if not training or p <= 0.0:
            return self
        mask = (rng.random(self.data.shape) >= p) / (1.0 - p)
        out = Tensor(self.data * mask, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * mask)

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._accum(np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    # -- loss -------------------------------------------------------------
    def softmax_cross_entropy(self, targets: np.ndarray, weight: np.ndarray | None = None):
        """`self` are logits of shape (N, C); `targets` int labels of shape (N,)."""
        logits = self.data
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        n = logits.shape[0]
        w = np.ones(n) if weight is None else weight[targets]
        loss_val = -(w * np.log(probs[np.arange(n), targets] + 1e-12)).sum() / w.sum()
        out = Tensor(loss_val, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                g = probs.copy()
                g[np.arange(n), targets] -= 1.0
                g *= (w / w.sum())[:, None]
                self._accum(g * out.grad)

        out._backward = _backward
        return out

    # -- engine ----------------------------------------------------------
    def _accum(self, g):
        self.grad = g if self.grad is None else self.grad + g

    def backward(self):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()
