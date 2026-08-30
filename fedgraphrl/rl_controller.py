"""REINFORCE controller for client selection + compute-budget allocation.

The policy is a small 2-layer MLP that maps each client's state row to two
scalars: a *selection logit* and a *budget logit*.  Clients are sampled without
replacement from a softmax over selection logits (Plackett-Luce); the fixed
epoch budget is divided across the chosen clients via a softmax over their
budget logits.  Trained with episodic REINFORCE and a moving-average baseline.
"""
from __future__ import annotations

import numpy as np


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class PolicyNet:
    def __init__(self, in_dim: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.normal(0, s, (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 2))
        self.b2 = np.zeros(2)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        cache = (X, h)
        return out[:, 0], out[:, 1], cache      # select_logits, budget_logits

    def backward(self, cache, d_select, d_budget):
        X, h = cache
        d_out = np.stack([d_select, d_budget], axis=1)
        dW2 = h.T @ d_out
        db2 = d_out.sum(0)
        dh = (d_out @ self.W2.T) * (1 - h ** 2)
        dW1 = X.T @ dh
        db1 = dh.sum(0)
        return [dW1, db1, dW2, db2]


class ValueNet:
    """State-value critic V(s).  Input is a permutation-invariant pooling of the
    per-client state matrix: [global(3), mean/max/min over clients of the 4
    dynamic features] -> 15 dims."""

    IN_DIM = 15

    def __init__(self, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(self.IN_DIM), (self.IN_DIM, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    @staticmethod
    def pool(state: np.ndarray) -> np.ndarray:
        g = state[0, :3]
        dyn = state[:, 3:]
        return np.concatenate([g, dyn.mean(0), dyn.max(0), dyn.min(0)])

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        v = float((h @ self.W2 + self.b2)[0])
        return v, (x, h)

    def backward(self, cache, dv):
        x, h = cache
        dW2 = np.outer(h, [dv])
        db2 = np.array([dv])
        dh = (self.W2[:, 0] * dv) * (1 - h ** 2)
        dW1 = np.outer(x, dh)
        db1 = dh
        return [dW1, db1, dW2, db2]


class ReinforceController:
    def __init__(self, env, *, lr: float = 0.01, gamma: float = 0.98,
                 entropy_coeff: float = 0.01, use_critic: bool = False,
                 critic_lr: float = 0.02, seed: int = 0):
        self.env = env
        self.pi = PolicyNet(env.feature_dim, seed=seed)
        self.lr = lr
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.use_critic = use_critic
        self.critic_lr = critic_lr
        self.critic = ValueNet(seed=seed + 2) if use_critic else None
        self.baseline = 0.0
        self.rng = np.random.default_rng(seed + 1)

    # -- action sampling ------------------------------------------------
    def act(self, state, greedy: bool = False):
        k = self.env.clients_per_round
        sel_logits, bud_logits, cache = self.pi.forward(state)

        probs = _softmax(sel_logits)
        if greedy:
            chosen = list(np.argsort(-probs)[:k])
            logp_sel = 0.0
        else:
            chosen, logp_sel = [], 0.0
            avail = list(range(len(probs)))
            lg = sel_logits.copy()
            for _ in range(k):
                p = _softmax(lg[avail])
                j = self.rng.choice(len(avail), p=p)
                logp_sel += np.log(p[j] + 1e-12)
                chosen.append(avail.pop(j))
        chosen = np.array(chosen, dtype=int)

        bshare = _softmax(bud_logits[chosen])
        extra = self.env.epoch_budget - k
        epochs = np.maximum(1, np.round(1 + extra * bshare)).astype(int)

        return chosen, epochs, {"cache": cache, "logp_sel": logp_sel,
                                "sel_probs": probs, "chosen": chosen,
                                "sel_logits": sel_logits,
                                "pooled": ValueNet.pool(state)}

    # -- episode rollout ---------------------------------------------
    def _rollout(self, greedy: bool):
        state = self.env.reset()
        traj, rewards = [], []
        done = False
        info = {}
        while not done:
            chosen, epochs, meta = self.act(state, greedy=greedy)
            state, r, done, info = self.env.step(chosen, epochs)
            traj.append(meta)
            rewards.append(r)
        return traj, rewards, info

    def run_episode(self, train: bool = True, rollouts: int = 1):
        """One optimisation step.

        With `rollouts` > 1, several independent trajectories are collected and
        their gradients are averaged before a single parameter update -- this is
        the multi-rollout variance-reduction that stabilises REINFORCE here.
        """
        if not train:
            _, rewards, info = self._rollout(greedy=True)
            return sum(rewards), info

        batch = [self._rollout(greedy=False) for _ in range(rollouts)]
        ep_returns = [sum(rw) for _, rw, _ in batch]
        self._update([(t, rw) for t, rw, _ in batch])
        return float(np.mean(ep_returns)), batch[-1][2]

    # -- REINFORCE update (averaged over a batch of trajectories) --------
    def _update(self, trajectories):
        # shared baseline across the whole batch for lower-variance advantages
        all_returns = []
        per_traj = []
        for traj, rewards in trajectories:
            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.append(G)
            returns = np.array(returns[::-1])
            per_traj.append((traj, returns))
            all_returns.append(returns)
        flat = np.concatenate(all_returns)
        self.baseline = 0.9 * self.baseline + 0.1 * flat.mean()
        std = flat.std() + 1e-8

        # -- critic: fit V(s) to the observed returns, use it as the baseline ---
        if self.use_critic:
            cgrads = [np.zeros_like(p) for p in self.critic.params()]
            cn = 0
            advs = []
            for traj, returns in per_traj:
                a_traj = []
                for meta, G in zip(traj, returns):
                    v, cache = self.critic.forward(meta["pooled"])
                    a_traj.append(G - v)
                    dv = np.clip(v - G, -50.0, 50.0)      # dMSE/dv
                    for i, gi in enumerate(self.critic.backward(cache, dv)):
                        cgrads[i] += gi
                    cn += 1
                advs.append(np.array(a_traj))
            for p, g in zip(self.critic.params(), cgrads):
                np.clip(g, -5.0, 5.0, out=g)
                p -= self.critic_lr * g / max(cn, 1)
            adv_std = np.concatenate(advs).std() + 1e-8

        grads = [np.zeros_like(p) for p in self.pi.params()]
        n_steps = 0
        for ti, (traj, returns) in enumerate(per_traj):
            if self.use_critic:
                adv = advs[ti] / adv_std
            else:
                adv = (returns - self.baseline) / std
            for meta, a in zip(traj, adv):
                probs = meta["sel_probs"]
                chosen = meta["chosen"]
                d_select = -probs.copy()
                d_select[chosen] += 1.0
                d_select *= a
                ent_grad = -(np.log(probs + 1e-12) + 1.0)
                d_select += self.entropy_coeff * (ent_grad - ent_grad.mean()) * probs

                d_budget = np.zeros(len(probs))
                d_budget[chosen] += 0.1 * a

                gs = self.pi.backward(meta["cache"], d_select, d_budget)
                for i in range(len(grads)):
                    grads[i] += gs[i]
                n_steps += 1

        for p, g in zip(self.pi.params(), grads):
            np.clip(g, -5.0, 5.0, out=g)
            p += self.lr * g / max(n_steps, 1)


# ---------------------------------------------------------------------
class HeuristicController:
    """Non-learning baselines for comparison."""

    def __init__(self, env, kind: str = "random", seed: int = 0):
        self.env = env
        self.kind = kind
        self.rng = np.random.default_rng(seed)

    def run_episode(self, train: bool = False):
        state = self.env.reset()
        k = self.env.clients_per_round
        total = 0.0
        done = False
        info = {}
        while not done:
            n = self.env.n_clients
            if self.kind == "random":
                chosen = self.rng.choice(n, size=k, replace=False)
            elif self.kind == "all":
                chosen = np.arange(min(k, n))
            elif self.kind == "fraud_greedy":
                chosen = np.argsort(-self.env.client_desc[:, 1])[:k]
            else:
                raise ValueError(self.kind)
            epochs = np.full(len(chosen), self.env.epoch_budget // len(chosen))
            state, r, done, info = self.env.step(chosen, epochs)
            total += r
        return total, info
