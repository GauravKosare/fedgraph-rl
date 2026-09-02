from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # graph / data
    num_accounts: int = 1200
    num_features: int = 16
    fraud_ring_count: int = 12
    num_clients: int = 8
    dirichlet_alpha: float = 0.10       # lower => more non-IID
    seed: int = 7

    # model
    hidden: int = 32
    dropout: float = 0.5

    # federated / RL env
    max_rounds: int = 25
    clients_per_round: int = 4
    epoch_budget: int = 12             # total local epochs to split each round
    cost_coeff: float = 0.06

    # v0.2 -- stragglers & per-round wall-clock deadline (opt-in; v0.1 canonical run keeps this off)
    stragglers: bool = False           # heterogeneous device speeds + late-update drop
    straggler_frac: float = 0.35       # fraction of clients that are "slow"
    straggler_slowdown: float = 3.0    # slow clients run this many x slower
    deadline_slack: float = 1.6        # round deadline = slack x (fast client's fair-share time)
    drop_penalty: float = 0.6          # v0.2.1: dense reward penalty per dropped (late) update

    # v0.2.1 -- REINFORCE stabilisation
    adv_std_floor: float = 1.0         # floor on the advantage-normalisation denominator
    adv_clip: float = 8.0              # clip normalised advantages to +/- this
    entropy_final_frac: float = 0.1    # entropy_coeff decays linearly to this fraction by the last episode

    # RL training
    episodes: int = 50
    rollouts_per_update: int = 4       # trajectories averaged per REINFORCE step
    lr: float = 0.02
    entropy_coeff: float = 0.02
    use_critic: bool = True            # learned V(s) baseline (actor-critic) vs moving average
    critic_lr: float = 0.03

    def model_cfg(self) -> dict:
        return {"hidden": self.hidden, "dropout": self.dropout, "n_classes": 2}

    def to_dict(self) -> dict:
        return asdict(self)
