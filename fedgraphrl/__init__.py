"""FedGraph-RL: RL-orchestrated federated Graph Neural Network fraud detection."""
from .data import make_transaction_graph, partition_non_iid, GraphData
from .payment_data import make_payment_graph, partition_by_bank, ScamEpisode
from .gnn import GCN
from .federated import FederatedClient, FederatedServer, fedavg
from .environment import FederatedEnv
from .rl_controller import ReinforceController, HeuristicController
from .metrics import binary_scores, money_weighted_scores

__version__ = "0.3.0.dev0"   # last released: 0.1.0 (see CHANGELOG); 0.2.x / 0.3.x unreleased
__all__ = [
    "make_transaction_graph", "partition_non_iid", "GraphData",
    "make_payment_graph", "partition_by_bank", "ScamEpisode", "GCN",
    "FederatedClient", "FederatedServer", "fedavg", "FederatedEnv",
    "ReinforceController", "HeuristicController",
    "binary_scores", "money_weighted_scores",
]
