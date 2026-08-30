"""FedGraph-RL: RL-orchestrated federated Graph Neural Network fraud detection."""
from .data import make_transaction_graph, partition_non_iid, GraphData
from .gnn import GCN
from .federated import FederatedClient, FederatedServer, fedavg
from .environment import FederatedEnv
from .rl_controller import ReinforceController, HeuristicController

__version__ = "0.1.0"
__all__ = [
    "make_transaction_graph", "partition_non_iid", "GraphData", "GCN",
    "FederatedClient", "FederatedServer", "fedavg", "FederatedEnv",
    "ReinforceController", "HeuristicController",
]
