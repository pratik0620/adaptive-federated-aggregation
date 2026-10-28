"""Federated learning baseline package for cross-silo thoracic disease prediction."""

from .aggregation import aggregate_fedavg, compute_fedavg_weights, validate_client_state_dicts
from .client import FederatedClient, SyntheticFederatedClient
from .config import FederatedConfig, create_federated_arg_parser
from .server import FederatedServer

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "SyntheticFederatedClient",
    "FederatedConfig",
    "create_federated_arg_parser",
    "aggregate_fedavg",
    "compute_fedavg_weights",
    "validate_client_state_dicts",
]
