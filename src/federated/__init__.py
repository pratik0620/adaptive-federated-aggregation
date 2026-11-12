"""Federated learning baseline package for cross-silo thoracic disease prediction."""

from .aggregation import (
    aggregate_capped_fedavg,
    aggregate_fedavg,
    aggregate_parameters,
    compute_capped_fedavg_weights,
    compute_fedavg_weights,
    validate_cap_value,
    validate_client_state_dicts,
)
from .client import FederatedClient, SyntheticFederatedClient
from .config import FederatedConfig, create_federated_arg_parser, resolve_federated_config
from .server import FederatedServer

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "SyntheticFederatedClient",
    "FederatedConfig",
    "create_federated_arg_parser",
    "resolve_federated_config",
    "aggregate_fedavg",
    "aggregate_capped_fedavg",
    "aggregate_parameters",
    "compute_fedavg_weights",
    "compute_capped_fedavg_weights",
    "validate_cap_value",
    "validate_client_state_dicts",
]
