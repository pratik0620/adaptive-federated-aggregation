"""Configuration dataclass and CLI argument parsing for federated baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FederatedConfig:
    """Configuration parameters for the three-client federated learning baseline."""

    experiment_name: str = "federated_baseline_fedavg"
    initialization_id: str = "imagenet_pretrained_resnet50"
    aggregation_method: str = "fedavg"
    cap_value: float | None = None
    num_rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-4
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    seed: int = 42
    device: str = "cpu"
    participating_clients: list[str] | None = None
    output_dir: str = "outputs/federated"
    patient_limit: int | None = None
    pos_weight: float | None = None
    metric_to_optimize: str = "mean_val_auroc"
    model_architecture: str = "resnet50"
    num_classes: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Save configuration dictionary to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FederatedConfig:
        """Instantiate FederatedConfig from dictionary, ignoring unknown keys."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load_json(cls, path: str | Path) -> FederatedConfig:
        """Load configuration from a JSON file."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def create_federated_arg_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for federated training."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Baseline using Standard FedAvg"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file overriding default settings.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment run identifier (default: federated_baseline_fedavg).",
    )
    parser.add_argument(
        "--initialization_id",
        type=str,
        default=None,
        help="Identifier for the initial model state or checkpoint (default: imagenet_pretrained_resnet50).",
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default=None,
        choices=["fedavg", "capped_fedavg"],
        help="Parameter aggregation algorithm (default: fedavg).",
    )
    parser.add_argument(
        "--cap_value",
        type=float,
        default=None,
        help="Optional pre-normalization clipping threshold c in (0, 1.0] (for Capped FedAvg).",
    )
    parser.add_argument(
        "--rounds",
        dest="num_rounds",
        type=int,
        default=None,
        help="Total number of federated communication rounds (default: 5).",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=None,
        help="Number of local training epochs per client per round (default: 1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Mini-batch size for client local training (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Local learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adamw", "sgd", "adam"],
        help="Client optimizer (default: adamw).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cpu' or 'cuda', default: cpu).",
    )
    parser.add_argument(
        "--clients",
        nargs="+",
        dest="participating_clients",
        default=None,
        help="List of participating client identifiers. If omitted, all registered clients participate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and metrics (default: outputs/federated).",
    )
    parser.add_argument(
        "--patient_limit",
        type=int,
        default=None,
        help="Optional max unique patients per client (for pilot experiments).",
    )
    return parser


def resolve_federated_config(args: argparse.Namespace) -> FederatedConfig:
    """Resolve FederatedConfig from optional JSON config with explicit CLI overrides.

    Args:
        args: Parsed CLI namespace from ``create_federated_arg_parser()``.

    Returns:
        Resolved ``FederatedConfig`` where JSON values are preserved unless
        the corresponding CLI argument was explicitly provided.
    """
    config_dict: dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        config_dict = json.loads(cfg_path.read_text(encoding="utf-8"))

    for key in (
        "experiment_name",
        "initialization_id",
        "aggregation_method",
        "cap_value",
        "num_rounds",
        "local_epochs",
        "batch_size",
        "lr",
        "optimizer",
        "seed",
        "device",
        "output_dir",
        "patient_limit",
        "participating_clients",
    ):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            config_dict[key] = cli_val

    return FederatedConfig.from_dict(config_dict)
