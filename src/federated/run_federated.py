"""CLI runner for three-client federated learning baseline using FedAvg.

Usage:
  # Synthetic pilot dry-run (2 clients, 2 rounds, no images needed):
  python src/federated/run_federated.py --synthetic --rounds 2 --clients client_01 client_02

  # Small pilot experiment on real datasets (e.g. 50 patients per client, 3 rounds):
  python src/federated/run_federated.py --patient_limit 50 --rounds 3 --local_epochs 1

  # Full federated baseline (with config file):
  python src/federated/run_federated.py --config configs/federated_baseline.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.federated.adapters import (
    build_chexpert_client,
    build_nih_client,
    build_padchest_client,
)
from src.federated.aggregation import aggregate_fedavg
from src.federated.client import FederatedClient, SyntheticFederatedClient
from src.federated.config import FederatedConfig, create_federated_arg_parser
from src.federated.server import FederatedServer
from src.models.resnet import build_resnet50_binary


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = create_federated_arg_parser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic baseline with in-memory generated data (no images or GPU required).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration and clients without executing training rounds.",
    )
    parser.add_argument(
        "--evaluate_test",
        action="store_true",
        help="Execute held-out test split evaluation after all communication rounds complete.",
    )
    args = parser.parse_args()

    # Load external config if provided
    config_dict: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        config_dict = json.loads(cfg_path.read_text(encoding="utf-8"))

    # CLI args override file config
    for k in (
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
        cli_val = getattr(args, k, None)
        if cli_val is not None:
            config_dict[k] = cli_val

    fed_config = FederatedConfig.from_dict(config_dict)
    set_seed(fed_config.seed)

    # Determine execution device
    device_str = fed_config.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Device Warning] CUDA requested but unavailable. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if fed_config.aggregation_method == "capped_fedavg":
        if fed_config.cap_value is None:
            print("[Config Notice] aggregation_method='capped_fedavg' specified without --cap_value. Defaulting cap to 0.40.")
            fed_config.cap_value = 0.40
        else:
            from src.federated.aggregation import validate_cap_value
            fed_config.cap_value = validate_cap_value(fed_config.cap_value)
        prefix = f"capped_fedavg_c{int(fed_config.cap_value * 100)}"
    else:
        prefix = "fedavg"

    out_dir = Path(fed_config.output_dir) / f"{prefix}_{timestamp}"
    fed_config.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fed_config.participating_clients is None:
        if args.synthetic:
            fed_config.participating_clients = ["client_01", "client_02", "client_03"]
        else:
            fed_config.participating_clients = ["client_01_NIH", "client_02_CheXpert", "client_03_PadChest"]

    print("=" * 70)
    if fed_config.aggregation_method == "capped_fedavg":
        print(f"  Federated Learning Experiment: Capped FedAvg (Cap = {fed_config.cap_value})")
    else:
        print("  Federated Learning Baseline: Standard FedAvg")
    print("=" * 70)
    print(f"Output Directory:       {out_dir}")
    print(f"Aggregation Method:     {fed_config.aggregation_method.upper()}")
    if fed_config.cap_value is not None:
        print(f"Cap Value (c):          {fed_config.cap_value}")
    print(f"Device:                 {device}")
    print(f"Rounds:                 {fed_config.num_rounds}")
    print(f"Local Epochs / Round:   {fed_config.local_epochs}")
    print(f"Batch Size:             {fed_config.batch_size}")
    print(f"Learning Rate:          {fed_config.lr}")
    print(f"Optimizer:              {fed_config.optimizer.upper()}")
    print(f"Patient Limit / Client: {fed_config.patient_limit}")
    print(f"Target Clients:         {fed_config.participating_clients}")
    print(f"Synthetic Mode:         {args.synthetic}")
    print("=" * 70)

    # Execute strict preflight validation
    from src.federated.preflight import PreflightError, run_federated_preflight

    print("\n[Preflight] Running federated experiment preflight checks...")
    try:
        preflight_summary = run_federated_preflight(
            config=fed_config,
            repo_root=REPO_ROOT,
            is_synthetic=args.synthetic,
            abort_on_failure=(not args.dry_run),
        )
        print(preflight_summary.format_report())
        if args.dry_run:
            if not preflight_summary.is_ready:
                print("\n[Dry-Run Notice] Preflight detected blocking issues above. Halting before training.", file=sys.stderr)
                sys.exit(1)
            else:
                print("\n[Dry-Run Complete] All preflight checks passed. Ready for training.")
                return
    except PreflightError as exc:
        print(f"\n[PREFLIGHT HALTED] {exc}", file=sys.stderr)
        sys.exit(1)

    clients: list[FederatedClient] = []

    if args.synthetic:
        # Build lightweight synthetic model and clients
        class SyntheticModel(nn.Module):
            def __init__(self, in_features: int = 16):
                super().__init__()
                self.fc = nn.Linear(in_features, 1)
                self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.training:
                    self.num_batches_tracked += 1
                return self.fc(x)

        global_model = SyntheticModel(in_features=16)

        for i, cid in enumerate(fed_config.participating_clients):
            c_model = SyntheticModel(in_features=16)
            c = SyntheticFederatedClient(
                client_id=cid,
                num_train_samples=50 + i * 25,
                num_val_samples=20,
                input_dim=16,
                model=c_model,
                seed=fed_config.seed + i * 10,
                device=device,
                config=fed_config.to_dict(),
            )
            clients.append(c)

    else:
        # Real healthcare clients (ResNet-50)
        try:
            global_model = build_resnet50_binary(pretrained=True, dropout_rate=0.0, device=device)
        except ImportError as exc:
            print(f"\n[Dependency Blocker] Failed to instantiate ResNet-50: {exc}")
            print("To run with synthetic data instead, re-run with: --synthetic")
            sys.exit(1)

        client_builders = {
            "client_01_NIH": build_nih_client,
            "NIH": build_nih_client,
            "client_02_CheXpert": build_chexpert_client,
            "CheXpert": build_chexpert_client,
            "client_03_PadChest": build_padchest_client,
            "PadChest": build_padchest_client,
        }

        for cid in fed_config.participating_clients:
            builder = client_builders.get(cid)
            if builder is None:
                raise ValueError(f"Unknown client identifier: '{cid}'. Expected one of {list(client_builders.keys())}")
            try:
                c_model = build_resnet50_binary(pretrained=True, dropout_rate=0.0, device=device)
                client = builder(
                    repo_root=REPO_ROOT,
                    patient_limit=fed_config.patient_limit,
                    batch_size=fed_config.batch_size,
                    seed=fed_config.seed,
                    device=device,
                    model=c_model,
                    config=fed_config.to_dict(),
                )
                clients.append(client)
                print(f"[Client Loaded] {client.client_id} with {client.get_sample_count():,} training samples.")
            except Exception as exc:
                print(f"[Client Warning] Failed to initialize client '{cid}': {exc}")
                raise

    if args.dry_run:
        print("\n[Dry-Run] All clients initialized successfully. Exiting without training rounds.")
        return

    # Initialize central server
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        config=fed_config,
        device=device,
        output_dir=out_dir,
    )

    # Execute federated learning rounds
    server.run_federated()

    if args.evaluate_test:
        print("\n[Evaluation Stage] Executing final held-out test split benchmark...")
        server.evaluate_test_benchmark()


if __name__ == "__main__":
    main()
