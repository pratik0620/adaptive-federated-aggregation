"""Central federated server coordinating standard FedAvg communication rounds.

Orchestrates:
  - Broadcast of identical global model parameters
  - Collection of client updates
  - Aggregation using standard FedAvg (decoupled aggregation function)
  - Validation evaluation across all participating and registered clients
  - Round checkpointing, metrics export (CSV/JSON), and run metadata persistence
"""

from __future__ import annotations

import copy
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..models.resnet import save_checkpoint
from .aggregation import aggregate_capped_fedavg, aggregate_fedavg
from .client import FederatedClient
from .config import FederatedConfig


class FederatedServer:
    """Central server orchestrator for federated training rounds."""

    def __init__(
        self,
        global_model: nn.Module,
        clients: dict[str, FederatedClient] | list[FederatedClient] | None = None,
        config: FederatedConfig | None = None,
        aggregation_fn: Callable = aggregate_fedavg,
        device: str | torch.device = "cpu",
        output_dir: str | Path = "outputs/federated",
    ) -> None:
        """Initialize central federated server.

        Args:
            global_model: Initial shared PyTorch model.
            clients: Registered clients (dict mapping client_id -> client or list of clients).
            config: Federated configuration instance.
            aggregation_fn: Parameter aggregation function (defaults to standard FedAvg).
            device: Computation device.
            output_dir: Directory where checkpoints and metrics are saved.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.global_model = global_model.to(self.device)
        self.config = config or FederatedConfig()

        if aggregation_fn is aggregate_fedavg and self.config.aggregation_method == "capped_fedavg":
            if self.config.cap_value is None:
                raise ValueError("cap_value must be provided when aggregation_method is 'capped_fedavg'.")
            cap = self.config.cap_value
            self.aggregation_fn = lambda client_states, sample_counts, client_ids=None: aggregate_capped_fedavg(
                client_states=client_states,
                sample_counts=sample_counts,
                cap_value=cap,
                client_ids=client_ids,
            )
        else:
            self.aggregation_fn = aggregation_fn

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clients: dict[str, FederatedClient] = {}
        if clients is not None:
            if isinstance(clients, dict):
                self.clients = dict(clients)
            elif isinstance(clients, (list, tuple)):
                for c in clients:
                    self.register_client(c)

        self.round_history: list[dict[str, Any]] = []
        self.best_metric_value: float = -float("inf")
        self.best_round: int = -1

    def register_client(self, client: FederatedClient) -> None:
        """Register a federated client node."""
        self.clients[client.client_id] = client

    def get_global_parameters(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU clone of current global parameters."""
        raw_model = self.global_model.module if hasattr(self.global_model, "module") else self.global_model
        return {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}

    def broadcast_parameters(self, client_ids: Sequence[str] | None = None) -> None:
        """Distribute identical global parameters to specified or all registered clients.

        Args:
            client_ids: Target client IDs. If None, broadcasts to all registered clients.
        """
        global_params = self.get_global_parameters()
        targets = client_ids if client_ids is not None else list(self.clients.keys())
        for cid in targets:
            if cid in self.clients:
                self.clients[cid].set_parameters(global_params)

    def run_round(
        self,
        round_idx: int,
        participating_client_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Execute a single communication round of federated learning.

        Workflow:
          1. Select participating clients.
          2. Broadcast identical global parameters to participating clients.
          3. Perform local training on each participating client.
          4. Collect locally updated parameters, sample counts, and training metadata.
          5. Aggregate parameters using standard FedAvg (p_k = n_k / N).
          6. Update the central global model with aggregated parameters.
          7. Broadcast updated parameters and evaluate validation splits for each client.
          8. Record round metrics, weights, and save round checkpoint.

        Args:
            round_idx: 1-indexed communication round number.
            participating_client_ids: Subset of clients participating this round.

        Returns:
            Dictionary summarizing round results, weights, and metrics.
        """
        round_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"  Federated Communication Round [{round_idx}/{self.config.num_rounds}]")
        print(f"{'='*70}")

        if participating_client_ids is None:
            active_cids = list(self.clients.keys())
        else:
            active_cids = [cid for cid in participating_client_ids if cid in self.clients]

        if not active_cids:
            # Fall back to all registered clients if none matched
            active_cids = list(self.clients.keys())

        if not active_cids:
            raise ValueError("No active clients available to participate in this round.")

        print(f"Participating clients ({len(active_cids)}): {', '.join(active_cids)}")

        # Step 1: Broadcast identical parameters to active clients
        self.broadcast_parameters(active_cids)

        # Step 2: Train locally on each active client
        client_states: list[dict[str, torch.Tensor]] = []
        sample_counts: list[int] = []
        train_summaries: dict[str, Any] = {}

        for cid in active_cids:
            client = self.clients[cid]
            res = client.fit_round(
                local_epochs=self.config.local_epochs,
                lr=self.config.lr,
                optimizer_name=self.config.optimizer,
                weight_decay=self.config.weight_decay,
            )
            client_states.append(res["parameters"])
            sample_counts.append(res["sample_count"])
            train_summaries[cid] = res["train_metrics"]
            print(
                f"  [{cid}] Completed {self.config.local_epochs} local epoch(s) | "
                f"Samples (n_k): {res['sample_count']:,} | "
                f"Train Loss: {res['train_metrics']['loss']:.4f}"
            )

        # Step 3: Central Aggregation (FedAvg)
        aggregated_state, aggregation_weights = self.aggregation_fn(
            client_states=client_states,
            sample_counts=sample_counts,
            client_ids=active_cids,
        )

        # Step 4: Update central global model
        raw_model = self.global_model.module if hasattr(self.global_model, "module") else self.global_model
        raw_model.load_state_dict({k: v.to(self.device) for k, v in aggregated_state.items()})

        # Step 5: Evaluate updated global model on validation split of each registered client
        val_evaluations: dict[str, Any] = {}
        valid_aurocs: list[float] = []

        print("\n  Validation Evaluation on Updated Global Model:")
        for cid, client in self.clients.items():
            val_metrics = client.evaluate(model_state_dict=aggregated_state, split="val")
            val_evaluations[cid] = val_metrics
            auc_str = f"{val_metrics['auroc']:.4f}" if val_metrics.get("auroc") is not None else "N/A"
            loss_str = f"{val_metrics.get('loss', 0.0):.4f}"
            print(
                f"    * {cid:20s} | Val AUROC: {auc_str:7s} | Val Loss: {loss_str} | "
                f"Status: {val_metrics.get('evaluation_status', 'N/A')}"
            )
            if val_metrics.get("auroc") is not None:
                valid_aurocs.append(val_metrics["auroc"])

        mean_val_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else None

        round_duration = time.time() - round_start_time

        # Compile round summary
        round_summary = {
            "round": round_idx,
            "experiment_name": self.config.experiment_name,
            "aggregation_method": self.config.aggregation_method,
            "cap_value": self.config.cap_value,
            "seed": self.config.seed,
            "initialization_id": self.config.initialization_id,
            "duration_seconds": round_duration,
            "participating_clients": active_cids,
            "sample_counts": {cid: n_k for cid, n_k in zip(active_cids, sample_counts)},
            "aggregation_weights": aggregation_weights,
            "mean_val_auroc": mean_val_auroc,
            "train_metrics": train_summaries,
            "val_metrics": val_evaluations,
        }
        self.round_history.append(round_summary)

        # Save round checkpoint
        round_ckpt_path = self.output_dir / f"global_checkpoint_round_{round_idx:02d}.pt"
        save_checkpoint(
            checkpoint_path=round_ckpt_path,
            model=self.global_model,
            epoch=round_idx,
            best_val_metric=mean_val_auroc,
            metric_name="mean_val_auroc",
            config=self.config.to_dict(),
        )

        # Check for new best checkpoint
        current_metric = mean_val_auroc if mean_val_auroc is not None else -1.0
        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            self.best_round = round_idx
            best_ckpt_path = self.output_dir / "best_global_checkpoint.pt"
            save_checkpoint(
                checkpoint_path=best_ckpt_path,
                model=self.global_model,
                epoch=round_idx,
                best_val_metric=self.best_metric_value,
                metric_name="mean_val_auroc",
                config=self.config.to_dict(),
            )
            print(f"  --> Saved new best global checkpoint at round {round_idx} (Mean Val AUROC: {current_metric:.4f})")

        return round_summary

    def run_federated(self, num_rounds: int | None = None) -> list[dict[str, Any]]:
        """Run all federated learning communication rounds.

        Saves:
          - round_metrics.json: Full structured log of all rounds.
          - history.csv: Tabular metric summary per round.
          - federated_config.json: Configuration record.
          - final_global_checkpoint.pt: Final parameters.
          - best_global_checkpoint.pt: Checkpoint with optimal validation AUROC.
        """
        rounds = num_rounds or self.config.num_rounds
        print(f"\n[Federated Server] Starting {rounds} rounds of standard FedAvg training.")
        total_start = time.time()

        # Save config
        self.config.save_json(self.output_dir / "federated_config.json")

        for r in range(1, rounds + 1):
            self.run_round(
                round_idx=r,
                participating_client_ids=self.config.participating_clients,
            )

        total_time = time.time() - total_start
        print(f"\n[Federated Training Complete] Completed {rounds} rounds in {total_time:.1f}s.")
        print(f"Best Round: {self.best_round} | Best Mean Val AUROC: {self.best_metric_value:.4f}")

        # Save final checkpoint
        save_checkpoint(
            checkpoint_path=self.output_dir / "final_global_checkpoint.pt",
            model=self.global_model,
            epoch=rounds,
            best_val_metric=self.best_metric_value,
            metric_name="mean_val_auroc",
            config=self.config.to_dict(),
        )

        # Export run metadata
        metadata = {
            "experiment_name": self.config.experiment_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "seed": self.config.seed,
            "initialization_id": self.config.initialization_id,
            "aggregation_method": self.config.aggregation_method,
            "cap_value": self.config.cap_value,
            "total_rounds_completed": rounds,
            "total_training_time_seconds": total_time,
            "best_round": self.best_round,
            "best_metric_value": self.best_metric_value,
            "metric_optimized": self.config.metric_to_optimize,
            "checkpoint_paths": {
                "best_checkpoint": str(self.output_dir / "best_global_checkpoint.pt"),
                "final_checkpoint": str(self.output_dir / "final_global_checkpoint.pt"),
            },
            "config": self.config.to_dict(),
        }
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        # Export round_metrics.json
        (self.output_dir / "round_metrics.json").write_text(
            json.dumps(self.round_history, indent=2, default=str), encoding="utf-8"
        )

        # Export history.csv
        rows: list[dict[str, Any]] = []
        for r_data in self.round_history:
            row: dict[str, Any] = {
                "round": r_data["round"],
                "experiment_name": r_data.get("experiment_name"),
                "aggregation_method": r_data.get("aggregation_method"),
                "cap_value": r_data.get("cap_value"),
                "seed": r_data.get("seed"),
                "duration_seconds": r_data["duration_seconds"],
                "mean_val_auroc": r_data.get("mean_val_auroc"),
            }
            # Add client weights and metrics
            for cid, w in r_data.get("aggregation_weights", {}).items():
                row[f"{cid}_weight"] = w
            for cid, n_k in r_data.get("sample_counts", {}).items():
                row[f"{cid}_samples"] = n_k
            for cid, v_metrics in r_data.get("val_metrics", {}).items():
                row[f"{cid}_val_loss"] = v_metrics.get("loss")
                row[f"{cid}_val_auroc"] = v_metrics.get("auroc")
                row[f"{cid}_val_pr_auc"] = v_metrics.get("pr_auc")
                row[f"{cid}_val_f1"] = v_metrics.get("f1")
                row[f"{cid}_val_balanced_accuracy"] = v_metrics.get("balanced_accuracy")
            rows.append(row)

        history_df = pd.DataFrame(rows)
        history_df.to_csv(self.output_dir / "history.csv", index=False)

        return self.round_history

    def evaluate_test_benchmark(
        self,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate the specified global checkpoint on held-out test splits.

        IMPORTANT RESEARCH RULE:
          Test split evaluation must ONLY be invoked after all training rounds and
          hyperparameter/cap selection are completely finalized. Test data must never
          influence model selection, threshold sweeps, or aggregation weights.
        """
        target_path = Path(checkpoint_path) if checkpoint_path else (self.output_dir / "best_global_checkpoint.pt")
        if not target_path.is_file():
            target_path = self.output_dir / "final_global_checkpoint.pt"
        if not target_path.is_file():
            raise FileNotFoundError(f"Global checkpoint not found at: {target_path}")

        print(f"\n[Test Evaluation] Evaluating global checkpoint on held-out test splits: {target_path.name}")
        ckpt = torch.load(target_path, map_location=self.device)
        model_state = ckpt.get("model_state_dict", ckpt)

        test_results: dict[str, Any] = {}
        for cid, client in self.clients.items():
            t_metrics = client.evaluate(model_state_dict=model_state, split="test")
            test_results[cid] = t_metrics
            auc_str = f"{t_metrics['auroc']:.4f}" if t_metrics.get("auroc") is not None else "N/A"
            print(f"  * {cid:20s} | Test AUROC: {auc_str:7s} | Status: {t_metrics.get('evaluation_status', 'N/A')}")

        test_payload = {
            "checkpoint_evaluated": str(target_path),
            "experiment_name": self.config.experiment_name,
            "aggregation_method": self.config.aggregation_method,
            "cap_value": self.config.cap_value,
            "seed": self.config.seed,
            "clients": test_results,
        }

        (self.output_dir / "test_metrics.json").write_text(
            json.dumps(test_payload, indent=2, default=str), encoding="utf-8"
        )
        return test_payload
