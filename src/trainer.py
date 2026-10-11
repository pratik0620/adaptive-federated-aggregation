"""Trainer engine for reproducible NIH ChestX-ray14 binary classification.

Handles epoch loops, mixed-precision training, validation-based model selection,
checkpoint saving in PyTorch state_dict format, test evaluation, and artifact export.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_binary_metrics, format_metrics_summary
from .models.resnet import load_checkpoint, save_checkpoint


class ClientTrainer:
    """Orchestrates model training, validation, checkpointing, and evaluation across federated clients."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        config: dict[str, Any] | None = None,
        output_dir: str | Path = "outputs/client/run",
        device: str | torch.device = "cuda",
    ) -> None:
        self.config = config or {}
        self.dataset_name = str(self.config.get("dataset_name", self.config.get("dataset", "NIH")))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if isinstance(device, str):
            if device == "cuda" and not torch.cuda.is_available():
                print("[Device] CUDA requested but not available; falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Training hyperparameters
        self.lr = float(self.config.get("lr", 1e-4))
        self.weight_decay = float(self.config.get("weight_decay", 1e-4))
        self.epochs = int(self.config.get("epochs", 10))
        self.use_amp = bool(self.config.get("use_amp", False)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Optimizer
        opt_name = str(self.config.get("optimizer", "adamw")).lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "sgd":
            momentum = float(self.config.get("momentum", 0.9))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=momentum, weight_decay=self.weight_decay
            )
        else:  # default adamw
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # Scheduler
        sched_name = self.config.get("scheduler", "cosine")
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.epochs)
            )
        elif sched_name == "step":
            step_size = int(self.config.get("step_size", 5))
            gamma = float(self.config.get("gamma", 0.1))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=2
            )
        else:
            self.scheduler = None

        # Loss function: BCEWithLogitsLoss for single-logit binary classification
        pos_weight = self.config.get("pos_weight")
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([float(pos_weight)], device=self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.metric_to_optimize = self.config.get("metric_to_optimize", "val_auroc")
        self.best_metric_value: float = -float("inf")
        self.best_epoch: int = -1
        self.best_val_metrics: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def _train_one_epoch(self, epoch: int) -> tuple[float, float | None]:
        self.model.train()
        total_loss = 0.0
        all_targets: list[int] = []
        all_probs: list[float] = []

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images).squeeze(-1)
                    loss = self.loss_fn(logits, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images).squeeze(-1)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.detach().cpu().numpy().astype(int).tolist())

        avg_loss = total_loss / max(1, len(all_targets))
        metrics = compute_binary_metrics(all_targets, all_probs)
        return avg_loss, metrics.get("auroc")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "val") -> dict[str, Any]:
        """Evaluate model on a given DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_targets: list[int] = []
        all_probs: list[float] = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images).squeeze(-1)
            loss = self.loss_fn(logits, targets)

            total_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.cpu().numpy().astype(int).tolist())

        avg_loss = total_loss / max(1, len(all_targets))
        metrics = compute_binary_metrics(all_targets, all_probs)
        metrics["loss"] = avg_loss
        metrics["split"] = split_name
        return metrics

    def train(self) -> dict[str, Any]:
        """Execute full training loop, model selection, and artifact saving."""
        print(f"\n[Training] Starting {self.dataset_name} ResNet-50 training for {self.epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Train
            train_loss, train_auroc = self._train_one_epoch(epoch)

            # Validate
            val_metrics = self.evaluate(self.val_loader, split_name="val")
            val_loss = val_metrics["loss"]
            val_auroc = val_metrics["auroc"]

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # plateau uses val_auroc (or negative val_loss)
                    score = val_auroc if val_auroc is not None else -val_loss
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            epoch_duration = time.time() - epoch_start

            # Log history row
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auroc": train_auroc,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_pr_auc": val_metrics.get("pr_auc"),
                "val_sensitivity": val_metrics.get("sensitivity"),
                "val_specificity": val_metrics.get("specificity"),
                "val_f1": val_metrics.get("f1"),
                "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
                "lr": current_lr,
                "epoch_time_seconds": epoch_duration,
            }
            self.history.append(history_row)

            # Format status printout
            train_auc_str = f"{train_auroc:.4f}" if train_auroc is not None else "N/A"
            val_auc_str = f"{val_auroc:.4f}" if val_auroc is not None else "N/A"
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train AUROC: {train_auc_str} | "
                f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auc_str} | "
                f"LR: {current_lr:.2e} | Time: {epoch_duration:.1f}s"
            )

            # Model selection: determine if best
            metric_val = val_auroc if self.metric_to_optimize == "val_auroc" else -val_loss
            if metric_val is not None and metric_val > self.best_metric_value:
                self.best_metric_value = metric_val
                self.best_epoch = epoch
                self.best_val_metrics = val_metrics

                # Save best checkpoint
                save_checkpoint(
                    checkpoint_path=self.output_dir / "best_checkpoint.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_val_metric=metric_val,
                    metric_name=self.metric_to_optimize,
                    config=self.config,
                )
                print(f"  --> Saved new best checkpoint at epoch {epoch} (Val AUROC: {val_auc_str})")

        total_training_time = time.time() - start_time
        print(f"\n[Training Complete] Total time: {total_training_time:.1f}s. Best epoch: {self.best_epoch}")

        # Save final checkpoint
        save_checkpoint(
            checkpoint_path=self.output_dir / "final_checkpoint.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epochs,
            best_val_metric=self.best_metric_value,
            metric_name=self.metric_to_optimize,
            config=self.config,
        )

        # Save history CSV
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / "history.csv", index=False)

        # Check validation coverage vs official split size
        total_official_val = int(self.config.get("total_official_val_images", 5836))
        val_evaluated_count = int(self.best_val_metrics.get("total_samples", len(self.val_loader.dataset)))
        is_official_full_val = (val_evaluated_count >= total_official_val)
        val_coverage_pct = (val_evaluated_count / total_official_val * 100.0) if total_official_val > 0 else 0.0

        self.best_val_metrics["is_official_full_val"] = is_official_full_val
        self.best_val_metrics["evaluated_samples"] = val_evaluated_count
        self.best_val_metrics["total_official_split_samples"] = total_official_val
        self.best_val_metrics["coverage_percentage"] = val_coverage_pct
        self.best_val_metrics["missing_samples_count"] = max(0, total_official_val - val_evaluated_count)

        if is_official_full_val:
            self.best_val_metrics["evaluation_status"] = "OFFICIAL_FULL_VALIDATION_BENCHMARK"
        else:
            self.best_val_metrics["evaluation_status"] = "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL"
            disclaimer_val = (
                f"UNOFFICIAL: Validation evaluated on partial accessible subset "
                f"({val_evaluated_count:,} / {total_official_val:,} images, {val_coverage_pct:.2f}% coverage). "
                f"{self.best_val_metrics['missing_samples_count']:,} images missing from local archives."
            )
            self.best_val_metrics["disclaimer"] = disclaimer_val
            self.best_val_metrics.setdefault("warnings", []).append(disclaimer_val)

        # Save validation metrics JSON
        (self.output_dir / "val_metrics.json").write_text(
            json.dumps(self.best_val_metrics, indent=2, default=str), encoding="utf-8"
        )

        # Evaluate on test set ONLY after model selection is finalized
        test_metrics: dict[str, Any] = {}
        if self.test_loader is not None:
            print("\n[Evaluation] Evaluating best checkpoint on held-out test split...")
            best_ckpt_path = self.output_dir / "best_checkpoint.pt"
            if best_ckpt_path.is_file():
                load_checkpoint(best_ckpt_path, model=self.model, device=self.device)

            test_metrics = self.evaluate(self.test_loader, split_name="test")

            # Check coverage vs official split size
            total_official_test = int(self.config.get("total_official_test_images", 14519))
            evaluated_count = int(test_metrics.get("total_samples", len(self.test_loader.dataset)))
            is_official_full = (evaluated_count >= total_official_test)
            coverage_pct = (evaluated_count / total_official_test * 100.0) if total_official_test > 0 else 0.0

            test_metrics["is_official_full_test"] = is_official_full
            test_metrics["evaluated_samples"] = evaluated_count
            test_metrics["total_official_split_samples"] = total_official_test
            test_metrics["coverage_percentage"] = coverage_pct
            test_metrics["missing_samples_count"] = max(0, total_official_test - evaluated_count)

            if is_official_full:
                test_metrics["evaluation_status"] = "OFFICIAL_FULL_TEST_BENCHMARK"
                title = f"Official {self.dataset_name} Test Split Benchmark Results"
            else:
                test_metrics["evaluation_status"] = "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL"
                disclaimer = (
                    f"UNOFFICIAL: Evaluated on an incomplete accessible subset "
                    f"({evaluated_count:,} / {total_official_test:,} images, {coverage_pct:.2f}% coverage). "
                    f"{test_metrics['missing_samples_count']:,} images missing from local archives. "
                    "These metrics MUST NOT be reported or published as the official full-test benchmark."
                )
                test_metrics["disclaimer"] = disclaimer
                test_metrics.setdefault("warnings", []).append(disclaimer)
                title = f"[UNOFFICIAL] {self.dataset_name} Partial Test Subset ({evaluated_count:,} / {total_official_test:,} images)"

            (self.output_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, indent=2, default=str), encoding="utf-8"
            )
            print(format_metrics_summary(test_metrics, title=title))

        # Save run metadata
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "total_training_time_seconds": total_training_time,
            "best_epoch": self.best_epoch,
            "best_metric_value": self.best_metric_value,
            "metric_optimized": self.metric_to_optimize,
            "config": self.config,
        }
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        # Save training configuration
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, default=str), encoding="utf-8"
        )

        return {
            "best_val_metrics": self.best_val_metrics,
            "test_metrics": test_metrics,
            "best_epoch": self.best_epoch,
            "total_training_time": total_training_time,
        }


# Backward-compatible alias for existing NIH code and test suites
NIHTrainer = ClientTrainer
