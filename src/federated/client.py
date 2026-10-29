"""Federated client abstractions and dataset adapters.

Enforces:
  - Local model parameter synchronization (set_parameters / get_parameters)
  - Optimizer state reset at the beginning of each communication round
  - Sample counting tracking exact samples used per round
  - Local validation evaluation without leaking raw images or patient records
  - Clean separation between local optimizer state and global model state
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..metrics import compute_binary_metrics


class FederatedClient:
    """Abstract base / general federated client for binary classification."""

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        device: str | torch.device = "cpu",
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize federated client node.

        Args:
            client_id: Unique client identifier (e.g. 'NIH', 'CheXpert', 'PadChest').
            model: PyTorch model instance matching the global architecture.
            train_loader: DataLoader for local private training partition.
            val_loader: DataLoader for local private validation partition.
            test_loader: Optional DataLoader for held-out test split (strictly for final benchmark).
            device: PyTorch device ('cpu' or 'cuda').
            config: Client hyperparameters (lr, optimizer, pos_weight, etc.).
            metadata: Dataset metadata (unique patient count, archive coverage, missing counts).
        """
        self.client_id = client_id
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        self.metadata = metadata or {}

        # Default hyperparameters
        self.lr = float(self.config.get("lr", 1e-4))
        self.weight_decay = float(self.config.get("weight_decay", 1e-4))
        self.optimizer_name = str(self.config.get("optimizer", "adamw")).lower()
        self.momentum = float(self.config.get("momentum", 0.9))

        # Loss function
        pos_weight = self.config.get("pos_weight")
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([float(pos_weight)], device=self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def set_parameters(
        self,
        global_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Load global model parameters into the local model.

        Args:
            global_state_dict: Parameter dictionary from the central server.
            strict: Whether to enforce exact key and shape matching.
        """
        device_state = {k: v.to(self.device) for k, v in global_state_dict.items()}
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(device_state, strict=strict)

    def get_parameters(self) -> dict[str, torch.Tensor]:
        """Return a copy of local model parameters on CPU."""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        return {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}

    def get_sample_count(self) -> int:
        """Return the number of training samples in the private local dataset."""
        if hasattr(self.train_loader, "dataset") and self.train_loader.dataset is not None:
            return len(self.train_loader.dataset)
        return 0

    def _build_fresh_optimizer(
        self,
        optimizer_name: str | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
    ) -> torch.optim.Optimizer:
        """Construct a fresh optimizer instance, resetting all momentum and state buffers.

        Critical Policy:
          Local optimizer state is reset at the start of every communication round.
          Adaptive optimizers (e.g. AdamW) retain momentum buffers that would otherwise
          drift against newly injected global model parameters.
        """
        opt_name = (optimizer_name or self.optimizer_name).lower()
        learn_rate = self.lr if lr is None else float(lr)
        wd = self.weight_decay if weight_decay is None else float(weight_decay)

        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=learn_rate, momentum=self.momentum, weight_decay=wd
            )
        else:  # default adamw
            return torch.optim.AdamW(self.model.parameters(), lr=learn_rate, weight_decay=wd)

    def fit_round(
        self,
        local_epochs: int = 1,
        lr: float | None = None,
        optimizer_name: str | None = None,
        weight_decay: float | None = None,
    ) -> dict[str, Any]:
        """Perform local training for one federated communication round.

        Args:
            local_epochs: Number of epochs to train locally on private data.
            lr: Optional learning rate override for this round.
            optimizer_name: Optional optimizer override ('adamw', 'sgd', 'adam').
            weight_decay: Optional weight decay override.

        Returns:
            Dictionary containing:
              - 'parameters': Updated local model state_dict (CPU tensors).
              - 'sample_count': Exact number of private training samples used.
              - 'train_metrics': Loss and training metrics summary.
              - 'client_id': Client node identifier.
        """
        self.model.train()
        start_time = time.time()

        # Enforce optimizer reset policy
        optimizer = self._build_fresh_optimizer(
            optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay
        )

        total_samples_used = 0
        total_loss = 0.0
        all_targets: list[int] = []
        all_probs: list[float] = []

        for epoch in range(1, local_epochs + 1):
            epoch_samples = 0
            for batch in self.train_loader:
                if isinstance(batch, (tuple, list)):
                    images, targets = batch[0], batch[1]
                else:
                    raise TypeError(f"DataLoader emitted unexpected batch type: {type(batch)}")

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(images).squeeze(-1)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

                batch_len = len(targets)
                epoch_samples += batch_len
                total_loss += loss.item() * batch_len

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.extend(probs.tolist())
                all_targets.extend(targets.detach().cpu().numpy().astype(int).tolist())

            total_samples_used = epoch_samples  # Number of unique samples visited per epoch

        duration = time.time() - start_time
        avg_loss = total_loss / max(1, len(all_targets))
        train_metrics = compute_binary_metrics(all_targets, all_probs)
        train_metrics["loss"] = avg_loss
        train_metrics["duration_seconds"] = duration
        train_metrics["local_epochs"] = local_epochs

        # Fallback for sample_count if loader was empty
        effective_sample_count = max(total_samples_used, self.get_sample_count())

        return {
            "client_id": self.client_id,
            "parameters": self.get_parameters(),
            "sample_count": effective_sample_count,
            "train_metrics": train_metrics,
        }

    @torch.no_grad()
    def evaluate(
        self,
        model_state_dict: dict[str, torch.Tensor] | None = None,
        split: str = "val",
    ) -> dict[str, Any]:
        """Evaluate the model on the private local validation or test split.

        Args:
            model_state_dict: Optional parameter dictionary to evaluate.
                              If provided, loaded temporarily into the model.
            split: 'val' or 'test'.

        Returns:
            Comprehensive evaluation metrics dictionary.
        """
        if split == "test":
            loader = self.test_loader
            if loader is None:
                return {
                    "client_id": self.client_id,
                    "split": "test",
                    "total_samples": 0,
                    "warnings": ["Test loader not provided for this client."],
                }
        else:
            loader = self.val_loader

        if loader is None or len(loader.dataset) == 0:
            return {
                "client_id": self.client_id,
                "split": split,
                "total_samples": 0,
                "auroc": None,
                "pr_auc": None,
                "sensitivity": None,
                "specificity": None,
                "f1": None,
                "balanced_accuracy": None,
                "warnings": [f"{split} split is empty or unavailable."],
            }

        if model_state_dict is not None:
            self.set_parameters(model_state_dict)

        self.model.eval()
        total_loss = 0.0
        all_targets: list[int] = []
        all_probs: list[float] = []

        for batch in loader:
            if isinstance(batch, (tuple, list)):
                images, targets = batch[0], batch[1]
            else:
                raise TypeError(f"Unexpected batch structure: {type(batch)}")

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images).squeeze(-1)
            loss = self.loss_fn(logits, targets)

            batch_len = len(targets)
            total_loss += loss.item() * batch_len

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.cpu().numpy().astype(int).tolist())

        avg_loss = total_loss / max(1, len(all_targets))
        metrics = compute_binary_metrics(all_targets, all_probs)
        metrics["client_id"] = self.client_id
        metrics["split"] = split
        metrics["loss"] = avg_loss

        # Merge metadata (archive coverage, missing images, unique patients)
        for meta_k in (
            "unique_patient_count",
            "is_official_full_val",
            "missing_samples_count",
            "coverage_percentage",
            "evaluation_status",
        ):
            if meta_k in self.metadata:
                metrics[meta_k] = self.metadata[meta_k]

        return metrics


class SyntheticFederatedClient(FederatedClient):
    """Synthetic federated client generating in-memory binary data for smoke and unit testing."""

    def __init__(
        self,
        client_id: str,
        num_train_samples: int = 50,
        num_val_samples: int = 20,
        input_dim: int = 16,
        model: nn.Module | None = None,
        seed: int = 42,
        device: str | torch.device = "cpu",
        config: dict[str, Any] | None = None,
        learnable: bool = False,
        true_w: torch.Tensor | None = None,
    ) -> None:
        """Create a synthetic client with deterministic in-memory tensors.

        Args:
            client_id: Identifier for the client.
            num_train_samples: Number of synthetic training examples.
            num_val_samples: Number of synthetic validation examples.
            input_dim: Feature dimension.
            model: PyTorch model. If None, instantiates a TinyModel with integer buffer.
            seed: RNG seed for reproducible feature/label generation.
            device: Computation device.
            config: Optional configuration dictionary.
            learnable: If True, generates labels via a ground-truth linear decision boundary
                       y = I(x @ true_w > 0), creating a learnable classification problem.
                       If False (default), generates independent random Bernoulli labels for smoke tests.
            true_w: Optional ground-truth weight vector of shape (input_dim, 1) when learnable=True.
        """
        rng = torch.Generator().manual_seed(seed)

        # Generate synthetic features
        train_x = torch.randn(num_train_samples, input_dim, generator=rng)
        val_x = torch.randn(num_val_samples, input_dim, generator=rng)

        if learnable:
            if true_w is None:
                w_rng = torch.Generator().manual_seed(999)
                true_w = torch.randn(input_dim, 1, generator=w_rng)
            train_y = ((train_x @ true_w).squeeze(-1) > 0.0).float()
            val_y = ((val_x @ true_w).squeeze(-1) > 0.0).float()
        else:
            train_y = torch.randint(0, 2, (num_train_samples,), generator=rng).float()
            val_y = torch.randint(0, 2, (num_val_samples,), generator=rng).float()

        # Ensure both classes are present in training and validation sets
        if torch.sum(train_y == 1) == 0:
            train_y[0] = 1.0
        elif torch.sum(train_y == 0) == 0:
            train_y[0] = 0.0

        if torch.sum(val_y == 1) == 0:
            val_y[0] = 1.0
        elif torch.sum(val_y == 0) == 0:
            val_y[0] = 0.0

        train_ds = TensorDataset(train_x, train_y)
        val_ds = TensorDataset(val_x, val_y)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        if model is None:
            # Default tiny linear classifier with a simulated integer buffer
            class TinyModel(nn.Module):
                def __init__(self, dim: int):
                    super().__init__()
                    self.linear = nn.Linear(dim, 1)
                    self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if self.training:
                        self.num_batches_tracked += 1
                    return self.linear(x)

            model = TinyModel(input_dim)

        metadata = {
            "unique_patient_count": num_train_samples,
            "is_synthetic": True,
            "dataset_name": f"Synthetic_{client_id}",
        }

        super().__init__(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            metadata=metadata,
        )
