"""ResNet-50 binary classification model for thoracic disease prediction.

Standardized for binary classification (e.g. Pleural Effusion) across
NIH ChestX-ray14, CheXpert, and PadChest clients, with PyTorch state_dict
compatibility for future federated learning aggregation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ResNet50Binary(nn.Module):
    """ImageNet-pretrained ResNet-50 with a single-logit binary classification head.

    Output convention:
        - Output is raw logit tensor of shape (batch_size, 1).
        - Probabilities: torch.sigmoid(logits).
        - Binary loss: nn.BCEWithLogitsLoss().
        - Binary prediction: (torch.sigmoid(logits) >= threshold).long().
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.0) -> None:
        super().__init__()
        try:
            import torchvision.models as tv_models
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                backbone = tv_models.resnet50(weights=weights)
            except (ImportError, AttributeError):
                # Older torchvision fallback
                backbone = tv_models.resnet50(pretrained=pretrained)
        except ImportError as exc:
            raise ImportError(
                "torchvision is required to instantiate ResNet50Binary. "
                "Please install torchvision via: pip install torchvision"
            ) from exc

        in_features = backbone.fc.in_features  # 2048 for ResNet-50
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if dropout_rate > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 1),
            )
        else:
            self.classifier = nn.Linear(in_features, 1)

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.num_classes = 1
        self.head_type = "single_logit_binary"
        self.backbone_name = "resnet50"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (batch_size, 3, height, width).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted probabilities P(target=1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def build_resnet50_binary(
    pretrained: bool = True,
    dropout_rate: float = 0.0,
    device: str | torch.device = "cpu",
) -> ResNet50Binary:
    """Build and return an instance of ResNet50Binary moved to specified device."""
    model = ResNet50Binary(pretrained=pretrained, dropout_rate=dropout_rate)
    return model.to(device)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    best_val_metric: float | None = None,
    metric_name: str = "val_auroc",
    config: dict[str, Any] | None = None,
) -> Path:
    """Save model checkpoint in standard PyTorch state_dict format.

    Designed for clean export and future federated learning parameter aggregation.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DataParallel or DistributedDataParallel if present
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_architecture": "resnet50",
        "num_classes": 1,
        "classification_head": "single_logit_binary",
        "pretrained": getattr(raw_model, "pretrained", True),
        "dropout_rate": getattr(raw_model, "dropout_rate", 0.0),
        "model_state_dict": raw_model.state_dict(),
        "epoch": epoch,
        "best_val_metric": best_val_metric,
        "metric_name": metric_name,
        "config": config or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights and training state from checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
