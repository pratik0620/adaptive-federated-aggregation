"""Dataset adapters for federated healthcare clients (NIH, CheXpert, PadChest).

Reuses existing dataset loaders, path resolvers, patient samplers, and frontal view
filters without duplicating data logic or violating patient isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torch.nn as nn

from ..data.chexpert_dataset import (
    CheXpertPathResolver,
    create_chexpert_dataloaders,
    filter_chexpert_frontal_views,
)
from ..data.nih_dataset import create_nih_dataloaders
from ..data.padchest_dataset import (
    PadChestPathResolver,
    create_padchest_dataloaders,
    filter_padchest_frontal_views,
)
from ..data.path_resolver import NIHPathResolver
from ..data.patient_sampler import select_training_patients
from ..models.resnet import build_resnet50_binary
from .client import FederatedClient


def build_nih_client(
    repo_root: str | Path | None = None,
    processed_dir: str | Path = "data/processed/NIH",
    image_dir: str | Path | None = None,
    patient_limit: int | None = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    device: str | torch.device = "cpu",
    model: nn.Module | None = None,
    config: dict[str, Any] | None = None,
) -> FederatedClient:
    """Build an NIH ChestX-ray14 FederatedClient instance.

    Args:
        repo_root: Root repository directory.
        processed_dir: Path to directory containing processed train.csv, val.csv, test.csv.
        image_dir: Path to raw image archives.
        patient_limit: Optional limit on unique patients for pilot experiments.
        batch_size: Mini-batch size.
        img_size: Resize square resolution.
        num_workers: DataLoader workers.
        seed: Random seed for patient sampling.
        device: Target execution device.
        model: Optional pre-constructed model; if None, instantiates ResNet50Binary.
        config: Client training hyperparameters.

    Returns:
        Configured FederatedClient instance for NIH.
    """
    root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    proc_path = (root / processed_dir).resolve() if not Path(processed_dir).is_absolute() else Path(processed_dir)

    train_df = pd.read_csv(proc_path / "train.csv")
    val_df = pd.read_csv(proc_path / "val.csv")
    test_df = pd.read_csv(proc_path / "test.csv")

    search_roots = []
    if image_dir:
        search_roots.append(Path(image_dir).resolve())
    search_roots.extend([
        root / "data" / "raw" / "NIH" / "images",
        root / "data" / "raw" / "NIH",
        root / "data" / "raw",
    ])

    resolver = NIHPathResolver(search_roots=search_roots)
    resolver.build_index()

    # Patient selection
    accessible_train_pts = resolver.get_accessible_patient_ids(train_df)
    selected_train_df, manifest = select_training_patients(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        patient_limit=patient_limit,
        seed=seed,
        accessible_patient_ids=accessible_train_pts if accessible_train_pts else None,
    )

    resolved_train, train_report = resolver.resolve_dataframe(selected_train_df, split_name="train", filter_missing=True)
    resolved_val, val_report = resolver.resolve_dataframe(val_df, split_name="val", filter_missing=True)
    resolved_test, test_report = resolver.resolve_dataframe(test_df, split_name="test", filter_missing=True)

    if len(resolved_train) == 0:
        raise RuntimeError("No accessible training images resolved for NIH client.")

    loaders = create_nih_dataloaders(
        train_df=resolved_train,
        val_df=resolved_val,
        test_df=resolved_test,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
    )

    total_official_val = 5836
    val_evaluated_count = len(resolved_val)
    is_official_full_val = val_evaluated_count >= total_official_val
    missing_samples = max(0, total_official_val - val_evaluated_count)
    coverage_pct = (val_evaluated_count / total_official_val * 100.0) if total_official_val > 0 else 0.0

    metadata = {
        "dataset_name": "NIH_ChestX-ray14",
        "client_id": "client_01_NIH",
        "unique_patient_count": manifest.get("selected_patient_count", resolved_train["patient_id"].nunique()),
        "is_official_full_val": is_official_full_val,
        "missing_samples_count": missing_samples,
        "coverage_percentage": coverage_pct,
        "evaluation_status": "OFFICIAL_FULL_VALIDATION_BENCHMARK" if is_official_full_val else "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL",
        "path_report": {"train": train_report, "val": val_report, "test": test_report},
    }

    if model is None:
        model = build_resnet50_binary(pretrained=True, dropout_rate=0.0, device=device)

    return FederatedClient(
        client_id="client_01_NIH",
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders.get("test"),
        device=device,
        config=config,
        metadata=metadata,
    )


def build_chexpert_client(
    repo_root: str | Path | None = None,
    processed_dir: str | Path = "data/processed/CheXpert",
    image_dir: str | Path | None = None,
    patient_limit: int | None = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    device: str | torch.device = "cpu",
    model: nn.Module | None = None,
    config: dict[str, Any] | None = None,
) -> FederatedClient:
    """Build a CheXpert FederatedClient instance (Client 02)."""
    root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    proc_path = (root / processed_dir).resolve() if not Path(processed_dir).is_absolute() else Path(processed_dir)

    train_df = pd.read_csv(proc_path / "train.csv")
    val_df = pd.read_csv(proc_path / "val.csv")
    test_df = pd.read_csv(proc_path / "test.csv")

    # Filter frontal views
    train_df, _ = filter_chexpert_frontal_views(train_df, split_name="train")
    val_df, _ = filter_chexpert_frontal_views(val_df, split_name="val")
    test_df, _ = filter_chexpert_frontal_views(test_df, split_name="test")

    search_roots = []
    if image_dir:
        search_roots.append(Path(image_dir).resolve())
    search_roots.extend([
        root / "data" / "raw" / "CheXpert" / "images",
        root / "data" / "raw" / "CheXpert" / "CheXpert-v1.0-small",
        root / "data" / "raw" / "CheXpert",
        root / "data" / "raw",
    ])

    resolver = CheXpertPathResolver(search_roots=search_roots)
    resolver.build_index()

    accessible_train_pts = resolver.get_accessible_patient_ids(train_df)
    selected_train_df, manifest = select_training_patients(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        patient_limit=patient_limit,
        seed=seed,
        accessible_patient_ids=accessible_train_pts if accessible_train_pts else None,
    )

    resolved_train, train_report = resolver.resolve_dataframe(selected_train_df, split_name="train", filter_missing=True)
    resolved_val, val_report = resolver.resolve_dataframe(val_df, split_name="val", filter_missing=True)
    resolved_test, test_report = resolver.resolve_dataframe(test_df, split_name="test", filter_missing=True)

    if len(resolved_train) == 0:
        raise RuntimeError("No accessible training images resolved for CheXpert client.")

    loaders = create_chexpert_dataloaders(
        train_df=resolved_train,
        val_df=resolved_val,
        test_df=resolved_test,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
    )

    total_official_val = 10024
    val_evaluated_count = len(resolved_val)
    is_official_full_val = val_evaluated_count >= total_official_val
    missing_samples = max(0, total_official_val - val_evaluated_count)
    coverage_pct = (val_evaluated_count / total_official_val * 100.0) if total_official_val > 0 else 0.0

    metadata = {
        "dataset_name": "CheXpert",
        "client_id": "client_02_CheXpert",
        "unique_patient_count": manifest.get("selected_patient_count", resolved_train["patient_id"].nunique()),
        "is_official_full_val": is_official_full_val,
        "missing_samples_count": missing_samples,
        "coverage_percentage": coverage_pct,
        "evaluation_status": "OFFICIAL_FULL_VALIDATION_BENCHMARK" if is_official_full_val else "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL",
        "path_report": {"train": train_report, "val": val_report, "test": test_report},
    }

    if model is None:
        model = build_resnet50_binary(pretrained=True, dropout_rate=0.0, device=device)

    return FederatedClient(
        client_id="client_02_CheXpert",
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders.get("test"),
        device=device,
        config=config,
        metadata=metadata,
    )


def build_padchest_client(
    repo_root: str | Path | None = None,
    processed_dir: str | Path = "data/processed/PadChest",
    image_dir: str | Path | None = None,
    patient_limit: int | None = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    device: str | torch.device = "cpu",
    model: nn.Module | None = None,
    config: dict[str, Any] | None = None,
) -> FederatedClient:
    """Build a PadChest FederatedClient instance (Client 03)."""
    root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    proc_path = (root / processed_dir).resolve() if not Path(processed_dir).is_absolute() else Path(processed_dir)

    train_df = pd.read_csv(proc_path / "train.csv")
    val_df = pd.read_csv(proc_path / "val.csv")
    test_df = pd.read_csv(proc_path / "test.csv")

    train_df, _ = filter_padchest_frontal_views(train_df, split_name="train")
    val_df, _ = filter_padchest_frontal_views(val_df, split_name="val")
    test_df, _ = filter_padchest_frontal_views(test_df, split_name="test")

    search_roots = []
    if image_dir:
        search_roots.append(Path(image_dir).resolve())
    search_roots.extend([
        root / "data" / "raw" / "PadChest" / "images",
        root / "data" / "raw" / "PadChest",
        root / "data" / "raw",
    ])

    resolver = PadChestPathResolver(search_roots=search_roots)
    resolver.build_index()

    accessible_train_pts = resolver.get_accessible_patient_ids(train_df)
    selected_train_df, manifest = select_training_patients(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        patient_limit=patient_limit,
        seed=seed,
        accessible_patient_ids=accessible_train_pts if accessible_train_pts else None,
    )

    resolved_train, train_report = resolver.resolve_dataframe(selected_train_df, split_name="train", filter_missing=True)
    resolved_val, val_report = resolver.resolve_dataframe(val_df, split_name="val", filter_missing=True)
    resolved_test, test_report = resolver.resolve_dataframe(test_df, split_name="test", filter_missing=True)

    if len(resolved_train) == 0:
        raise RuntimeError("No accessible training images resolved for PadChest client.")

    loaders = create_padchest_dataloaders(
        train_df=resolved_train,
        val_df=resolved_val,
        test_df=resolved_test,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
    )

    val_evaluated_count = len(resolved_val)
    total_official_val = 10000  # standard split size
    is_official_full_val = val_evaluated_count >= total_official_val
    missing_samples = max(0, total_official_val - val_evaluated_count)
    coverage_pct = (val_evaluated_count / total_official_val * 100.0) if total_official_val > 0 else 0.0

    metadata = {
        "dataset_name": "PadChest",
        "client_id": "client_03_PadChest",
        "unique_patient_count": manifest.get("selected_patient_count", resolved_train["patient_id"].nunique()),
        "is_official_full_val": is_official_full_val,
        "missing_samples_count": missing_samples,
        "coverage_percentage": coverage_pct,
        "evaluation_status": "OFFICIAL_FULL_VALIDATION_BENCHMARK" if is_official_full_val else "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL",
        "path_report": {"train": train_report, "val": val_report, "test": test_report},
    }

    if model is None:
        model = build_resnet50_binary(pretrained=True, dropout_rate=0.0, device=device)

    return FederatedClient(
        client_id="client_03_PadChest",
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders.get("test"),
        device=device,
        config=config,
        metadata=metadata,
    )
