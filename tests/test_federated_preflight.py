"""Unit tests for federated experiment preflight verification and readiness auditing."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from PIL import Image
import pytest

from src.federated.config import FederatedConfig
from src.federated.preflight import (
    check_dependencies,
    audit_client_readiness,
    run_federated_preflight,
    PreflightSummary,
)


def _create_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    img.save(path)


def _setup_mock_nih(root: Path, populate_images: bool = True) -> None:
    proc_dir = root / "data" / "processed" / "NIH"
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame([
        {"image_id": "00000001_000.png", "patient_id": "00000001", "target": 1},
        {"image_id": "00000002_000.png", "patient_id": "00000002", "target": 0},
    ])
    val_df = pd.DataFrame([
        {"image_id": "00000003_000.png", "patient_id": "00000003", "target": 1},
    ])
    test_df = pd.DataFrame([
        {"image_id": "00000004_000.png", "patient_id": "00000004", "target": 0},
    ])

    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    if populate_images:
        img_dir = root / "data" / "raw" / "NIH" / "images"
        for img_id in ["00000001_000.png", "00000002_000.png", "00000003_000.png", "00000004_000.png"]:
            _create_dummy_image(img_dir / img_id)


def _setup_mock_chexpert(root: Path, populate_images: bool = False) -> None:
    proc_dir = root / "data" / "processed" / "CheXpert"
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame([
        {
            "image_id": "patient00001/study1/view1_frontal.jpg",
            "patient_id": "patient00001",
            "target": 1,
            "view_position": "Frontal",
            "projection": "PA",
            "image_path": "patient00001/study1/view1_frontal.jpg",
        }
    ])
    val_df = pd.DataFrame([
        {
            "image_id": "patient00002/study1/view1_frontal.jpg",
            "patient_id": "patient00002",
            "target": 0,
            "view_position": "Frontal",
            "projection": "PA",
            "image_path": "patient00002/study1/view1_frontal.jpg",
        }
    ])
    test_df = pd.DataFrame([
        {
            "image_id": "patient00003/study1/view1_frontal.jpg",
            "patient_id": "patient00003",
            "target": 0,
            "view_position": "Frontal",
            "projection": "PA",
            "image_path": "patient00003/study1/view1_frontal.jpg",
        }
    ])

    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    if populate_images:
        img_dir = root / "data" / "raw" / "CheXpert" / "images"
        for row in pd.concat([train_df, val_df, test_df]).itertuples():
            _create_dummy_image(img_dir / row.image_path)


def _setup_mock_padchest(root: Path, populate_images: bool = False) -> None:
    proc_dir = root / "data" / "processed" / "PadChest"
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame([
        {
            "image_id": "216840111366964012819207061112010577107_00-001-001.png",
            "patient_id": "padchest_pt_1",
            "target": 1,
            "projection": "PA",
            "view_position": "PA",
        }
    ])
    val_df = pd.DataFrame([
        {
            "image_id": "216840111366964012819207061112010577107_00-001-002.png",
            "patient_id": "padchest_pt_2",
            "target": 0,
            "projection": "PA",
            "view_position": "PA",
        }
    ])
    test_df = pd.DataFrame([
        {
            "image_id": "216840111366964012819207061112010577107_00-001-003.png",
            "patient_id": "padchest_pt_3",
            "target": 0,
            "projection": "PA",
            "view_position": "PA",
        }
    ])

    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    if populate_images:
        img_dir = root / "data" / "raw" / "PadChest" / "images"
        for row in pd.concat([train_df, val_df, test_df]).itertuples():
            _create_dummy_image(img_dir / row.image_id)


def test_dependency_check_passes():
    """Verify check_dependencies validates torch, torchvision, pillow, and resnet50 builder."""
    all_ok, details = check_dependencies()
    assert details["torch"] is True
    assert details["torchvision"] is True
    assert details["pillow"] is True
    assert details["resnet50_model"] is True
    assert all_ok is True


def test_synthetic_preflight_always_passes():
    """Verify synthetic mode bypasses raw data checks and passes cleanly."""
    config = FederatedConfig(
        participating_clients=["client_01_NIH", "client_02_CheXpert", "client_03_PadChest"],
    )
    summary = run_federated_preflight(config, is_synthetic=True, abort_on_failure=False)
    assert isinstance(summary, PreflightSummary)
    assert summary.is_ready is True
    assert summary.is_synthetic is True
    assert len(summary.blockers) == 0


def test_audit_nih_client_readiness(tmp_path: Path):
    """Verify NIH client readiness auditing correctly reflects available images on disk."""
    _setup_mock_nih(tmp_path, populate_images=True)
    status = audit_client_readiness("client_01_NIH", tmp_path, patient_limit=50)
    assert status.client_id == "client_01_NIH"
    assert status.is_ready is True
    assert status.usable_train_images > 0
    assert status.usable_train_patients > 0
    assert status.manifest_train_rows > 0

    # Negative scenario: when images are missing, client is marked BLOCKED
    empty_root = tmp_path / "empty_nih"
    _setup_mock_nih(empty_root, populate_images=False)
    blocked_status = audit_client_readiness("client_01_NIH", empty_root, patient_limit=50)
    assert blocked_status.is_ready is False
    assert blocked_status.usable_train_images == 0
    assert blocked_status.blocker_reason is not None


def test_audit_chexpert_and_padchest_blocked_on_missing_images(tmp_path: Path):
    """Verify CheXpert and PadChest are correctly marked BLOCKED due to unpopulated image dirs."""
    _setup_mock_chexpert(tmp_path, populate_images=False)
    _setup_mock_padchest(tmp_path, populate_images=False)

    chexpert_status = audit_client_readiness("client_02_CheXpert", tmp_path, patient_limit=50)
    assert chexpert_status.client_id == "client_02_CheXpert"
    assert chexpert_status.is_ready is False
    assert chexpert_status.usable_train_images == 0
    assert chexpert_status.blocker_reason is not None

    padchest_status = audit_client_readiness("client_03_PadChest", tmp_path, patient_limit=50)
    assert padchest_status.client_id == "client_03_PadChest"
    assert padchest_status.is_ready is False
    assert padchest_status.usable_train_images == 0
    assert padchest_status.blocker_reason is not None

    # Positive scenario: when images ARE populated, both become READY
    ready_root = tmp_path / "ready_clients"
    _setup_mock_chexpert(ready_root, populate_images=True)
    _setup_mock_padchest(ready_root, populate_images=True)

    ready_chex = audit_client_readiness("client_02_CheXpert", ready_root, patient_limit=50)
    assert ready_chex.is_ready is True
    assert ready_chex.usable_train_images > 0

    ready_pad = audit_client_readiness("client_03_PadChest", ready_root, patient_limit=50)
    assert ready_pad.is_ready is True
    assert ready_pad.usable_train_images > 0


def test_real_data_preflight_prevents_silent_client_degradation(tmp_path: Path):
    """Verify preflight halts and reports blockers when 3-client run lacks 2 clients."""
    # NIH is ready, but CheXpert and PadChest are unpopulated (blocked)
    _setup_mock_nih(tmp_path, populate_images=True)
    _setup_mock_chexpert(tmp_path, populate_images=False)
    _setup_mock_padchest(tmp_path, populate_images=False)

    config = FederatedConfig(
        participating_clients=["client_01_NIH", "client_02_CheXpert", "client_03_PadChest"],
        patient_limit=50,
    )
    summary = run_federated_preflight(config, repo_root=tmp_path, is_synthetic=False, abort_on_failure=False)
    assert summary.is_ready is False
    assert len(summary.blockers) >= 2
    # Ensure both CheXpert and PadChest blockers are recorded
    blocker_text = " ".join(summary.blockers)
    assert "client_02_CheXpert" in blocker_text
    assert "client_03_PadChest" in blocker_text

    # Formatted report should contain clear status
    report_text = summary.format_report()
    assert "PREFLIGHT RESULT: FAILED" in report_text
    assert "READY" in report_text  # NIH
    assert "BLOCKED" in report_text  # CheXpert / PadChest
