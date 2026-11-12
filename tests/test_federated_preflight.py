"""Unit tests for federated experiment preflight verification and readiness auditing."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.federated.config import FederatedConfig
from src.federated.preflight import (
    check_dependencies,
    audit_client_readiness,
    run_federated_preflight,
    PreflightSummary,
)


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


def test_audit_nih_client_readiness():
    """Verify NIH client readiness auditing correctly reflects available images on disk."""
    repo_root = Path(__file__).resolve().parent.parent
    status = audit_client_readiness("client_01_NIH", repo_root, patient_limit=50)
    assert status.client_id == "client_01_NIH"
    assert status.is_ready is True
    assert status.usable_train_images > 0
    assert status.usable_train_patients > 0
    assert status.manifest_train_rows > 0


def test_audit_chexpert_and_padchest_blocked_on_missing_images():
    """Verify CheXpert and PadChest are correctly marked BLOCKED due to unpopulated image dirs."""
    repo_root = Path(__file__).resolve().parent.parent
    
    chexpert_status = audit_client_readiness("client_02_CheXpert", repo_root, patient_limit=50)
    assert chexpert_status.client_id == "client_02_CheXpert"
    assert chexpert_status.is_ready is False
    assert chexpert_status.usable_train_images == 0
    assert chexpert_status.blocker_reason is not None

    padchest_status = audit_client_readiness("client_03_PadChest", repo_root, patient_limit=50)
    assert padchest_status.client_id == "client_03_PadChest"
    assert padchest_status.is_ready is False
    assert padchest_status.usable_train_images == 0
    assert padchest_status.blocker_reason is not None


def test_real_data_preflight_prevents_silent_client_degradation():
    """Verify preflight halts and reports blockers when 3-client run lacks 2 clients."""
    config = FederatedConfig(
        participating_clients=["client_01_NIH", "client_02_CheXpert", "client_03_PadChest"],
        patient_limit=50,
    )
    summary = run_federated_preflight(config, is_synthetic=False, abort_on_failure=False)
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
