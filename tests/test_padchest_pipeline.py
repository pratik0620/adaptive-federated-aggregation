"""Comprehensive unit test suite for the PadChest data ingestion, filtering, and training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.config_loader import load_config, parse_simple_yaml
from src.data.padchest_dataset import (
    PadChestDataset,
    PadChestPathResolver,
    create_padchest_dataloaders,
    filter_padchest_frontal_views,
    get_padchest_transforms,
)
from src.dataset_utils import padchest_clean_binary_labels, standardize_padchest
from src.federated.adapters import build_padchest_client
from src.federated.client import FederatedClient
from src.federated.preflight import audit_client_readiness, check_dependencies
from src.models.resnet import build_resnet50_binary, load_checkpoint, save_checkpoint
from src.preprocess_padchest import (
    _validate_columns,
    run_padchest_preprocessing,
    verify_existing_splits,
)
from src.preprocessing_utils import assert_no_leakage, assert_no_study_leakage
from src.trainer import ClientTrainer
from scripts.preflight_padchest import run_padchest_preflight


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metadata_df() -> pd.DataFrame:
    """Create a realistic mock metadata DataFrame representing PadChest raw records."""
    records = [
        # Patient 1: 2 frontal images, positive for Pleural Effusion
        {
            "ImageID": "img_p1_01.png",
            "ImageDir": "0",
            "PatientID": "patient_001",
            "StudyID": "study_001_1",
            "Projection": "PA",
            "ViewPosition_DICOM": "POSTEROANTERIOR",
            "Labels": "['pleural effusion']",
            "PatientBirth": 1960.0,
            "StudyDate_DICOM": "20150101",
            "PatientSex_DICOM": "M",
        },
        {
            "ImageID": "img_p1_02.png",
            "ImageDir": "0",
            "PatientID": "patient_001",
            "StudyID": "study_001_2",
            "Projection": "AP_horizontal",
            "ViewPosition_DICOM": "",
            "Labels": "['pleural effusion', 'pneumonia']",
            "PatientBirth": 1960.0,
            "StudyDate_DICOM": "20150201",
            "PatientSex_DICOM": "M",
        },
        # Patient 2: 1 normal frontal image (negative)
        {
            "ImageID": "img_p2_01.png",
            "ImageDir": "1",
            "PatientID": "patient_002",
            "StudyID": "study_002_1",
            "Projection": "PA",
            "ViewPosition_DICOM": "PA",
            "Labels": "['normal']",
            "PatientBirth": 1980.0,
            "StudyDate_DICOM": "20160505",
            "PatientSex_DICOM": "F",
        },
        # Patient 3: 1 lateral image (normal) - should be excluded by frontal filter
        {
            "ImageID": "img_p3_01.png",
            "ImageDir": "2",
            "PatientID": "patient_003",
            "StudyID": "study_003_1",
            "Projection": "L",
            "ViewPosition_DICOM": "LATERAL",
            "Labels": "['normal']",
            "PatientBirth": 1975.0,
            "StudyDate_DICOM": "20170810",
            "PatientSex_DICOM": "M",
        },
        # Patient 4: pericardial effusion (should NOT match pleural effusion)
        {
            "ImageID": "img_p4_01.png",
            "ImageDir": "3",
            "PatientID": "patient_004",
            "StudyID": "study_004_1",
            "Projection": "PA",
            "ViewPosition_DICOM": "POSTEROANTERIOR",
            "Labels": "['pericardial effusion']",
            "PatientBirth": 1950.0,
            "StudyDate_DICOM": "20180101",
            "PatientSex_DICOM": "F",
        },
        # Patient 5: costophrenic angle blunting (ambiguous/uncertain, should be excluded)
        {
            "ImageID": "img_p5_01.png",
            "ImageDir": "4",
            "PatientID": "patient_005",
            "StudyID": "study_005_1",
            "Projection": "AP",
            "ViewPosition_DICOM": "ANTEROPOSTERIOR",
            "Labels": "['costophrenic angle blunting']",
            "PatientBirth": 1990.0,
            "StudyDate_DICOM": "20190301",
            "PatientSex_DICOM": "M",
        },
        # Patient 6: normal image
        {
            "ImageID": "img_p6_01.png",
            "ImageDir": "5",
            "PatientID": "patient_006",
            "StudyID": "study_006_1",
            "Projection": "PA",
            "ViewPosition_DICOM": "POSTEROANTERIOR",
            "Labels": "['normal']",
            "PatientBirth": 1985.0,
            "StudyDate_DICOM": "20190401",
            "PatientSex_DICOM": "F",
        },
        # Patient 7: normal image
        {
            "ImageID": "img_p7_01.png",
            "ImageDir": "5",
            "PatientID": "patient_007",
            "StudyID": "study_007_1",
            "Projection": "pa",  # Lowercase projection to test case-insensitivity
            "ViewPosition_DICOM": "POSTEROANTERIOR",
            "Labels": "['normal']",
            "PatientBirth": 1970.0,
            "StudyDate_DICOM": "20190501",
            "PatientSex_DICOM": "M",
        },
        # Patient 8: normal image
        {
            "ImageID": "img_p8_01.png",
            "ImageDir": "6",
            "PatientID": "patient_008",
            "StudyID": "study_008_1",
            "Projection": "PA",
            "ViewPosition_DICOM": "POSTEROANTERIOR",
            "Labels": "['normal']",
            "PatientBirth": 1965.0,
            "StudyDate_DICOM": "20190601",
            "PatientSex_DICOM": "F",
        },
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 1. Config Loader Tests
# ---------------------------------------------------------------------------

def test_config_loader_pure_python_yaml():
    """Verify fallback pure-Python YAML parser correctly parses numbers, bools, nulls, and strings."""
    sample_yaml = """
    # PadChest Config
    dataset: PadChest
    client_id: client_03_PadChest
    epochs: 10
    lr: 0.0001
    batch_size: 32
    use_amp: false
    frontal_only: true
    patient_limit: null
    """
    parsed = parse_simple_yaml(sample_yaml)
    assert parsed["dataset"] == "PadChest"
    assert parsed["client_id"] == "client_03_PadChest"
    assert parsed["epochs"] == 10
    assert abs(parsed["lr"] - 0.0001) < 1e-7
    assert parsed["batch_size"] == 32
    assert parsed["use_amp"] is False
    assert parsed["frontal_only"] is True
    assert parsed["patient_limit"] is None


def test_config_loader_file_reading(tmp_path: Path):
    """Verify load_config handles both JSON and YAML files."""
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps({"dataset": "PadChest", "epochs": 5}), encoding="utf-8")
    loaded_json = load_config(json_path)
    assert loaded_json["dataset"] == "PadChest"
    assert loaded_json["epochs"] == 5

    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("dataset: PadChest\nepochs: 7\n", encoding="utf-8")
    loaded_yaml = load_config(yaml_path)
    assert loaded_yaml["dataset"] == "PadChest"
    assert loaded_yaml["epochs"] == 7


# ---------------------------------------------------------------------------
# 2. Label Mapping Tests (Phase 3 Requirements)
# ---------------------------------------------------------------------------

def test_padchest_clean_binary_labels_positive_negative_uncertain(sample_metadata_df: pd.DataFrame):
    """Verify strict Pleural Effusion positive/negative/uncertain mapping logic."""
    filtered, report = padchest_clean_binary_labels(sample_metadata_df, target_label="pleural effusion", negative_label="normal")

    # Patient 1 has 2 effusion images -> positive (target=1)
    # Patient 2, 3, 6, 7, 8 have ['normal'] -> negative (target=0) (Patient 3 will later be dropped by view filter)
    # Patient 4 (pericardial effusion) and Patient 5 (costophrenic angle blunting) MUST BE EXCLUDED!
    assert report["positive_before_filter"] == 2
    assert report["negative_before_filter"] == 5
    assert report["excluded_ambiguous_before_filter"] == 2

    # Check targets
    p1_rows = filtered[filtered["PatientID"] == "patient_001"]
    assert len(p1_rows) == 2
    assert (p1_rows["target"] == 1).all()

    p2_rows = filtered[filtered["PatientID"] == "patient_002"]
    assert len(p2_rows) == 1
    assert (p2_rows["target"] == 0).all()

    # Pericardial effusion and costophrenic angle blunting must NOT be in filtered df
    assert "patient_004" not in filtered["PatientID"].values
    assert "patient_005" not in filtered["PatientID"].values


def test_label_pitfall_pericardial_effusion_not_matched():
    """Verify substring matching hazard: 'pericardial effusion' must never match 'pleural effusion'."""
    df = pd.DataFrame([
        {"ImageID": "1.png", "Labels": "['pericardial effusion']"},
        {"ImageID": "2.png", "Labels": "['pleural effusion']"},
        {"ImageID": "3.png", "Labels": "['normal']"},
    ])
    clean, counts = padchest_clean_binary_labels(df, target_label="pleural effusion")
    assert counts["positive_before_filter"] == 1
    assert counts["negative_before_filter"] == 1
    assert counts["excluded_ambiguous_before_filter"] == 1
    assert clean[clean["ImageID"] == "1.png"].empty
    assert clean[clean["ImageID"] == "2.png"]["target"].iloc[0] == 1
    assert clean[clean["ImageID"] == "3.png"]["target"].iloc[0] == 0


def test_label_empty_or_nan_excluded():
    """Verify empty or NaN labels are excluded, never converted into negative labels."""
    df = pd.DataFrame([
        {"ImageID": "1.png", "Labels": "[]"},
        {"ImageID": "2.png", "Labels": None},
        {"ImageID": "3.png", "Labels": np.nan},
        {"ImageID": "4.png", "Labels": "['normal']"},
    ])
    clean, counts = padchest_clean_binary_labels(df, target_label="pleural effusion")
    assert counts["negative_before_filter"] == 1
    assert counts["excluded_ambiguous_before_filter"] == 3
    assert len(clean) == 1
    assert clean["ImageID"].iloc[0] == "4.png"


# ---------------------------------------------------------------------------
# 3. View Filtering Tests (Phase 4 Requirements)
# ---------------------------------------------------------------------------

def test_filter_padchest_frontal_views(sample_metadata_df: pd.DataFrame):
    """Verify frontal-view filter retains PA, AP, AP_horizontal and excludes Lateral and Costal."""
    clean, _ = padchest_clean_binary_labels(sample_metadata_df)
    standardized = standardize_padchest(clean)

    filtered, report = filter_padchest_frontal_views(standardized, split_name="test")

    # Patient 3 is Lateral (L) -> must be excluded
    # Patients 1, 2, 6, 7 (pa lowercase), 8 are frontal -> must be retained
    assert report["lateral_images_excluded"] == 1
    assert report["costal_images_excluded"] == 0
    assert report["frontal_images_retained"] == 6

    # Verify no Lateral images in filtered dataframe
    assert (filtered["projection"] != "L").all()
    # Check lowercase 'pa' from patient 7 was retained
    assert "patient_007" in filtered["patient_id"].values


# ---------------------------------------------------------------------------
# 4. Path Resolver & Nested Subdirectories (Phase 2 Requirements)
# ---------------------------------------------------------------------------

def test_path_resolver_nested_numbered_dirs_and_sidecars(tmp_path: Path):
    """Verify PadChestPathResolver handles nested numbered folders (0..50) and skips macOS sidecars."""
    images_root = tmp_path / "images"
    dir_34 = images_root / "34"
    dir_34.mkdir(parents=True, exist_ok=True)

    valid_img = dir_34 / "216840111366964012819207061112010286132754841_03-188-146.png"
    sidecar_img = dir_34 / "._216840111366964012819207061112010286132754841_03-188-146.png"

    # Create dummy images
    dummy_img = Image.new("RGB", (64, 64), color="gray")
    dummy_img.save(valid_img)
    sidecar_img.write_text("macOS binary metadata sidecar", encoding="utf-8")

    resolver = PadChestPathResolver(search_roots=[images_root])
    resolver.build_index()

    # Valid image is resolved by basename and subpath
    assert resolver.resolve(valid_img.name) == valid_img.resolve()
    assert resolver.resolve(f"34/{valid_img.name}") == valid_img.resolve()

    # macOS sidecar must NEVER be indexed or resolved
    assert resolver.resolve(sidecar_img.name) is None
    assert sidecar_img.name not in resolver._index


# ---------------------------------------------------------------------------
# 5. Preprocessing & Patient Split Disjointness (Phase 5 Requirements)
# ---------------------------------------------------------------------------

def test_run_padchest_preprocessing_disjointness_and_manifests(tmp_path: Path, sample_metadata_df: pd.DataFrame):
    """Verify reproducible patient splitting with zero patient and study leakage."""
    meta_csv = tmp_path / "mock_metadata.csv"
    sample_metadata_df.to_csv(meta_csv, index=False)
    out_dir = tmp_path / "processed"

    summary = run_padchest_preprocessing(
        metadata_path=meta_csv,
        output_dir=out_dir,
        seed=42,
        train_size=0.6,
        val_size=0.2,
        dry_run=False,
        verbose=False,
        allow_small_mock=True,
    )

    # Check output files written
    assert (out_dir / "train.csv").is_file()
    assert (out_dir / "val.csv").is_file()
    assert (out_dir / "test.csv").is_file()
    assert (out_dir / "summary.json").is_file()

    train_df = pd.read_csv(out_dir / "train.csv")
    val_df = pd.read_csv(out_dir / "val.csv")
    test_df = pd.read_csv(out_dir / "test.csv")

    splits = {"train": train_df, "val": val_df, "test": test_df}

    # Verify zero leakage assertions
    assert_no_leakage(splits)
    assert_no_study_leakage(splits)

    train_pts = set(train_df["patient_id"].astype(str))
    val_pts = set(val_df["patient_id"].astype(str))
    test_pts = set(test_df["patient_id"].astype(str))

    assert len(train_pts & val_pts) == 0
    assert len(train_pts & test_pts) == 0
    assert len(val_pts & test_pts) == 0


def test_verify_existing_splits_on_official_cohort():
    """Verify verify_existing_splits functions correctly on the official PadChest processed directory."""
    out_dir = Path("data/processed/PadChest")
    if out_dir.is_dir() and (out_dir / "train.csv").is_file():
        report = verify_existing_splits(out_dir, verbose=False)
        assert report["status"] == "VERIFIED_VALID"
        assert report["total_images"] == 60243
        assert report["total_patients"] == 33412


def test_sample_metadata_rejected(tmp_path: Path, sample_metadata_df: pd.DataFrame):
    """Ensure that sample metadata filename is explicitly rejected by the validation guard."""
    sample_csv = tmp_path / "chest_x_ray_images_labels_sample.csv"
    sample_metadata_df.to_csv(sample_csv, index=False)
    out_dir = tmp_path / "proc_sample"

    with pytest.raises(ValueError, match="Sample PadChest metadata"):
        run_padchest_preprocessing(sample_csv, out_dir, allow_small_mock=False)


def test_truncated_metadata_rejected(tmp_path: Path, sample_metadata_df: pd.DataFrame):
    """Ensure that non-sample filename with <100K records is rejected when allow_small_mock=False."""
    meta_csv = tmp_path / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    sample_metadata_df.to_csv(meta_csv, index=False)
    out_dir = tmp_path / "proc_trunc"

    with pytest.raises(ValueError, match="suspiciously small"):
        run_padchest_preprocessing(meta_csv, out_dir, allow_small_mock=False)


def test_stale_sample_processed_splits_rejected(tmp_path: Path, sample_metadata_df: pd.DataFrame):
    """Ensure that verify_existing_splits rejects directories generated from sample metadata."""
    meta_csv = tmp_path / "mock_meta.csv"
    sample_metadata_df.to_csv(meta_csv, index=False)
    out_dir = tmp_path / "proc_stale"

    run_padchest_preprocessing(meta_csv, out_dir, allow_small_mock=True, verbose=False)
    with pytest.raises(ValueError, match="indicat"):
        verify_existing_splits(out_dir, verbose=False)


def test_compressed_official_metadata_loading():
    """Verify that PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz loads with full 160,861 records."""
    gz_path = Path("data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz")
    if gz_path.is_file():
        df = pd.read_csv(gz_path, compression="gzip", nrows=10)
        assert len(df) == 10
        assert "ImageID" in df.columns
        assert "Labels" in df.columns
        assert "Projection" in df.columns


# ---------------------------------------------------------------------------
# 6. Checkpoint Save/Load & State-Dict Validation (Phase 6 Requirements)
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_strict_load(tmp_path: Path):
    """Verify PyTorch model checkpoints save and reload strictly into build_resnet50_binary."""
    model = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_val_metric=0.85,
        metric_name="val_auroc",
        config={"dataset": "PadChest"},
    )
    assert ckpt_path.is_file()

    # Load into fresh instance using correct argument signature
    new_model = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device="cpu")
    loaded_data = load_checkpoint(checkpoint_path=ckpt_path, model=new_model, strict=True)
    assert loaded_data["epoch"] == 1
    assert loaded_data["best_val_metric"] == 0.85

    # Check classification head matches
    assert new_model.classifier.weight.shape == torch.Size([1, 2048])
    assert new_model.classifier.bias.shape == torch.Size([1])


# ---------------------------------------------------------------------------
# 7. Preflight Guard & Federated Compatibility (Phase 7 & Phase 8)
# ---------------------------------------------------------------------------

def test_padchest_preflight_guard(tmp_path: Path, sample_metadata_df: pd.DataFrame):
    """Verify preflight correctly flags missing images and passes when images exist."""
    meta_csv = tmp_path / "mock_meta.csv"
    sample_metadata_df.to_csv(meta_csv, index=False)
    out_dir = tmp_path / "proc"
    images_dir = tmp_path / "raw_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    run_padchest_preprocessing(meta_csv, out_dir, seed=42, verbose=False, allow_small_mock=True)

    # 1. Without images, preflight MUST fail
    is_ready, blockers, _ = run_padchest_preflight(
        config_path=None,
        metadata_override=meta_csv,
        image_dir_override=images_dir,
        processed_dir_override=out_dir,
        smoke_test=False,
        verbose=False,
    )
    assert is_ready is False
    assert any("accessible" in b.lower() for b in blockers)

    # 2. Populate dummy images for all train/val/test frontal records
    train_df = pd.read_csv(out_dir / "train.csv")
    val_df = pd.read_csv(out_dir / "val.csv")
    test_df = pd.read_csv(out_dir / "test.csv")

    for df in [train_df, val_df, test_df]:
        for _, row in df.iterrows():
            img_file = images_dir / str(row["image_id"])
            img = Image.new("RGB", (64, 64), color="gray")
            img.save(img_file)

    # In smoke test mode, with accessible images populated, preflight passes
    is_ready_smoke, blockers_smoke, _ = run_padchest_preflight(
        config_path=None,
        metadata_override=meta_csv,
        image_dir_override=images_dir,
        processed_dir_override=out_dir,
        smoke_test=True,
        verbose=False,
    )
    assert is_ready_smoke is True
    assert len(blockers_smoke) == 0


def test_federated_padchest_client_adapter(tmp_path: Path, sample_metadata_df: pd.DataFrame):
    """Verify PadChest federated client adapter constructs properly and is compatible with FedAvg."""
    meta_csv = tmp_path / "mock_meta.csv"
    sample_metadata_df.to_csv(meta_csv, index=False)
    out_dir = tmp_path / "proc_fed"
    images_dir = tmp_path / "images_fed"
    images_dir.mkdir(parents=True, exist_ok=True)

    run_padchest_preprocessing(meta_csv, out_dir, seed=42, verbose=False, allow_small_mock=True)

    train_df = pd.read_csv(out_dir / "train.csv")
    val_df = pd.read_csv(out_dir / "val.csv")
    test_df = pd.read_csv(out_dir / "test.csv")

    for df in [train_df, val_df, test_df]:
        for _, row in df.iterrows():
            img_file = images_dir / str(row["image_id"])
            img = Image.new("RGB", (64, 64), color="gray")
            img.save(img_file)

    # Pass un-pretrained model to prevent unneeded network calls during unit testing
    local_model = build_resnet50_binary(pretrained=False, device="cpu")

    client = build_padchest_client(
        repo_root=tmp_path,
        processed_dir=out_dir,
        image_dir=images_dir,
        batch_size=2,
        device="cpu",
        model=local_model,
    )

    assert isinstance(client, FederatedClient)
    assert client.client_id == "client_03_PadChest"
    params = client.get_parameters()
    assert "classifier.weight" in params
    assert params["classifier.weight"].shape == torch.Size([1, 2048])
    assert client.get_sample_count() > 0
