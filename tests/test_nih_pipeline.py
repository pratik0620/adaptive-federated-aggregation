"""Unit tests and pre-training readiness audit for NIH ChestX-ray14 pipeline.

Tests:
  - Metric computation and single-class edge cases
  - Patient sampler reproducible seeded stratified sampling, natural prevalence, and leakage checks
  - Path resolver indexing, cross-platform paths, and availability audits
  - Real image decoding and transform pipeline using local archive images
  - PyTorch loss function, optimizer, and checkpoint serialization compatibility
  - Explicit dependency audit for torchvision and pillow
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.nih_dataset import NIHChestXRayDataset, get_nih_transforms, load_image_rgb
from src.data.path_resolver import NIHPathResolver, normalize_cross_platform_path
from src.data.patient_sampler import select_training_patients
from src.metrics import compute_binary_metrics, format_metrics_summary
from src.models.resnet import build_resnet50_binary, load_checkpoint, save_checkpoint


def test_metrics_standard():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.85])

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["total_samples"] == 8
    assert metrics["positive_count"] == 4
    assert metrics["negative_count"] == 4
    assert metrics["positive_fraction"] == 0.5
    assert metrics["auroc"] is not None
    assert metrics["auroc"] > 0.9
    assert metrics["pr_auc"] is not None
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 4}
    assert len(metrics["warnings"]) == 0

    summary_text = format_metrics_summary(metrics)
    assert "AUROC (Primary):" in summary_text


def test_metrics_single_class_edge_case():
    """Verify that single-class splits return undefined/None without crashing."""
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_binary_metrics(y_true, y_prob)

    assert metrics["total_samples"] == 4
    assert metrics["positive_count"] == 0
    assert metrics["negative_count"] == 4
    assert metrics["auroc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["sensitivity"] is None
    assert metrics["specificity"] == 1.0
    assert len(metrics["warnings"]) > 0


def test_patient_sampler_seeded_stratification_and_leakage():
    train_data = {
        "image_id": [f"img_{i}.png" for i in range(10)],
        "patient_id": ["P01", "P01", "P02", "P03", "P03", "P04", "P05", "P06", "P07", "P08"],
        "target": [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    }
    val_data = {
        "image_id": ["val_1.png", "val_2.png"],
        "patient_id": ["P09", "P10"],
        "target": [0, 1],
    }
    test_data = {
        "image_id": ["test_1.png"],
        "patient_id": ["P11"],
        "target": [0],
    }

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Test limit 4 patients
    subset_df, manifest = select_training_patients(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        patient_limit=4,
        seed=42,
    )

    assert manifest["selected_patient_count"] == 4
    assert manifest["selected_positive_patients"] in (1, 2)
    assert manifest["selected_negative_patients"] in (2, 3)
    assert manifest["natural_cohort_positive_patients"] == 3
    assert manifest["natural_cohort_negative_patients"] == 5

    # Determinism test: identical seed returns identical patient order
    _, manifest2 = select_training_patients(train_df=train_df, patient_limit=4, seed=42)
    assert manifest["selected_patient_ids"] == manifest2["selected_patient_ids"]

    # Test leakage assertion detects overlap
    leaking_val_df = pd.DataFrame({"patient_id": ["P01"], "target": [1]})
    with pytest.raises(AssertionError, match="Patient leakage detected"):
        select_training_patients(train_df=train_df, val_df=leaking_val_df, patient_limit=5)


def test_path_resolver_and_audit():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        nested_dir = tmp_path / "images_001" / "images"
        nested_dir.mkdir(parents=True)
        img1 = nested_dir / "00000001_000.png"
        img1.write_text("dummy")

        resolver = NIHPathResolver(search_roots=[tmp_path])
        assert resolver.build_index() == 1

        resolved = resolver.resolve("00000001_000.png")
        assert resolved is not None
        assert resolved.name == "00000001_000.png"

        assert resolver.resolve("missing_000.png") is None

        # Test availability audit
        test_df = pd.DataFrame({
            "image_id": ["00000001_000.png", "00000002_000.png", "00000003_000.png"],
            "patient_id": ["P1", "P2", "P3"],
            "target": [1, 0, 1],
        })
        audit = resolver.audit_split_availability(test_df, split_name="test")
        assert audit["total_official_images"] == 3
        assert audit["accessible_images"] == 1
        assert audit["missing_images"] == 2
        assert audit["is_complete"] is False
        assert audit["coverage_status"] == "PARTIAL_ARCHIVE_INCOMPLETE"
        assert len(audit["sample_missing_image_ids"]) == 2


def test_lightweight_image_decoding_on_available_images():
    """Verify lightweight image decoding and transform pipeline using existing local images."""
    image_dir = Path("data/raw/NIH/images/images_001")
    if not image_dir.is_dir():
        pytest.skip("data/raw/NIH/images/images_001 is not available locally.")

    sample_files = list(image_dir.glob("*.png"))[:3]
    assert len(sample_files) > 0, "No image files found in images_001"

    for sample_path in sample_files:
        # 1. Test image decoding
        rgb = load_image_rgb(sample_path)
        assert isinstance(rgb, np.ndarray)
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"Expected (H, W, 3), got {rgb.shape}"
        assert rgb.shape[0] > 100 and rgb.shape[1] > 100

        # 2. Test transform pipeline
        transform_train = get_nih_transforms("train", img_size=224)
        tensor_train = transform_train(rgb)
        assert isinstance(tensor_train, torch.Tensor)
        assert tensor_train.shape == torch.Size([3, 224, 224])
        assert tensor_train.dtype == torch.float32

        transform_val = get_nih_transforms("val", img_size=224)
        tensor_val = transform_val(rgb)
        assert tensor_val.shape == torch.Size([3, 224, 224])

    # 3. Test Dataset and DataLoader batching
    df_sample = pd.DataFrame({
        "image_id": [p.name for p in sample_files],
        "resolved_path": [str(p) for p in sample_files],
        "target": [0, 1, 0][: len(sample_files)],
    })
    dataset = NIHChestXRayDataset(df_sample, transform=get_nih_transforms("val", img_size=224))
    assert len(dataset) == len(sample_files)

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_images, batch_targets = next(iter(loader))
    assert batch_images.shape == torch.Size([2, 3, 224, 224])
    assert batch_targets.shape == torch.Size([2])


def test_pytorch_loss_optimizer_and_checkpoint_compatibility():
    """Verify loss function, optimizer, and checkpoint compatibility with installed PyTorch version."""
    # Dummy binary model to test PyTorch compatibility without torchvision
    class DummyBinaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyBinaryModel()
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # 1. Loss function compatibility: BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()
    logits = model(dummy_input).squeeze(-1)
    loss = loss_fn(logits, dummy_target)
    assert not torch.isnan(loss)
    loss.backward()

    # 2. Optimizer compatibility: AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.step()
    optimizer.zero_grad()

    # 3. Checkpoint saving and loading compatibility
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            best_val_metric=0.85,
            config={"lr": 1e-4},
        )
        assert ckpt_path.is_file()

        loaded = load_checkpoint(ckpt_path, model=model, optimizer=optimizer)
        assert loaded["epoch"] == 1
        assert loaded["best_val_metric"] == 0.85
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded


def test_dependency_audit_torchvision():
    """Audit installed torchvision and report dependency status explicitly instead of skipping silently."""
    try:
        import torchvision
        has_torchvision = True
    except ImportError:
        has_torchvision = False

    try:
        import PIL
        has_pil = True
    except ImportError:
        has_pil = False

    if not has_torchvision:
        # Explicitly verify that build_resnet50_binary raises descriptive ImportError
        with pytest.raises(ImportError, match="torchvision is required"):
            build_resnet50_binary()
        print("\n[DEPENDENCY AUDIT] TorchVision: NOT INSTALLED. (Run: pip install torchvision)")
    else:
        model = build_resnet50_binary(pretrained=False, dropout_rate=0.2, device="cpu")
        assert model.num_classes == 1
        print("\n[DEPENDENCY AUDIT] TorchVision: INSTALLED.")

    if not has_pil:
        print("[DEPENDENCY AUDIT] Pillow (PIL): NOT INSTALLED. (Using OpenCV fallback)")
    else:
        print("[DEPENDENCY AUDIT] Pillow (PIL): INSTALLED.")
