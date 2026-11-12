"""Verify PyTorch checkpoint compatibility with shared ResNet-50 binary architecture for federated aggregation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.resnet import build_resnet50_binary, load_checkpoint


def verify_checkpoint(checkpoint_path: Path, output_json: Path | None = None) -> dict[str, Any]:
    print("=" * 70)
    print("  PadChest Client 03 — Federated Checkpoint Compatibility Audit")
    print("=" * 70)
    print(f"Checkpoint File: {checkpoint_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load raw checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 2. Reference federated architecture
    ref_model = build_resnet50_binary(pretrained=False, device="cpu")
    ref_state_dict = ref_model.state_dict()

    ref_keys = set(ref_state_dict.keys())
    ckpt_keys = set(state_dict.keys())

    missing_in_ckpt = sorted(list(ref_keys - ckpt_keys))
    unexpected_in_ckpt = sorted(list(ckpt_keys - ref_keys))

    # 3. Shape verification
    shape_mismatches = []
    matched_params = 0
    total_elements = 0
    for key in ref_keys.intersection(ckpt_keys):
        ref_shape = list(ref_state_dict[key].shape)
        ckpt_shape = list(state_dict[key].shape)
        if ref_shape != ckpt_shape:
            shape_mismatches.append({"key": key, "expected_shape": ref_shape, "actual_shape": ckpt_shape})
        else:
            matched_params += 1
            total_elements += state_dict[key].numel()

    # 4. Classification head verification
    classifier_weight_shape = list(state_dict.get("classifier.weight", torch.tensor([])).shape)
    classifier_bias_shape = list(state_dict.get("classifier.bias", torch.tensor([])).shape)

    head_ok = (classifier_weight_shape == [1, 2048] and classifier_bias_shape == [1])

    # 5. Model load and forward pass
    model = build_resnet50_binary(pretrained=False, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)

    forward_ok = (logits.shape == torch.Size([2, 1]) and probs.shape == torch.Size([2, 1]))

    is_compatible = (
        len(missing_in_ckpt) == 0
        and len(unexpected_in_ckpt) == 0
        and len(shape_mismatches) == 0
        and head_ok
        and forward_ok
    )

    report = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_val_metric": ckpt.get("best_val_metric"),
        "metric_name": ckpt.get("metric_name"),
        "total_state_tensors": len(state_dict),
        "expected_state_tensors": len(ref_state_dict),
        "matched_tensors_count": matched_params,
        "total_parameters_count": total_elements,
        "missing_keys": missing_in_ckpt,
        "unexpected_keys": unexpected_in_ckpt,
        "shape_mismatches": shape_mismatches,
        "classification_head": {
            "weight_shape": classifier_weight_shape,
            "bias_shape": classifier_bias_shape,
            "expected_weight_shape": [1, 2048],
            "expected_bias_shape": [1],
            "verified": head_ok,
        },
        "forward_pass_test": {
            "input_shape": list(dummy_input.shape),
            "output_logits_shape": list(logits.shape),
            "expected_logits_shape": [2, 1],
            "verified": forward_ok,
        },
        "federated_aggregation_compatible": is_compatible,
        "status": "PASS" if is_compatible else "FAIL",
    }

    print(f"\n[Parameters] Total Tensors: {len(state_dict)} / {len(ref_state_dict)} matched")
    print(f"[Parameters] Total Weights: {total_elements:,}")
    print(f"[Head Spec]  Linear(2048 -> 1) Single-Logit: {'VERIFIED' if head_ok else 'FAILED'}")
    print(f"[Inference]  Forward pass logits shape: {list(logits.shape)}: {'VERIFIED' if forward_ok else 'FAILED'}")
    print(f"[Federated]  FedAvg Parameter Compatibility: {'PASSED' if is_compatible else 'FAILED'}")
    print("=" * 70 + "\n")

    if output_json:
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved compatibility report to: {output_json}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify federated checkpoint compatibility.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save verification report JSON.")
    args = parser.parse_args()

    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        # Search for latest best_checkpoint.pt in outputs/padchest
        out_root = REPO_ROOT / "outputs" / "padchest"
        candidates = sorted(out_root.glob("*/best_checkpoint.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            ckpt_path = candidates[0]
        else:
            print("No checkpoint found in outputs/padchest/*/best_checkpoint.pt. Please specify --checkpoint.")
            return 1

    out_json = Path(args.output_json).resolve() if args.output_json else ckpt_path.parent / "checkpoint_compatibility_report.json"
    report = verify_checkpoint(ckpt_path, out_json)
    return 0 if report["federated_aggregation_compatible"] else 1


if __name__ == "__main__":
    sys.exit(main())
