"""Verify Python environment, dependencies, and ResNet-50 model execution for PadChest Client 03."""

import sys
from pathlib import Path

def main() -> int:
    print("=" * 70)
    print("  PadChest Client 03 — Environment & Dependency Verification")
    print("=" * 70)

    # 1. Python version check
    py_ver = sys.version_info
    print(f"\n[1/6] Python Version: {sys.version.split()[0]}")
    if py_ver.major != 3 or py_ver.minor < 10:
        print(f"  [ERROR] Python >= 3.10 required. Detected: {sys.version}")
        return 1
    print("  --> Python version is compatible.")

    # 2. Virtual environment check
    is_venv = sys.prefix != sys.base_prefix
    print(f"\n[2/6] Virtual Environment: {'Active (' + sys.prefix + ')' if is_venv else 'Not detected'}")

    # 3. Core dependencies import check
    print("\n[3/6] Verifying Required Package Imports...")
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-Learn"),
        ("yaml", "PyYAML"),
    ]

    all_passed = True
    for mod_name, label in required_packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [PASS] {label:<15}: {ver}")
        except ImportError as e:
            print(f"  [FAIL] {label:<15}: Missing ({e})")
            all_passed = False

    if not all_passed:
        print("\n  [ERROR] Missing required packages. Run: pip install -r requirements.txt")
        return 1

    # 4. Hardware and PyTorch CPU device check
    import torch
    print(f"\n[4/6] PyTorch Device Configuration...")
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cpu")
    print(f"  PyTorch Build:    {torch.__version__}")
    print(f"  CUDA Available:   {cuda_avail}")
    print(f"  Resolved Device:  {device} (CPU)")
    if cuda_avail:
        print("  [NOTE] GPU detected, but Client 03 training is configured for CPU execution.")
    else:
        print("  [NOTE] Confirmed CPU-only execution environment.")

    # 5. Shared ResNet-50 Binary Model Instantiation
    print("\n[5/6] Verifying ResNet-50 Binary Architecture...")
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from src.models.resnet import build_resnet50_binary
        model = build_resnet50_binary(pretrained=False, device=device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_count = len(list(model.state_dict().keys()))
        print(f"  Architecture:     ResNet50Binary (torchvision ResNet-50 backbone)")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Params: {trainable_params:,}")
        print(f"  State Dict Tensors: {param_count} (expected 320)")
        assert param_count == 320, f"Expected 320 state dict parameters, got {param_count}"
        assert model.classifier.out_features == 1, "Expected single logit classifier output"
        print("  --> Model architecture verified strictly against federated standard.")
    except Exception as e:
        print(f"  [ERROR] Model instantiation failed: {e}")
        return 1

    # 6. Dummy Forward Pass Test
    print("\n[6/6] Executing Dummy Forward Pass...")
    try:
        dummy_input = torch.randn(2, 3, 224, 224, device=device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            probs = model.predict_proba(dummy_input)
        print(f"  Input Tensor:     shape={list(dummy_input.shape)}, dtype={dummy_input.dtype}")
        print(f"  Output Logits:    shape={list(output.shape)} (expected [2, 1])")
        print(f"  Probabilities:    shape={list(probs.shape)} (range: [{probs.min():.4f}, {probs.max():.4f}])")
        assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
        assert probs.shape == (2, 1), f"Expected probs shape (2, 1), got {probs.shape}"
        print("  --> Forward pass succeeded.")
    except Exception as e:
        print(f"  [ERROR] Forward pass failed: {e}")
        return 1

    print("\n" + "=" * 70)
    print("  [SUCCESS] All Environment & Architecture Verifications Passed!")
    print("=" * 70 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
