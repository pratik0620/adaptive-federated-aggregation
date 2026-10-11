"""PyTorch Dataset and transform pipelines for NIH ChestX-ray14.

Features:
  - Standard ImageNet preprocessing (Resize 224x224, ImageNet normalization)
  - Explicit channel handling (converts 1-channel grayscale radiographs to 3-channel RGB)
  - Dual image loader support (PIL with OpenCV fallback)
  - Explicit error reporting for missing, corrupted, or unreadable image files
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file and return it as an RGB float32 or uint8 array.

    Supports PIL or OpenCV, explicitly raising descriptive errors on corrupted files.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Image file does not exist at path: '{path_obj}'")

    # Try PIL first
    try:
        from PIL import Image
        with Image.open(path_obj) as img:
            rgb_img = img.convert("RGB")
            return np.array(rgb_img)
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Corrupted or unreadable image file via PIL at '{path_obj}': {exc}") from exc

    # Try OpenCV fallback
    try:
        import cv2
        bgr = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV could not decode image at '{path_obj}' (returned None)")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    except ImportError as exc:
        raise ImportError(
            "Neither PIL (Pillow) nor OpenCV (cv2) is available to read images. "
            "Please install Pillow: pip install pillow"
        ) from exc


def get_nih_transforms(
    split: str = "train",
    img_size: int = 224,
) -> Callable[[Any], torch.Tensor]:
    """Return image preprocessing and augmentation transforms.

    Args:
        split: 'train' (includes random horizontal flip) or 'val'/'test' (deterministic resize).
        img_size: Target square image dimension (default: 224 for ResNet-50).

    Returns:
        Transform callable converting image to normalized torch.Tensor of shape (3, H, W).
    """
    try:
        import torchvision.transforms as T
        from PIL import Image

        if split == "train":
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return transform

    except ImportError:
        # Pure NumPy / PyTorch fallback if torchvision is not yet installed
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

        def fallback_transform(img_array: np.ndarray) -> torch.Tensor:
            if not isinstance(img_array, np.ndarray):
                img_array = np.array(img_array)

            # Resize using OpenCV or simple nearest/bilinear interpolation
            try:
                import cv2
                resized = cv2.resize(img_array, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                # Basic nearest-neighbor resize fallback
                h, w, c = img_array.shape
                y_idx = (np.linspace(0, h - 1, img_size)).astype(int)
                x_idx = (np.linspace(0, w - 1, img_size)).astype(int)
                resized = img_array[np.ix_(y_idx, x_idx)]

            if split == "train" and np.random.rand() > 0.5:
                resized = np.fliplr(resized).copy()

            # Normalize to [0, 1] then ImageNet standardize
            norm = (resized.astype(np.float32) / 255.0 - mean) / std
            # Transpose (H, W, C) -> (C, H, W)
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
            return tensor

        return fallback_transform


class NIHChestXRayDataset(Dataset):
    """PyTorch Dataset for NIH ChestX-ray14 binary classification."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable[[Any], torch.Tensor] | None = None,
        return_metadata: bool = False,
    ) -> None:
        """Initialize NIH dataset.

        Args:
            frame: DataFrame with 'target' and 'resolved_path' (or 'image_path').
            transform: Callable transform pipeline.
            return_metadata: If True, __getitem__ returns (image, target, metadata_dict).
        """
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.return_metadata = return_metadata

        if "resolved_path" in self.frame.columns:
            self.path_col = "resolved_path"
        elif "image_path" in self.frame.columns:
            self.path_col = "image_path"
        else:
            raise KeyError("DataFrame must contain 'resolved_path' or 'image_path' column.")

        if "target" not in self.frame.columns:
            raise KeyError("DataFrame must contain binary 'target' column.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.frame.iloc[idx]
        image_path = row[self.path_col]

        if pd.isna(image_path) or not image_path:
            raise FileNotFoundError(
                f"Missing image path for row {idx} (image_id='{row.get('image_id', 'unknown')}')"
            )

        # Load image with explicit error handling
        try:
            img = load_image_rgb(image_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image for sample {idx} (image_id='{row.get('image_id')}', "
                f"path='{image_path}'): {exc}"
            ) from exc

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            # DefaultToTensor if no transform supplied
            image_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        target_value = float(row["target"])
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        if self.return_metadata:
            meta = {
                "image_id": str(row.get("image_id", "")),
                "patient_id": str(row.get("patient_id", "")),
                "view_position": str(row.get("view_position", "")),
                "image_path": str(image_path),
            }
            return image_tensor, target_tensor, meta

        return image_tensor, target_tensor


def create_nih_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and optional test splits."""
    train_transform = get_nih_transforms(split="train", img_size=img_size)
    eval_transform = get_nih_transforms(split="val", img_size=img_size)

    train_ds = NIHChestXRayDataset(train_df, transform=train_transform)
    val_ds = NIHChestXRayDataset(val_df, transform=eval_transform)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(len(train_ds) > batch_size),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }

    if test_df is not None and not test_df.empty:
        test_ds = NIHChestXRayDataset(test_df, transform=eval_transform)
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return loaders
