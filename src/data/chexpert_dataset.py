"""PyTorch Dataset, transforms, and path resolver for CheXpert (Client 02).

Standardized for binary classification of Pleural Effusion on CheXpert.
Supports:
  - Robust cross-platform path resolution across nested patient/study/view layouts
  - Explicit exclusion of macOS sidecar files (._*)
  - Frontal-view filtering (Frontal AP/PA only) based on metadata
  - Availability audits for incomplete local archives
  - ImageNet normalization with 1-channel to 3-channel RGB expansion
  - Dual image loader (PIL with OpenCV fallback)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .nih_dataset import IMAGENET_MEAN, IMAGENET_STD, load_image_rgb
from .path_resolver import normalize_cross_platform_path


def extract_chexpert_patient_key(path_str: str | Path | None) -> str | None:
    """Extract canonical 'patientXXXXX/studyY/viewZ_frontal.jpg' subpath identifier."""
    if path_str is None or pd.isna(path_str):
        return None
    clean = str(path_str).replace("\\", "/").strip()
    match = re.search(r"(patient\d+/study\d+/[^/]+)$", clean, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


class CheXpertPathResolver:
    """Resolves CheXpert image paths to absolute filesystem paths.

    CheXpert images typically reside under:
      - <root>/CheXpert-v1.0-small/train/patientXXXXX/studyY/viewZ_*.jpg
      - <root>/train/patientXXXXX/studyY/viewZ_*.jpg
      - <root>/images/patientXXXXX/studyY/viewZ_*.jpg
      - <root>/patientXXXXX/studyY/viewZ_*.jpg
    Also safely ignores macOS metadata files (e.g. ._view1_frontal.jpg).
    """

    def __init__(
        self,
        search_roots: Sequence[str | Path] | str | Path | None = None,
        max_scan_depth: int = 6,
    ) -> None:
        self.search_roots: list[Path] = []
        if search_roots is not None:
            if isinstance(search_roots, (str, Path)):
                self.search_roots.append(Path(search_roots).expanduser().resolve())
            else:
                for root in search_roots:
                    self.search_roots.append(Path(root).expanduser().resolve())

        if search_roots is None:
            cwd = Path.cwd().resolve()
            standard_defaults = [
                cwd / "data" / "raw" / "CheXpert" / "images",
                cwd / "data" / "raw" / "CheXpert" / "CheXpert-v1.0-small",
                cwd / "data" / "raw" / "CheXpert" / "CheXpert-v1.0",
                cwd / "data" / "raw" / "CheXpert",
                cwd / "data" / "raw",
            ]
            for candidate in standard_defaults:
                if candidate not in self.search_roots and candidate.exists():
                    self.search_roots.append(candidate)

        self.max_scan_depth = max_scan_depth
        # Index stores multiple lookup keys -> absolute Path
        self._index: dict[str, Path] = {}
        self._indexed: bool = False

    def build_index(self, force_refresh: bool = False) -> int:
        """Scan search roots and index all valid radiographs, skipping macOS sidecars."""
        if self._indexed and not force_refresh:
            return len(self._index)

        valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        new_index: dict[str, Path] = {}

        for root in self.search_roots:
            if not root.is_dir():
                continue

            root_depth = len(root.parts)
            for dirpath, _, filenames in os.walk(root):
                current_depth = len(Path(dirpath).parts) - root_depth
                if current_depth > self.max_scan_depth:
                    continue

                for fname in filenames:
                    # STRICT RULE: Ignore macOS metadata sidecars
                    if fname.startswith("._"):
                        continue

                    ext = os.path.splitext(fname)[1]
                    if ext not in valid_extensions:
                        continue

                    full_path = (Path(dirpath) / fname).resolve()
                    clean_str = str(full_path).replace("\\", "/")

                    # 1. Key by canonical patient subpath (e.g. 'patient10592/study1/view1_frontal.jpg')
                    pkey = extract_chexpert_patient_key(clean_str)
                    if pkey and pkey not in new_index:
                        new_index[pkey] = full_path

                    # 2. Key by relative path from the scanned root
                    try:
                        rel_to_root = full_path.relative_to(root).as_posix().lower()
                        if rel_to_root not in new_index:
                            new_index[rel_to_root] = full_path
                    except ValueError:
                        pass

                    # 3. Key by relative path from 'CheXpert' or 'train'/'valid' if present
                    match_tv = re.search(r"((train|valid)/patient\d+/study\d+/[^/]+)$", clean_str, re.IGNORECASE)
                    if match_tv:
                        tv_key = match_tv.group(1).lower()
                        if tv_key not in new_index:
                            new_index[tv_key] = full_path

        self._index = new_index
        self._indexed = True
        return len(self._index)

    def resolve(self, image_id: str, recorded_path: str | None = None) -> Path | None:
        """Resolve a single CheXpert image to an absolute path on disk."""
        if not self._indexed:
            self.build_index()

        # 1. Check canonical patient subpath lookup (O(1))
        for candidate_str in (recorded_path, image_id):
            if candidate_str:
                pkey = extract_chexpert_patient_key(candidate_str)
                if pkey and pkey in self._index:
                    return self._index[pkey]

                # Check train/valid relative subpath
                match_tv = re.search(r"((train|valid)/patient\d+/study\d+/[^/]+)$", str(candidate_str).replace("\\", "/"), re.IGNORECASE)
                if match_tv:
                    tv_key = match_tv.group(1).lower()
                    if tv_key in self._index:
                        return self._index[tv_key]

        # 2. Check direct cross-platform normalized path
        if recorded_path and pd.notna(recorded_path):
            norm = normalize_cross_platform_path(recorded_path)
            if norm is not None and not norm.name.startswith("._"):
                return norm

        return None

    def audit_split_availability(
        self,
        frame: pd.DataFrame,
        split_name: str = "split",
    ) -> dict[str, Any]:
        """Perform a thorough audit of accessible vs missing images for a split."""
        if not self._indexed:
            self.build_index()

        total_requested = len(frame)
        total_patients = int(frame["patient_id"].nunique(dropna=True)) if "patient_id" in frame.columns else 0

        has_image_path = "image_path" in frame.columns
        has_target = "target" in frame.columns

        official_pos = int((frame["target"] == 1).sum()) if has_target else 0
        official_neg = int((frame["target"] == 0).sum()) if has_target else 0
        official_prev = official_pos / total_requested if total_requested > 0 else 0.0

        img_ids = frame["image_id"].astype(str).str.strip().tolist()
        rec_paths = frame["image_path"].tolist() if has_image_path else [None] * total_requested
        patient_ids = frame["patient_id"].astype(str).tolist() if "patient_id" in frame.columns else [""] * total_requested
        targets = frame["target"].tolist() if has_target else [-1] * total_requested

        accessible_rows: list[int] = []
        missing_records: list[dict[str, Any]] = []

        for idx, (img_id, rec_path, pid, tgt) in enumerate(zip(img_ids, rec_paths, patient_ids, targets)):
            resolved = self.resolve(img_id, rec_path)
            if resolved is not None:
                accessible_rows.append(idx)
            else:
                missing_records.append({
                    "image_id": img_id,
                    "patient_id": pid,
                    "target": int(tgt) if pd.notna(tgt) else None,
                })

        accessible_count = len(accessible_rows)
        missing_count = len(missing_records)
        accessible_fraction = accessible_count / total_requested if total_requested > 0 else 0.0
        missing_fraction = missing_count / total_requested if total_requested > 0 else 0.0

        accessible_subset = frame.iloc[accessible_rows]
        accessible_patients = int(accessible_subset["patient_id"].nunique(dropna=True)) if "patient_id" in frame.columns else 0
        accessible_pos = int((accessible_subset["target"] == 1).sum()) if has_target else 0
        accessible_neg = int((accessible_subset["target"] == 0).sum()) if has_target else 0
        accessible_prev = accessible_pos / accessible_count if accessible_count > 0 else 0.0

        is_complete = (missing_count == 0)

        return {
            "split": split_name,
            "is_complete": is_complete,
            "coverage_status": "COMPLETE_OFFICIAL_SPLIT" if is_complete else "PARTIAL_ARCHIVE_INCOMPLETE",
            "total_official_images": total_requested,
            "total_official_patients": total_patients,
            "accessible_images": accessible_count,
            "missing_images": missing_count,
            "accessible_fraction": accessible_fraction,
            "missing_fraction": missing_fraction,
            "accessible_patients": accessible_patients,
            "missing_patients": max(0, total_patients - accessible_patients),
            "official_class_distribution": {
                "positive": official_pos,
                "negative": official_neg,
                "prevalence": official_prev,
            },
            "accessible_class_distribution": {
                "positive": accessible_pos,
                "negative": accessible_neg,
                "prevalence": accessible_prev,
            },
            "searched_roots": [str(r) for r in self.search_roots],
            "total_indexed_on_disk": len(self._index),
            "sample_missing_image_ids": [m["image_id"] for m in missing_records[:10]],
        }

    def get_accessible_patient_ids(self, frame: pd.DataFrame) -> set[str]:
        """Return the set of unique patient IDs from frame with >=1 accessible image on disk."""
        if not self._indexed:
            self.build_index()

        accessible_pts: set[str] = set()
        img_ids = frame["image_id"].astype(str).str.strip().tolist()
        rec_paths = frame["image_path"].tolist() if "image_path" in frame.columns else [None] * len(frame)
        pids = frame["patient_id"].astype(str).tolist()

        for img_id, rec_path, pid in zip(img_ids, rec_paths, pids):
            resolved = self.resolve(img_id, rec_path)
            if resolved is not None:
                accessible_pts.add(pid)

        return accessible_pts

    def resolve_dataframe(
        self,
        frame: pd.DataFrame,
        split_name: str = "split",
        strict: bool = False,
        filter_missing: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Resolve paths for all records in a split and return filtered df + audit report."""
        audit_report = self.audit_split_availability(frame, split_name=split_name)

        if audit_report["missing_images"] > 0 and strict:
            sample_ids = audit_report["sample_missing_image_ids"][:5]
            raise FileNotFoundError(
                f"Strict path resolution failed for CheXpert '{split_name}' split! "
                f"{audit_report['missing_images']} out of {audit_report['total_official_images']} "
                f"images were not found on disk. Sample missing IDs: {sample_ids}. "
                f"Searched roots: {audit_report['searched_roots']}."
            )

        df_out = frame.copy()
        img_ids = df_out["image_id"].astype(str).str.strip().tolist()
        rec_paths = df_out["image_path"].tolist() if "image_path" in df_out.columns else [None] * len(df_out)

        resolved_paths: list[str | None] = []
        for img_id, rec_path in zip(img_ids, rec_paths):
            resolved = self.resolve(img_id, rec_path)
            resolved_paths.append(str(resolved) if resolved is not None else None)

        df_out["resolved_path"] = resolved_paths

        if filter_missing and audit_report["missing_images"] > 0:
            df_out = df_out[df_out["resolved_path"].notna()].copy().reset_index(drop=True)
            audit_report["filtered_to_available"] = True
            audit_report["final_image_count"] = len(df_out)
        else:
            audit_report["filtered_to_available"] = False
            audit_report["final_image_count"] = len(df_out)

        return df_out, audit_report


def filter_chexpert_frontal_views(
    frame: pd.DataFrame,
    split_name: str = "split",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter CheXpert metadata for frontal radiographs only (AP/PA).

    Frontal policy:
      - view_position must be 'Frontal' (case-insensitive)
      - projection must be 'AP' or 'PA' (case-insensitive)
      - Excludes 'Lateral' views and rare lateral orientation tags (LL, RL).
    """
    total_before = len(frame)
    if total_before == 0:
        return frame.copy(), {"total_before": 0, "frontal_count": 0, "dropped": 0}

    # Normalize view_position and projection
    view_col = frame["view_position"].astype(str).str.strip().str.capitalize()
    proj_col = frame["projection"].astype(str).str.strip().str.upper()

    is_frontal = view_col == "Frontal"
    is_ap_or_pa = proj_col.isin(["AP", "PA"])

    frontal_mask = is_frontal & is_ap_or_pa
    filtered_df = frame[frontal_mask].copy().reset_index(drop=True)

    lateral_count = int((view_col == "Lateral").sum())
    other_dropped = total_before - len(filtered_df) - lateral_count

    report = {
        "split": split_name,
        "total_before_filtering": total_before,
        "frontal_images_retained": len(filtered_df),
        "lateral_images_excluded": lateral_count,
        "non_ap_pa_images_excluded": other_dropped,
        "frontal_retention_fraction": len(filtered_df) / total_before if total_before > 0 else 0.0,
        "projection_distribution": filtered_df["projection"].value_counts().to_dict(),
        "class_distribution": {
            "positive": int((filtered_df["target"] == 1).sum()),
            "negative": int((filtered_df["target"] == 0).sum()),
            "prevalence": float((filtered_df["target"] == 1).mean()) if len(filtered_df) > 0 else 0.0,
        },
    }

    return filtered_df, report


def get_chexpert_transforms(
    split: str = "train",
    img_size: int = 224,
) -> Callable[[Any], torch.Tensor]:
    """Return image preprocessing and augmentation transforms for CheXpert."""
    try:
        import torchvision.transforms as T
        if split == "train":
            return T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
    except ImportError:
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

        def fallback_transform(img_array: np.ndarray) -> torch.Tensor:
            if not isinstance(img_array, np.ndarray):
                img_array = np.array(img_array)

            try:
                import cv2
                resized = cv2.resize(img_array, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                h, w, c = img_array.shape
                y_idx = (np.linspace(0, h - 1, img_size)).astype(int)
                x_idx = (np.linspace(0, w - 1, img_size)).astype(int)
                resized = img_array[np.ix_(y_idx, x_idx)]

            if split == "train" and np.random.rand() > 0.5:
                resized = np.fliplr(resized).copy()

            norm = (resized.astype(np.float32) / 255.0 - mean) / std
            return torch.from_numpy(norm.transpose(2, 0, 1)).float()

        return fallback_transform


class CheXpertDataset(Dataset):
    """PyTorch Dataset for CheXpert binary Pleural Effusion classification."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable[[Any], torch.Tensor] | None = None,
        return_metadata: bool = False,
    ) -> None:
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

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.frame.iloc[idx]
        image_path = row[self.path_col]

        if pd.isna(image_path) or not image_path:
            raise FileNotFoundError(
                f"Missing image path for row {idx} (image_id='{row.get('image_id', 'unknown')}')"
            )

        try:
            img = load_image_rgb(image_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image for CheXpert sample {idx} (image_id='{row.get('image_id')}', "
                f"path='{image_path}'): {exc}"
            ) from exc

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        target_value = float(row["target"])
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        if self.return_metadata:
            meta = {
                "image_id": str(row.get("image_id", "")),
                "patient_id": str(row.get("patient_id", "")),
                "study_id": str(row.get("study_id", "")),
                "view_position": str(row.get("view_position", "")),
                "projection": str(row.get("projection", "")),
                "image_path": str(image_path),
            }
            return image_tensor, target_tensor, meta

        return image_tensor, target_tensor


def create_chexpert_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create PyTorch DataLoaders for CheXpert splits."""
    train_transform = get_chexpert_transforms(split="train", img_size=img_size)
    eval_transform = get_chexpert_transforms(split="val", img_size=img_size)

    train_ds = CheXpertDataset(train_df, transform=train_transform)
    val_ds = CheXpertDataset(val_df, transform=eval_transform)

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
        test_ds = CheXpertDataset(test_df, transform=eval_transform)
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return loaders
