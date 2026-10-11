"""Robust cross-platform image path resolver for NIH ChestX-ray14 archives.

Handles arbitrary archive structures across Windows and WSL / Linux environments:
  - Flat folder: <image_dir>/<image_id>
  - Archive subfolder: <image_dir>/images_001/<image_id>
  - Nested archive: <image_dir>/images_001/images/<image_id>
  - Multi-archive: <image_dir>/images_001/, images_002/, ..., images_012/
  - Preserved relative or Windows/WSL path translation from CSV metadata

Provides comprehensive split availability audits (accessible vs missing counts)
for scientific integrity when working with partial archive downloads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def normalize_cross_platform_path(path_str: str | Path | None) -> Path | None:
    """Normalize and translate file paths across Windows, WSL, and Linux."""
    if path_str is None or pd.isna(path_str):
        return None

    clean_str = str(path_str).strip()
    if not clean_str:
        return None

    # Check direct path first
    direct = Path(clean_str)
    if direct.is_file():
        return direct.resolve()

    # Translate Windows drive letter on POSIX / WSL (e.g. E:\foo\bar -> /mnt/e/foo/bar)
    if os.name == "posix" and len(clean_str) >= 3 and clean_str[1] == ":" and clean_str[2] in ("\\", "/"):
        drive = clean_str[0].lower()
        rest = clean_str[3:].replace("\\", "/")
        wsl_path = Path(f"/mnt/{drive}/{rest}")
        if wsl_path.is_file():
            return wsl_path.resolve()

    # Normalize forward/back slashes for current OS
    norm = Path(os.path.normpath(clean_str))
    if norm.is_file():
        return norm.resolve()

    return None


class NIHPathResolver:
    """Resolves NIH image filenames to absolute filesystem paths with cross-platform support."""

    def __init__(
        self,
        search_roots: Sequence[str | Path] | str | Path | None = None,
        max_scan_depth: int = 4,
    ) -> None:
        self.search_roots: list[Path] = []
        if search_roots is not None:
            if isinstance(search_roots, (str, Path)):
                self.search_roots.append(Path(search_roots).expanduser().resolve())
            else:
                for root in search_roots:
                    self.search_roots.append(Path(root).expanduser().resolve())

        # Include standard repository paths if search_roots is None or requested
        if search_roots is None:
            cwd = Path.cwd().resolve()
            standard_defaults = [
                cwd / "data" / "raw" / "NIH" / "images",
                cwd / "data" / "raw" / "NIH",
                cwd / "data" / "raw",
            ]
            for candidate in standard_defaults:
                if candidate not in self.search_roots and candidate.exists():
                    self.search_roots.append(candidate)

        self.max_scan_depth = max_scan_depth
        self._file_index: dict[str, Path] = {}
        self._indexed: bool = False

    def build_index(self, force_refresh: bool = False) -> int:
        """Scan search roots and index all image files by basename."""
        if self._indexed and not force_refresh:
            return len(self._file_index)

        valid_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
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
                    ext = os.path.splitext(fname)[1]
                    if ext in valid_extensions and fname not in new_index:
                        new_index[fname] = (Path(dirpath) / fname).resolve()

        self._file_index = new_index
        self._indexed = True
        return len(self._file_index)

    def resolve(self, image_id: str, recorded_path: str | None = None) -> Path | None:
        """Resolve a single image ID to its existing file path on disk.

        Prioritizes fast in-memory index lookup before falling back to filesystem checks.
        """
        clean_id = str(image_id).strip()

        # 1. Fast in-memory lookup from scanned search roots (O(1))
        if not self._indexed:
            self.build_index()

        if clean_id in self._file_index:
            return self._file_index[clean_id]

        # 2. Check normalized recorded_path from CSV
        if recorded_path and pd.notna(recorded_path):
            norm_path = normalize_cross_platform_path(recorded_path)
            if norm_path is not None:
                self._file_index[clean_id] = norm_path
                return norm_path

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

        official_pos_images = int((frame["target"] == 1).sum()) if has_target else 0
        official_neg_images = int((frame["target"] == 0).sum()) if has_target else 0
        official_prev = official_pos_images / total_requested if total_requested > 0 else 0.0

        # Fast list-based iteration (1000x faster than iterrows)
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
        accessible_pos_images = int((accessible_subset["target"] == 1).sum()) if has_target else 0
        accessible_neg_images = int((accessible_subset["target"] == 0).sum()) if has_target else 0
        accessible_prev = accessible_pos_images / accessible_count if accessible_count > 0 else 0.0

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
                "positive": official_pos_images,
                "negative": official_neg_images,
                "prevalence": official_prev,
            },
            "accessible_class_distribution": {
                "positive": accessible_pos_images,
                "negative": accessible_neg_images,
                "prevalence": accessible_prev,
            },
            "searched_roots": [str(r) for r in self.search_roots],
            "total_indexed_on_disk": len(self._file_index),
            "sample_missing_image_ids": [m["image_id"] for m in missing_records[:10]],
        }

    def get_accessible_patient_ids(self, frame: pd.DataFrame) -> set[str]:
        """Return the set of unique patient IDs from frame who have >=1 accessible image on disk."""
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
                f"Strict path resolution failed for '{split_name}' split! "
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
