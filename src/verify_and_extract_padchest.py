"""PadChest Archive Extraction & Image Integrity Verification Script.

Extracts archives (0.zip, 1.zip) into data/raw/PadChest/images/ safely,
verifies files against archive manifests (0.zip.unzip-l.txt, 1.zip.unzip-l.txt),
detects corruptions, duplicates, and missing files, and writes an audit report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


def parse_manifest(manifest_path: Path) -> dict[str, int]:
    """Parse unzip -l manifest file into {filename: expected_size_bytes}."""
    expected_files: dict[str, int] = {}
    if not manifest_path.is_file():
        return expected_files

    # Match unzip -l format: Length Date Time Name
    line_pattern = re.compile(r"^\s*(\d+)\s+[\d\-]+\s+[\d:]+\s+(.+)$")

    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = line_pattern.match(line)
            if match:
                size_str, name = match.group(1), match.group(2).strip()
                if name.startswith("Archive:") or name.startswith("---------") or "files" in name:
                    continue
                basename = Path(name).name
                if basename.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")):
                    expected_files[basename] = int(size_str)

    return expected_files


def audit_archive_against_manifest(zip_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify archive zip structure against its unzip -l manifest file without extracting."""
    archive_name = zip_path.name
    print(f"\n[Audit] Inspecting {archive_name} against {manifest_path.name}...")

    manifest_files = parse_manifest(manifest_path)
    if not zip_path.is_file():
        return {
            "archive": archive_name,
            "exists": False,
            "status": "MISSING",
            "manifest_file_count": len(manifest_files),
            "archive_file_count": 0,
            "missing_in_zip": len(manifest_files),
            "size_mismatches": 0,
        }

    zip_entries: dict[str, int] = {}
    total_uncompressed_bytes = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            fname = Path(info.filename).name
            if fname.startswith("._") or not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            zip_entries[fname] = info.file_size
            total_uncompressed_bytes += info.file_size

    missing_in_zip = sorted(list(set(manifest_files.keys()) - set(zip_entries.keys())))
    extra_in_zip = sorted(list(set(zip_entries.keys()) - set(manifest_files.keys())))
    size_mismatches = [
        {"filename": k, "manifest_size": manifest_files[k], "zip_entry_size": zip_entries[k]}
        for k in manifest_files
        if k in zip_entries and manifest_files[k] != zip_entries[k]
    ]

    is_valid = (len(missing_in_zip) == 0 and len(size_mismatches) == 0)
    audit_res = {
        "archive": archive_name,
        "archive_size_bytes": zip_path.stat().st_size,
        "exists": True,
        "status": "PASSED" if is_valid else "FAILED",
        "manifest_file_count": len(manifest_files),
        "archive_file_count": len(zip_entries),
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "missing_in_zip_count": len(missing_in_zip),
        "missing_in_zip_sample": missing_in_zip[:5],
        "extra_in_zip_count": len(extra_in_zip),
        "size_mismatches_count": len(size_mismatches),
    }

    print(f"  Archive Size (compressed):   {zip_path.stat().st_size:,} bytes ({zip_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"  Manifest Expected Entries:   {len(manifest_files):,}")
    print(f"  Archive Found Valid Entries: {len(zip_entries):,}")
    print(f"  Total Uncompressed Size:     {total_uncompressed_bytes:,} bytes ({total_uncompressed_bytes / (1024**3):.2f} GB)")
    print(f"  Missing in Zip:              {len(missing_in_zip)}")
    print(f"  Size Mismatches:             {len(size_mismatches)}")
    print(f"  Archive Audit Result:        {audit_res['status']}")

    return audit_res


def extract_archive(
    zip_path: Path,
    dest_dir: Path,
    expected_manifest: dict[str, int] | None = None,
    verify_readability: bool = False,
) -> dict[str, Any]:
    """Extract files from a zip archive into dest_dir, avoiding re-extraction of valid files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "archive": zip_path.name,
        "already_valid": 0,
        "extracted_new": 0,
        "corrupted_skipped": 0,
        "errors": [],
    }

    print(f"\n[Extraction] Processing archive: {zip_path.name} -> {dest_dir}")
    if not zip_path.is_file():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    start_t = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()
        total_in_zip = len(infolist)
        print(f"  Total entries in archive: {total_in_zip:,}")

        for idx, info in enumerate(infolist, 1):
            fname = Path(info.filename).name
            if fname.startswith("._") or not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            target_file = dest_dir / fname
            expected_sz = expected_manifest.get(fname, info.file_size) if expected_manifest else info.file_size

            # Check if valid file already exists on disk with matching file size
            if target_file.is_file() and target_file.stat().st_size == expected_sz:
                stats["already_valid"] += 1
                if idx % 500 == 0 or idx == total_in_zip:
                    print(f"    [{idx:,}/{total_in_zip:,}] Skipped already valid: {stats['already_valid']:,} | Extracted: {stats['extracted_new']:,}")
                continue

            # Extract file safely
            try:
                with zf.open(info) as src, open(target_file, "wb") as dst:
                    while True:
                        buf = src.read(1024 * 1024)
                        if not buf:
                            break
                        dst.write(buf)

                # Quick integrity check
                if verify_readability:
                    try:
                        with Image.open(target_file) as img:
                            img.verify()
                    except Exception as img_err:
                        stats["corrupted_skipped"] += 1
                        stats["errors"].append(f"{fname}: image verify failed ({img_err})")
                        target_file.unlink(missing_ok=True)
                        continue

                stats["extracted_new"] += 1
            except Exception as e:
                stats["corrupted_skipped"] += 1
                stats["errors"].append(f"{fname}: extraction error ({e})")
                if target_file.is_file():
                    target_file.unlink(missing_ok=True)

            if idx % 500 == 0 or idx == total_in_zip:
                print(f"    [{idx:,}/{total_in_zip:,}] Skipped already valid: {stats['already_valid']:,} | Extracted: {stats['extracted_new']:,}")

    elapsed = time.time() - start_t
    print(f"  Archive {zip_path.name} finished in {elapsed:.1f}s. "
          f"Valid: {stats['already_valid']:,}, New Extracted: {stats['extracted_new']:,}, Errors: {len(stats['errors'])}")
    return stats


def verify_extracted_images(
    images_dir: Path,
    manifest_0: dict[str, int],
    manifest_1: dict[str, int],
    quick_sample_check: int = 100,
) -> dict[str, Any]:
    """Verify all images in images_dir against manifests, checking for corrupt, missing, and duplicate files."""
    print(f"\n[Verification] Auditing extracted images in: {images_dir}")
    all_expected = dict(manifest_0)
    all_expected.update(manifest_1)

    existing_files: dict[str, int] = {}
    corrupted_files: list[str] = []
    zero_byte_files: list[str] = []
    duplicates: list[str] = []

    for p in images_dir.iterdir():
        if p.is_file() and not p.name.startswith("._") and p.name.lower().endswith((".png", ".jpg", ".jpeg")):
            if p.name in existing_files:
                duplicates.append(p.name)
            sz = p.stat().st_size
            if sz == 0:
                zero_byte_files.append(p.name)
            else:
                existing_files[p.name] = sz

    total_existing = len(existing_files)
    total_expected = len(all_expected)

    missing_from_manifests = [fname for fname in all_expected if fname not in existing_files]
    size_mismatches = []
    for fname, exp_sz in all_expected.items():
        if fname in existing_files and existing_files[fname] != exp_sz:
            size_mismatches.append({"file": fname, "expected_size": exp_sz, "actual_size": existing_files[fname]})

    # Full readability verification on ALL extracted images (no partial sampling)
    print(f"  Testing full image readability across ALL {total_existing:,} extracted images...")
    sample_read_ok = 0
    sample_read_err = 0
    start_v = time.time()
    for idx, fname in enumerate(existing_files.keys(), 1):
        fpath = images_dir / fname
        try:
            with Image.open(fpath) as img:
                img.verify()
            sample_read_ok += 1
        except Exception as exc:
            sample_read_err += 1
            corrupted_files.append({"file": fname, "error": str(exc)})
        if idx % 1000 == 0 or idx == total_existing:
            print(f"    Verified {idx:,}/{total_existing:,} images (Readable: {sample_read_ok:,}, Corrupt: {len(corrupted_files)})")
    elapsed_v = time.time() - start_v
    print(f"  Full readability verification finished in {elapsed_v:.1f}s.")

    status = "PASS" if total_existing > 0 and len(corrupted_files) == 0 and len(zero_byte_files) == 0 else "WARNING"

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "images_directory": str(images_dir.resolve()),
        "total_extracted_images": total_existing,
        "manifest_0_count": len(manifest_0),
        "manifest_1_count": len(manifest_1),
        "total_expected_images": total_expected,
        "missing_count": len(missing_from_manifests),
        "missing_sample": missing_from_manifests[:10],
        "size_mismatch_count": len(size_mismatches),
        "zero_byte_count": len(zero_byte_files),
        "duplicate_count": len(duplicates),
        "corrupt_count": len(corrupted_files),
        "sample_verified_count": sample_read_ok,
        "sample_verified_errors": sample_read_err,
        "status": status,
    }

    print("=" * 70)
    print("  PadChest Extraction & Image Integrity Summary")
    print("=" * 70)
    print(f"  Extracted Image Count:       {total_existing:,}")
    print(f"  Expected (Manifest 0 & 1):   {total_expected:,}")
    print(f"  Missing Images:              {len(missing_from_manifests):,}")
    print(f"  Size Mismatches:             {len(size_mismatches):,}")
    print(f"  Zero-Byte / Truncated Files: {len(zero_byte_files):,}")
    print(f"  Duplicates Detected:         {len(duplicates):,}")
    print(f"  Full Readability Verified:   {sample_read_ok}/{total_existing} OK")
    print(f"  Verification Status:         {report['status']}")
    print("=" * 70 + "\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PadChest Archives Extractor & Verifier")
    parser.add_argument("--repo_root", type=str, default=".", help="Repository root path.")
    parser.add_argument("--audit_only", action="store_true", help="Audit 0.zip and 1.zip against manifests without extracting.")
    parser.add_argument("--extract", action="store_true", help="Extract archives if present.")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing images without extraction.")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    archives_dir = root / "data" / "raw" / "PadChest" / "archives"
    manifests_dir = root / "data" / "raw" / "PadChest" / "archive_manifests"
    images_dir = root / "data" / "raw" / "PadChest" / "images"

    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_0_path = manifests_dir / "0.zip.unzip-l.txt"
    manifest_1_path = manifests_dir / "1.zip.unzip-l.txt"
    archive_0_path = archives_dir / "0.zip"
    archive_1_path = archives_dir / "1.zip"

    manifest_0 = parse_manifest(manifest_0_path)
    manifest_1 = parse_manifest(manifest_1_path)

    print(f"Loaded manifests: 0.zip ({len(manifest_0):,} files), 1.zip ({len(manifest_1):,} files)")

    # 1. Audit archives separately
    audit_0 = audit_archive_against_manifest(archive_0_path, manifest_0_path)
    audit_1 = audit_archive_against_manifest(archive_1_path, manifest_1_path)

    archive_audit = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive_0": audit_0,
        "archive_1": audit_1,
        "both_archives_verified": (audit_0["status"] == "PASSED" and audit_1["status"] == "PASSED"),
    }
    audit_path = root / "data" / "raw" / "PadChest" / "archive_audit_report.json"
    audit_path.write_text(json.dumps(archive_audit, indent=2), encoding="utf-8")
    print(f"Saved archive audit report to: {audit_path}")

    if args.audit_only:
        return 0 if archive_audit["both_archives_verified"] else 1

    # 2. Extract archives if requested
    if args.extract:
        for archive_path, manifest in [(archive_0_path, manifest_0), (archive_1_path, manifest_1)]:
            if archive_path.is_file():
                extract_archive(archive_path, images_dir, expected_manifest=manifest)
            else:
                print(f"[NOTE] Archive {archive_path.name} not found in {archives_dir}; skipping extraction.")

    # 3. Verify extracted images in destination
    report = verify_extracted_images(images_dir, manifest_0, manifest_1)
    report["archive_audit"] = archive_audit

    report_path = root / "data" / "raw" / "PadChest" / "image_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved verification audit report to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
