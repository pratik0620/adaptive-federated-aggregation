"""Shared configuration defaults for the federated metadata pipelines."""

from pathlib import Path

TARGET_DISEASE = "Pleural Effusion"
RANDOM_SEED = 42
MAX_POSITIVE_SAMPLES = None
MAX_NEGATIVE_SAMPLES = None

STANDARD_COLUMNS = [
    "image_id",
    "patient_id",
    "study_id",
    "source",
    "target",
    "age",
    "sex",
    "view_position",
    "projection",
    "image_width",
    "image_height",
    "pixel_spacing_x",
    "pixel_spacing_y",
    "scanner_manufacturer",
    "image_path",
    "finding_labels",
    "follow_up",
]


def default_project_root() -> Path:
    """Return the repository root without assuming a machine-specific path."""
    return Path.cwd()
