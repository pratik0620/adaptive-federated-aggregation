"""Dataset-specific label and schema helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing_utils import ensure_standard_columns, first_existing_column, normalize_label, parse_label_list, parse_pixel_spacing


def nih_clean_binary_labels(frame: pd.DataFrame, target_disease: str) -> tuple[pd.DataFrame, dict]:
    labels = frame["Finding Labels"].map(parse_label_list)
    target = normalize_label(target_disease)
    normalized = labels.map(lambda values: {normalize_label(value) for value in values})
    positive = normalized.map(lambda values: target in values)
    negative = normalized.map(lambda values: values == {"no finding"})
    filtered = frame[positive | negative].copy()
    filtered["target"] = np.where(positive[filtered.index], 1, 0)
    return filtered, {
        "positive_before_filter": int(positive.sum()),
        "negative_before_filter": int(negative.sum()),
        "excluded_ambiguous_before_filter": int((~(positive | negative)).sum()),
    }


def standardize_nih(frame: pd.DataFrame, image_dir) -> pd.DataFrame:
    spacing = frame.get("OriginalImagePixelSpacing", pd.Series(np.nan, index=frame.index)).map(parse_pixel_spacing)
    result = pd.DataFrame(index=frame.index)
    result["image_id"] = frame["Image Index"]
    result["patient_id"] = frame["Patient ID"]
    result["study_id"] = np.nan
    result["source"] = "NIH"
    result["target"] = frame["target"]
    result["age"] = pd.to_numeric(frame.get("Patient Age", np.nan), errors="coerce")
    result["sex"] = frame.get("Patient Gender", np.nan)
    result["view_position"] = frame.get("View Position", np.nan)
    result["projection"] = frame.get("View Position", np.nan)
    result["image_width"] = pd.to_numeric(frame.get("OriginalImage Width", np.nan), errors="coerce")
    result["image_height"] = pd.to_numeric(frame.get("OriginalImage Height", np.nan), errors="coerce")
    result["pixel_spacing_x"] = spacing.map(lambda value: value[0])
    result["pixel_spacing_y"] = spacing.map(lambda value: value[1])
    result["scanner_manufacturer"] = np.nan
    result["image_path"] = result["image_id"].map(lambda value: str(image_dir / str(value)))
    result["finding_labels"] = frame["Finding Labels"]
    if "Follow-up #" in frame.columns:
        result["follow_up"] = frame["Follow-up #"]
    return ensure_standard_columns(result)


def chexpert_clean_binary_labels(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, dict]:
    values = pd.to_numeric(frame[target_column], errors="coerce")
    positive = values == 1
    negative = values == 0
    uncertain = values == -1
    missing = values.isna()
    filtered = frame[positive | negative].copy()
    filtered["target"] = values[filtered.index].astype(int)
    return filtered, {
        "positive_before_filter": int(positive.sum()),
        "negative_before_filter": int(negative.sum()),
        "uncertain_before_filter": int(uncertain.sum()),
        "missing_not_mentioned_before_filter": int(missing.sum()),
    }


def chexpert_patient_id(path_series: pd.Series) -> pd.Series:
    return path_series.astype(str).str.extract(r"(patient\d+)", expand=False)


def standardize_chexpert(frame: pd.DataFrame, image_root) -> pd.DataFrame:
    paths = frame["Path"].astype(str)
    result = pd.DataFrame(index=frame.index)
    result["image_id"] = paths
    result["patient_id"] = chexpert_patient_id(paths)
    result["study_id"] = paths.str.extract(r"(study\d+)", expand=False)
    result["source"] = "CheXpert"
    result["target"] = frame["target"]
    result["age"] = pd.to_numeric(frame.get("Age", np.nan), errors="coerce")
    result["sex"] = frame.get("Sex", np.nan)
    result["view_position"] = frame.get("Frontal/Lateral", np.nan)
    result["projection"] = frame.get("AP/PA", np.nan)
    result["image_width"] = np.nan
    result["image_height"] = np.nan
    result["pixel_spacing_x"] = np.nan
    result["pixel_spacing_y"] = np.nan
    result["scanner_manufacturer"] = np.nan
    result["image_path"] = paths.map(lambda value: str(image_root / value))
    result["path_original"] = paths
    return ensure_standard_columns(result)


def padchest_find_label_vocabulary(frame: pd.DataFrame, labels_column: str = "Labels") -> list[str]:
    vocabulary = {label for values in frame[labels_column].map(parse_label_list) for label in values}
    return sorted(vocabulary, key=normalize_label)


def padchest_clean_binary_labels(
    frame: pd.DataFrame, target_label: str, labels_column: str = "Labels", negative_label: str = "no finding"
) -> tuple[pd.DataFrame, dict]:
    labels = frame[labels_column].map(parse_label_list)
    normalized_target = normalize_label(target_label)
    normalized_negative = normalize_label(negative_label)
    normalized = labels.map(lambda values: {normalize_label(value) for value in values})
    positive = normalized.map(lambda values: normalized_target in values)
    negative = normalized.map(lambda values: values == {normalized_negative})
    filtered = frame[positive | negative].copy()
    filtered["target"] = np.where(positive[filtered.index], 1, 0)
    return filtered, {
        "positive_before_filter": int(positive.sum()),
        "negative_before_filter": int(negative.sum()),
        "excluded_ambiguous_before_filter": int((~(positive | negative)).sum()),
    }


def standardize_padchest(frame: pd.DataFrame, image_dir) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["image_id"] = frame["ImageID"]
    result["patient_id"] = frame.get("PatientID", np.nan)
    result["study_id"] = frame.get("StudyID", np.nan)
    result["source"] = "PadChest"
    result["target"] = frame["target"]
    result["age"] = pd.to_numeric(first_existing_column(frame, ["PatientAge", "PatientAge_DICOM"]), errors="coerce")
    result["sex"] = first_existing_column(frame, ["PatientSex_DICOM", "PatientSex"])
    result["view_position"] = first_existing_column(frame, ["ViewPosition_DICOM", "ViewPosition"])
    result["projection"] = frame.get("Projection", np.nan)
    result["image_width"] = pd.to_numeric(first_existing_column(frame, ["Columns_DICOM", "Columns"]), errors="coerce")
    result["image_height"] = pd.to_numeric(first_existing_column(frame, ["Rows_DICOM", "Rows"]), errors="coerce")
    spacing = first_existing_column(frame, ["PixelSpacing_DICOM", "PixelSpacing"]).map(parse_pixel_spacing)
    result["pixel_spacing_x"] = spacing.map(lambda value: value[0])
    result["pixel_spacing_y"] = spacing.map(lambda value: value[1])
    result["scanner_manufacturer"] = first_existing_column(frame, ["Manufacturer_DICOM", "Manufacturer"])
    result["image_path"] = result["image_id"].map(lambda value: str(image_dir / str(value)))
    for column in ["Labels", "ReportID", "Modality_DICOM"]:
        if column in frame.columns:
            result[column.casefold()] = frame[column]
    return ensure_standard_columns(result)
