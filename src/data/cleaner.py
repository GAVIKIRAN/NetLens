"""Dataset-level cleaning (Phase 1).

Only operations that are *independent of any learned statistic* happen here, so
that nothing leaks from the test split into training.  Anything that requires
fitting (imputation values, scaling parameters, variance thresholds) lives in
:mod:`src.preprocessing.pipeline` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.contract import EXCLUDED, LIVE_FEATURES
from src.utils.errors import DatasetError
from src.utils.helpers import normalise_label
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CleanReport:
    """What the cleaner actually did — surfaced in the dashboard and the log."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    rows_with_null_label: int = 0
    rows_all_nan: int = 0
    infinite_values_replaced: int = 0
    missing_values_remaining: int = 0
    columns_dropped: dict[str, str] = field(default_factory=dict)
    negative_durations: int = 0
    merged_rare_classes: dict[str, int] = field(default_factory=dict)
    label_renames: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed": self.rows_in - self.rows_out,
            "duplicates_removed": self.duplicates_removed,
            "rows_with_null_label": self.rows_with_null_label,
            "rows_all_nan": self.rows_all_nan,
            "infinite_values_replaced": self.infinite_values_replaced,
            "missing_values_remaining": self.missing_values_remaining,
            "negative_durations": self.negative_durations,
            "columns_dropped": self.columns_dropped,
            "merged_rare_classes": self.merged_rare_classes,
            "label_renames": self.label_renames,
        }


def clean_dataset(
    frame: pd.DataFrame,
    label_column: str,
    *,
    drop_duplicates: bool = True,
    min_samples_per_class: int = 50,
    rare_class_label: str = "Rare_Attack",
    benign_label: str = "BENIGN",
) -> tuple[pd.DataFrame, CleanReport]:
    """Clean a raw CIC-IDS2017 frame.

    Steps, in order:

    1. Drop identifier / degenerate columns listed in the feature contract.
    2. Canonicalise class labels (fixes the CP-1252 dash in the web-attack
       labels, collapses whitespace and normalises ``BENIGN``).
    3. Drop rows whose label is missing.
    4. Replace ``+/-inf`` (produced by CICFlowMeter's rate columns when the flow
       duration is zero) with ``NaN`` so the pipeline's imputer handles them.
    5. Drop rows in which *every* feature is ``NaN``.
    6. Drop exact duplicate rows.
    7. Merge classes with fewer than ``min_samples_per_class`` samples into a
       single ``Rare_Attack`` class - they cannot be trained or scored reliably.
    """
    if label_column not in frame.columns:
        raise DatasetError(f"Label column '{label_column}' is not in the frame.")

    report = CleanReport(rows_in=int(len(frame)))
    frame = frame.copy()

    # ------------------------------------------------------ 1. drop columns
    for column in list(frame.columns):
        if column == label_column:
            continue
        if column in EXCLUDED:
            report.columns_dropped[column] = EXCLUDED[column]
            frame = frame.drop(columns=[column])
        elif column not in LIVE_FEATURES:
            report.columns_dropped[column] = (
                "not part of the live-derivable feature contract"
            )
            frame = frame.drop(columns=[column])
    if report.columns_dropped:
        logger.info("Dropped %d non-contract column(s)", len(report.columns_dropped))

    # ------------------------------------------------------- 2. fix labels
    original = frame[label_column].astype(str)
    normalised = original.map(normalise_label)
    normalised = normalised.map(
        lambda x: benign_label if x.strip().upper() == benign_label else x
    )
    changed = original != normalised
    if bool(changed.any()):
        pairs = (
            pd.DataFrame({"before": original[changed], "after": normalised[changed]})
            .drop_duplicates()
            .head(20)
        )
        report.label_renames = dict(zip(pairs["before"], pairs["after"]))
    frame[label_column] = normalised

    # ------------------------------------------------- 3. drop null labels
    null_mask = frame[label_column].isin(["", "nan", "None", "NaN"]) | frame[
        label_column
    ].isna()
    report.rows_with_null_label = int(null_mask.sum())
    if report.rows_with_null_label:
        frame = frame.loc[~null_mask]

    feature_cols = [c for c in frame.columns if c != label_column]
    if not feature_cols:
        raise DatasetError(
            "No contract features survived cleaning.",
            hint="The CSV headers do not match CIC-IDS2017.",
        )

    # ------------------------------------------------------- 4. infinities
    numeric = frame[feature_cols].select_dtypes(include=[np.number])
    if not numeric.empty:
        inf_mask = np.isinf(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
        report.infinite_values_replaced = int(inf_mask.sum())
        if report.infinite_values_replaced:
            frame[numeric.columns] = numeric.replace([np.inf, -np.inf], np.nan)
            logger.info(
                "Replaced %s infinite value(s) with NaN",
                f"{report.infinite_values_replaced:,}",
            )

    # ------------------------------------------------------ 5. empty rows
    all_nan = frame[feature_cols].isna().all(axis=1)
    report.rows_all_nan = int(all_nan.sum())
    if report.rows_all_nan:
        frame = frame.loc[~all_nan]

    # ------------------------------------------------------- 6. duplicates
    if drop_duplicates:
        before = len(frame)
        frame = frame.drop_duplicates()
        report.duplicates_removed = int(before - len(frame))
        if report.duplicates_removed:
            logger.info("Removed %s duplicate row(s)", f"{report.duplicates_removed:,}")

    # --------------------------------------------- diagnostics (no changes)
    if "flow_duration" in frame.columns:
        report.negative_durations = int((frame["flow_duration"] < 0).sum())
        if report.negative_durations:
            logger.warning(
                "%d row(s) have a negative flow_duration (a known CICFlowMeter "
                "artefact); they are kept and winsorised by the pipeline.",
                report.negative_durations,
            )

    # ---------------------------------------------------- 7. rare classes
    counts = frame[label_column].value_counts()
    rare = counts[counts < int(min_samples_per_class)]
    if len(rare) > 0:
        report.merged_rare_classes = {str(k): int(v) for k, v in rare.items()}
        frame[label_column] = frame[label_column].where(
            ~frame[label_column].isin(rare.index), rare_class_label
        )
        logger.info(
            "Merged %d rare class(es) into '%s': %s",
            len(rare), rare_class_label, ", ".join(map(str, rare.index)),
        )

    report.missing_values_remaining = int(frame[feature_cols].isna().sum().sum())
    frame = frame.reset_index(drop=True)
    report.rows_out = int(len(frame))

    if report.rows_out == 0:
        raise DatasetError("Cleaning removed every row.")
    if frame[label_column].nunique() < 2:
        raise DatasetError(
            "Only one class survived cleaning; a classifier cannot be trained.",
            hint="Load more CIC-IDS2017 files (Monday is BENIGN-only).",
        )

    logger.info(
        "Cleaning complete: %s -> %s rows, %d feature column(s), %d class(es)",
        f"{report.rows_in:,}", f"{report.rows_out:,}",
        len(feature_cols), frame[label_column].nunique(),
    )
    return frame, report


def add_binary_label(
    frame: pd.DataFrame, label_column: str, benign_label: str = "BENIGN"
) -> pd.DataFrame:
    """Add a ``binary_label`` column (``BENIGN`` / ``ATTACK``)."""
    frame = frame.copy()
    frame["binary_label"] = np.where(
        frame[label_column].astype(str) == benign_label, benign_label, "ATTACK"
    )
    return frame


__all__ = ["CleanReport", "clean_dataset", "add_binary_label"]
