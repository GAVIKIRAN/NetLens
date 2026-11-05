"""Dataset profiling (Phase 1 reporting).

Produces the statistics the project report needs — record counts, class
distribution, missing/duplicate reports and per-feature descriptive statistics
— computed from the data actually loaded.  Nothing here is hard-coded or
estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dir, utc_now_iso, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetProfile:
    """Structured dataset statistics."""

    generated_at: str = ""
    n_records: int = 0
    n_features: int = 0
    memory_mb: float = 0.0
    class_distribution: dict[str, int] = field(default_factory=dict)
    class_distribution_pct: dict[str, float] = field(default_factory=dict)
    imbalance_ratio: float = 0.0
    missing_report: dict[str, int] = field(default_factory=dict)
    missing_pct: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    constant_columns: list[str] = field(default_factory=list)
    feature_statistics: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "n_records": self.n_records,
            "n_features": self.n_features,
            "memory_mb": round(self.memory_mb, 2),
            "class_distribution": self.class_distribution,
            "class_distribution_pct": self.class_distribution_pct,
            "imbalance_ratio": self.imbalance_ratio,
            "duplicate_rows": self.duplicate_rows,
            "constant_columns": self.constant_columns,
            "missing_report": self.missing_report,
            "missing_pct": self.missing_pct,
            "feature_statistics": self.feature_statistics,
            **self.extra,
        }

    def feature_statistics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.feature_statistics)

    def class_distribution_frame(self) -> pd.DataFrame:
        rows = [
            {
                "label": label,
                "count": count,
                "percent": self.class_distribution_pct.get(label, 0.0),
            }
            for label, count in self.class_distribution.items()
        ]
        return pd.DataFrame(rows).sort_values("count", ascending=False)


def profile_dataset(
    frame: pd.DataFrame,
    label_column: str,
    *,
    extra: dict[str, Any] | None = None,
    max_features_detailed: int = 200,
) -> DatasetProfile:
    """Compute descriptive statistics for a (cleaned or raw) dataset."""
    profile = DatasetProfile(generated_at=utc_now_iso())
    feature_cols = [c for c in frame.columns if c != label_column]

    profile.n_records = int(len(frame))
    profile.n_features = len(feature_cols)
    profile.memory_mb = float(frame.memory_usage(deep=True).sum()) / (1024 * 1024)

    if label_column in frame.columns:
        counts = frame[label_column].value_counts()
        profile.class_distribution = {str(k): int(v) for k, v in counts.items()}
        total = max(1, int(counts.sum()))
        profile.class_distribution_pct = {
            str(k): round(100.0 * int(v) / total, 4) for k, v in counts.items()
        }
        if len(counts) > 1 and int(counts.min()) > 0:
            profile.imbalance_ratio = round(float(counts.max()) / float(counts.min()), 2)

    missing = frame[feature_cols].isna().sum()
    profile.missing_report = {
        str(k): int(v) for k, v in missing.items() if int(v) > 0
    }
    profile.missing_pct = {
        k: round(100.0 * v / max(1, profile.n_records), 4)
        for k, v in profile.missing_report.items()
    }

    profile.duplicate_rows = int(frame.duplicated().sum())

    numeric = frame[feature_cols].select_dtypes(include=[np.number])
    if not numeric.empty:
        nunique = numeric.nunique(dropna=True)
        profile.constant_columns = [str(c) for c in nunique[nunique <= 1].index]

        described = numeric.describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T
        described = described.head(max_features_detailed)
        stats: list[dict[str, Any]] = []
        for name, row in described.iterrows():
            stats.append(
                {
                    "feature": str(name),
                    "count": int(row.get("count", 0)),
                    "mean": _clean(row.get("mean")),
                    "std": _clean(row.get("std")),
                    "min": _clean(row.get("min")),
                    "p25": _clean(row.get("25%")),
                    "median": _clean(row.get("50%")),
                    "p75": _clean(row.get("75%")),
                    "p99": _clean(row.get("99%")),
                    "max": _clean(row.get("max")),
                    "missing": int(missing.get(name, 0)),
                    "unique": int(nunique.get(name, 0)),
                }
            )
        profile.feature_statistics = stats

    if extra:
        profile.extra.update(extra)

    logger.info(
        "Profiled dataset: %s records, %d features, %d classes, %s duplicate row(s)",
        f"{profile.n_records:,}", profile.n_features,
        len(profile.class_distribution), f"{profile.duplicate_rows:,}",
    )
    return profile


def _clean(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, 6)


def save_profile(profile: DatasetProfile, reports_dir: str | Path) -> Path:
    """Persist the profile as ``reports/dataset_profile.json``."""
    directory = ensure_dir(reports_dir)
    path = directory / "dataset_profile.json"
    write_json(path, profile.to_dict())
    logger.info("Dataset profile written to %s", path)
    return path


def load_profile(reports_dir: str | Path) -> dict[str, Any] | None:
    """Load a previously saved profile, or ``None`` when it does not exist."""
    from src.utils.helpers import read_json

    return read_json(Path(reports_dir) / "dataset_profile.json")


__all__ = ["DatasetProfile", "profile_dataset", "save_profile", "load_profile"]
