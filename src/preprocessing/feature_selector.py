"""Feature selection fitted on the training split only.

Three complementary filters, all optional and all configurable:

1. **Variance** - drops columns that are constant in the training split.  In
   CIC-IDS2017 this removes the URG/PSH direction counters that never fire.
2. **Correlation** - of any pair of features with ``|r| >= threshold`` the one
   with the *lower* variance is dropped.  CICFlowMeter emits several exact
   aliases (``avg_fwd_segment_size`` == ``fwd_packet_length_mean``,
   ``subflow_fwd_bytes`` ~ ``total_length_of_fwd_packets``) that otherwise
   split the importance of a single underlying signal across two columns and
   make SHAP explanations harder to read.
3. **Importance** - optional top-K filter using a small Random Forest.

Every decision is recorded so that the dashboard can show *why* a feature was
removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SelectionReport:
    """Audit trail for the feature-selection stage."""

    kept: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    importances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_kept": len(self.kept),
            "n_dropped": len(self.dropped),
            "kept": self.kept,
            "dropped": self.dropped,
            "importances": self.importances,
        }


class FeatureSelector:
    """Learns which numeric columns to keep. Fit on train, apply everywhere."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        variance_threshold: float = 0.0,
        correlation_threshold: float | None = 0.98,
        importance_top_k: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.enabled = bool(enabled)
        self.variance_threshold = float(variance_threshold)
        self.correlation_threshold = correlation_threshold
        self.importance_top_k = importance_top_k
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.report = SelectionReport()
        self.selected_: list[str] = []
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: Sequence | None = None) -> "FeatureSelector":
        columns = [str(c) for c in X.columns]
        self.report = SelectionReport()

        if not self.enabled:
            self.selected_ = columns
            self.report.kept = columns
            self.fitted_ = True
            logger.info("Feature selection disabled: keeping all %d column(s)", len(columns))
            return self

        working = X.copy()
        working = working.replace([np.inf, -np.inf], np.nan)
        medians = working.median(numeric_only=True)
        working = working.fillna(medians).fillna(0.0)

        candidates = list(columns)

        # ------------------------------------------------------- 1. variance
        variances = working[candidates].var(ddof=0)
        for column in list(candidates):
            variance = float(variances.get(column, 0.0))
            if not np.isfinite(variance) or variance <= self.variance_threshold:
                self.report.dropped[column] = (
                    f"constant/near-constant in training data (var={variance:.3g})"
                )
                candidates.remove(column)
        if not candidates:  # pathological, but never crash
            logger.warning("Variance filter removed every column; restoring all.")
            candidates = list(columns)
            self.report.dropped.clear()

        # ---------------------------------------------------- 2. correlation
        if self.correlation_threshold is not None and len(candidates) > 1:
            threshold = float(self.correlation_threshold)
            corr = working[candidates].corr().abs()
            variances = working[candidates].var(ddof=0)
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            for column in list(upper.columns):
                if column not in candidates:
                    continue
                partners = upper.index[upper[column] >= threshold].tolist()
                for partner in partners:
                    if partner not in candidates or column not in candidates:
                        continue
                    # Drop whichever of the pair carries less variance.
                    weaker, stronger = (
                        (column, partner)
                        if variances[column] <= variances[partner]
                        else (partner, column)
                    )
                    candidates.remove(weaker)
                    self.report.dropped[weaker] = (
                        f"|r| >= {threshold} with '{stronger}' (redundant)"
                    )

        # ----------------------------------------------------- 3. importance
        if self.importance_top_k and y is not None and len(candidates) > int(self.importance_top_k):
            try:
                from sklearn.ensemble import RandomForestClassifier

                forest = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=16,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    class_weight="balanced_subsample",
                )
                sample = working[candidates]
                labels = pd.Series(y).reset_index(drop=True)
                if len(sample) > 100_000:
                    idx = np.random.default_rng(self.random_state).choice(
                        len(sample), size=100_000, replace=False
                    )
                    sample = sample.iloc[idx]
                    labels = labels.iloc[idx]
                forest.fit(sample, labels)
                ranking = pd.Series(forest.feature_importances_, index=candidates)
                ranking = ranking.sort_values(ascending=False)
                self.report.importances = {
                    str(k): float(v) for k, v in ranking.items()
                }
                keep = set(ranking.head(int(self.importance_top_k)).index)
                for column in list(candidates):
                    if column not in keep:
                        candidates.remove(column)
                        self.report.dropped[column] = (
                            f"outside top-{self.importance_top_k} by RF importance"
                        )
            except Exception as exc:  # pragma: no cover - optional stage
                logger.warning("Importance-based selection skipped: %s", exc)

        self.selected_ = [c for c in columns if c in set(candidates)]
        self.report.kept = list(self.selected_)
        self.fitted_ = True
        logger.info(
            "Feature selection kept %d of %d numeric column(s)",
            len(self.selected_), len(columns),
        )
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeatureSelector.transform called before fit.")
        missing = [c for c in self.selected_ if c not in X.columns]
        if missing:
            raise KeyError(f"Selected feature(s) missing from input: {missing}")
        return X[self.selected_]

    def fit_transform(self, X: pd.DataFrame, y: Sequence | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<FeatureSelector enabled={self.enabled} "
            f"kept={len(self.selected_)} dropped={len(self.report.dropped)}>"
        )


__all__ = ["FeatureSelector", "SelectionReport"]
