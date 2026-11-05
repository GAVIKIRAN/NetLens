"""The reproducible preprocessing pipeline (Phase 2).

One fitted :class:`PreprocessingPipeline` object is persisted alongside the
model.  It carries, in a single artefact:

* ``input_features``  - the exact ordered list of raw feature names the caller
  must supply (this is the contract the live flow meter honours);
* the fitted feature selector;
* the fitted ``ColumnTransformer`` (winsorise -> impute -> scale for numeric
  columns, one-hot for categorical columns);
* ``output_features`` - the transformed column names used by SHAP and LIME.

Because training and live inference both call :meth:`transform`, the feature
order can never diverge: :meth:`transform` re-indexes its input to
``input_features`` and raises if anything is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.features.contract import ResolvedFeatures, align_frame
from src.preprocessing.encoder import (
    PortBucketTransformer,
    build_port_encoder,
    build_protocol_encoder,
)
from src.preprocessing.feature_selector import FeatureSelector
from src.preprocessing.scaler import QuantileClipper, build_scaler
from src.utils.errors import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PIPELINE_FILENAME = "preprocessing_pipeline.joblib"
FEATURES_FILENAME = "feature_contract.json"


@dataclass
class PreprocessingConfig:
    """Configuration slice consumed by :class:`PreprocessingPipeline`."""

    numeric_imputation: str = "median"
    imputation_fill_value: float = 0.0
    scaler: str = "standard"
    clip_quantile: float | None = 0.999
    selection_enabled: bool = True
    variance_threshold: float = 0.0
    correlation_threshold: float | None = 0.98
    importance_top_k: int | None = None
    random_state: int = 42
    n_jobs: int = -1

    @classmethod
    def from_settings(cls, settings) -> "PreprocessingConfig":
        prep = settings.section("preprocessing")
        selection = prep.get("feature_selection", {}) or {}
        return cls(
            numeric_imputation=str(prep.get("numeric_imputation", "median")),
            imputation_fill_value=float(prep.get("imputation_fill_value", 0.0)),
            scaler=str(prep.get("scaler", "standard")),
            clip_quantile=prep.get("clip_quantile", 0.999),
            selection_enabled=bool(selection.get("enabled", True)),
            variance_threshold=float(selection.get("variance_threshold", 0.0)),
            correlation_threshold=selection.get("correlation_threshold", 0.98),
            importance_top_k=selection.get("importance_top_k"),
            random_state=int(settings.get("training.random_state", 42)),
            n_jobs=int(settings.get("training.n_jobs", -1)),
        )


@dataclass
class PipelineMetadata:
    """Everything needed to reason about a fitted pipeline after the fact."""

    input_features: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    selected_numeric: list[str] = field(default_factory=list)
    output_features: list[str] = field(default_factory=list)
    selection_report: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_features": self.input_features,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "selected_numeric": self.selected_numeric,
            "output_features": self.output_features,
            "n_input_features": len(self.input_features),
            "n_output_features": len(self.output_features),
            "selection_report": self.selection_report,
            "config": self.config,
        }


class PreprocessingPipeline:
    """Fit-once, apply-everywhere preprocessing."""

    def __init__(
        self, resolved: ResolvedFeatures, config: PreprocessingConfig | None = None
    ) -> None:
        self.resolved = resolved
        self.config = config or PreprocessingConfig()
        self.input_features: list[str] = list(resolved.all)
        self.numeric_features: list[str] = list(resolved.numeric)
        self.categorical_features: list[str] = list(resolved.categorical)
        self.selector = FeatureSelector(
            enabled=self.config.selection_enabled,
            variance_threshold=self.config.variance_threshold,
            correlation_threshold=self.config.correlation_threshold,
            importance_top_k=self.config.importance_top_k,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.transformer: ColumnTransformer | None = None
        self.output_features: list[str] = []
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: Sequence | None = None) -> "PreprocessingPipeline":
        frame = self._prepare_input(X, strict=False)

        numeric_frame = frame[self.numeric_features] if self.numeric_features else pd.DataFrame(index=frame.index)
        if self.numeric_features:
            self.selector.fit(numeric_frame, y)
            selected_numeric = list(self.selector.selected_)
        else:
            selected_numeric = []

        if not selected_numeric and not self.categorical_features:
            raise PreprocessingError(
                "Feature selection removed every input column.",
                hint="Relax preprocessing.feature_selection in config.yaml.",
            )

        self.transformer = self._build_transformer(selected_numeric)
        try:
            self.transformer.fit(frame, y)
        except Exception as exc:
            raise PreprocessingError(
                f"Could not fit the preprocessing pipeline: {exc}"
            ) from exc

        self.output_features = [
            str(name) for name in self.transformer.get_feature_names_out()
        ]
        self.fitted_ = True
        logger.info(
            "Preprocessing fitted: %d input feature(s) -> %d transformed feature(s)",
            len(self.input_features), len(self.output_features),
        )
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame | Sequence[dict]) -> np.ndarray:
        """Apply the fitted transformations. Column order is enforced here."""
        self._require_fitted()
        frame = self._prepare_input(X, strict=True)
        try:
            values = self.transformer.transform(frame)
        except Exception as exc:
            raise PreprocessingError(
                f"Could not apply the preprocessing pipeline: {exc}"
            ) from exc
        array = np.asarray(values, dtype=np.float64)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X: pd.DataFrame, y: Sequence | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def transform_to_frame(self, X: pd.DataFrame | Sequence[dict]) -> pd.DataFrame:
        """Same as :meth:`transform` but returns a named ``DataFrame``."""
        values = self.transform(X)
        return pd.DataFrame(values, columns=self.output_features)

    # -------------------------------------------------------------- helpers
    def _prepare_input(self, X, *, strict: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        elif isinstance(X, dict):
            frame = pd.DataFrame([X])
        else:
            frame = pd.DataFrame(list(X))
        frame = align_frame(frame, self.input_features, strict=strict, fill_value=0.0)
        for column in self.numeric_features:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.categorical_features:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
                frame[column] = frame[column].astype(np.int64)
        return frame

    def _build_transformer(self, selected_numeric: list[str]) -> ColumnTransformer:
        blocks: list[tuple[str, Any, list[str]]] = []

        if selected_numeric:
            steps: list[tuple[str, Any]] = [
                ("clip", QuantileClipper(self.config.clip_quantile)),
                ("impute", self._build_imputer()),
            ]
            scaler = build_scaler(self.config.scaler)
            if scaler is not None:
                steps.append(("scale", scaler))
            blocks.append(("num", Pipeline(steps), list(selected_numeric)))

        for column in self.categorical_features:
            if column == "protocol":
                blocks.append(
                    (
                        "proto",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="most_frequent")),
                                ("onehot", build_protocol_encoder()),
                            ]
                        ),
                        [column],
                    )
                )
            elif column == "destination_port":
                blocks.append(
                    (
                        "dport",
                        Pipeline(
                            [
                                (
                                    "impute",
                                    SimpleImputer(strategy="constant", fill_value=-1),
                                ),
                                ("bucket", PortBucketTransformer()),
                                ("onehot", build_port_encoder()),
                            ]
                        ),
                        [column],
                    )
                )

        return ColumnTransformer(
            transformers=blocks,
            remainder="drop",
            verbose_feature_names_out=True,
            n_jobs=None,
        )

    def _build_imputer(self) -> SimpleImputer:
        strategy = str(self.config.numeric_imputation).lower()
        if strategy == "constant":
            return SimpleImputer(
                strategy="constant", fill_value=float(self.config.imputation_fill_value)
            )
        if strategy not in {"median", "mean", "most_frequent"}:
            raise PreprocessingError(
                f"Unknown preprocessing.numeric_imputation '{strategy}'.",
                hint="Use median, mean or constant.",
            )
        return SimpleImputer(strategy=strategy)

    def _require_fitted(self) -> None:
        if not self.fitted_ or self.transformer is None:
            raise PreprocessingError(
                "The preprocessing pipeline has not been fitted.",
                hint="Run `python scripts/train_model.py` first.",
            )

    # ------------------------------------------------------------- metadata
    def metadata(self) -> PipelineMetadata:
        return PipelineMetadata(
            input_features=list(self.input_features),
            numeric_features=list(self.numeric_features),
            categorical_features=list(self.categorical_features),
            selected_numeric=list(self.selector.selected_),
            output_features=list(self.output_features),
            selection_report=self.selector.report.to_dict(),
            config=vars(self.config).copy(),
        )

    # ---------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        """Persist the fitted pipeline and its human-readable contract."""
        self._require_fitted()
        from src.utils.helpers import ensure_dir, write_json

        target = ensure_dir(directory)
        path = target / PIPELINE_FILENAME
        joblib.dump(self, path)
        write_json(target / FEATURES_FILENAME, self.metadata().to_dict())
        logger.info("Preprocessing pipeline saved to %s", path)
        return path

    @staticmethod
    def load(directory: str | Path) -> "PreprocessingPipeline":
        path = Path(directory)
        if path.is_dir():
            path = path / PIPELINE_FILENAME
        if not path.exists():
            raise PreprocessingError(
                f"Preprocessing pipeline not found: {path}",
                hint="Run `python scripts/train_model.py` to create it.",
            )
        try:
            obj = joblib.load(path)
        except Exception as exc:
            raise PreprocessingError(
                f"Preprocessing artefact is corrupted: {exc}",
                hint="Delete models/preprocessing and re-train.",
            ) from exc
        if not isinstance(obj, PreprocessingPipeline):
            raise PreprocessingError(f"{path} is not a PreprocessingPipeline.")
        return obj

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<PreprocessingPipeline fitted={self.fitted_} "
            f"in={len(self.input_features)} out={len(self.output_features)}>"
        )


__all__ = [
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "PipelineMetadata",
    "PIPELINE_FILENAME",
    "FEATURES_FILENAME",
]
