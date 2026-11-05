"""Real-time inference engine (Phase 7).

The :class:`Predictor` is the only object that ever touches the trained model.
It owns the four artefacts that must always travel together:

* the fitted preprocessing pipeline (which enforces feature order),
* the label codec (integer code <-> class name),
* the selected estimator,
* the training metadata (classes, feature policy, provenance).

Threshold policy
----------------
``argmax`` alone is a poor decision rule for an IDS: a flow whose most likely
class is ``DoS Hulk`` at 0.41 is not evidence of an attack.  So the predictor
applies ``inference.attack_probability_threshold``:

* if the top class is BENIGN, report BENIGN;
* if the top class is an attack **and** its probability is at or above the
  threshold, report that attack class;
* otherwise report BENIGN and set ``threshold_applied=True`` - the original
  argmax class and its probability are retained for the dashboard so the
  analyst can still see what the model was leaning towards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings
from src.features.contract import align_frame
from src.inference.risk import RiskAssessment, RiskScorer
from src.preprocessing.encoder import LabelCodec
from src.preprocessing.pipeline import PreprocessingPipeline
from src.training.train import (
    BACKGROUND_FILENAME,
    BEST_MODEL_FILENAME,
    LABEL_ENCODER_FILENAME,
    METADATA_FILENAME,
)
from src.utils.errors import ModelError
from src.utils.helpers import read_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """One classified flow."""

    predicted_label: str
    predicted_code: int
    is_attack: bool
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    risk: RiskAssessment | None = None
    argmax_label: str = ""
    argmax_confidence: float = 0.0
    threshold_applied: bool = False
    features: dict[str, float] = field(default_factory=dict)
    transformed: np.ndarray | None = None
    model_name: str = ""

    def top_probabilities(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def to_dict(self, include_features: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prediction": self.predicted_label,
            "predicted_code": self.predicted_code,
            "is_attack": self.is_attack,
            "confidence": round(float(self.confidence), 6),
            "argmax_label": self.argmax_label,
            "argmax_confidence": round(float(self.argmax_confidence), 6),
            "threshold_applied": self.threshold_applied,
            "model": self.model_name,
            "probabilities": {k: round(float(v), 6) for k, v in self.probabilities.items()},
        }
        if self.risk is not None:
            payload.update(self.risk.to_dict())
        if include_features:
            payload["features"] = self.features
        return payload


class Predictor:
    """Loads the trained artefacts and classifies flow-feature records."""

    def __init__(
        self,
        model,
        pipeline: PreprocessingPipeline,
        codec: LabelCodec,
        risk_scorer: RiskScorer,
        *,
        metadata: Mapping[str, Any] | None = None,
        benign_label: str = "BENIGN",
        attack_threshold: float = 0.6,
        background: np.ndarray | None = None,
        background_labels: np.ndarray | None = None,
        model_name: str = "model",
    ) -> None:
        self.model = model
        self.pipeline = pipeline
        self.codec = codec
        self.risk_scorer = risk_scorer
        self.metadata = dict(metadata or {})
        self.benign_label = str(benign_label)
        self.attack_threshold = float(attack_threshold)
        self.background = background
        self.background_labels = background_labels
        self.model_name = str(model_name)

        self.classes: list[str] = list(codec.classes_)
        self.input_features: list[str] = list(pipeline.input_features)
        self.output_features: list[str] = list(pipeline.output_features)
        self._benign_code = codec.code_of(self.benign_label)
        if self._benign_code < 0:
            logger.warning(
                "Benign label '%s' is not among the trained classes %s; "
                "every prediction will be treated as an attack.",
                self.benign_label, self.classes,
            )

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, settings: Settings | None = None) -> "Predictor":
        """Load every artefact produced by ``scripts/train_model.py``."""
        settings = settings or get_settings()
        model_dir = settings.path("model_dir")
        prep_dir = settings.path("preprocessing_dir")

        model_path = model_dir / BEST_MODEL_FILENAME
        if not model_path.exists():
            raise ModelError(
                f"Trained model not found: {model_path}",
                hint="Run `python scripts/train_model.py` first.",
            )
        try:
            model = joblib.load(model_path)
        except Exception as exc:
            raise ModelError(
                f"The model artefact could not be loaded ({exc}).",
                hint="It may be corrupted or produced by a different library "
                     "version. Delete models/trained and re-train.",
            ) from exc

        pipeline = PreprocessingPipeline.load(prep_dir)
        codec = LabelCodec.load(prep_dir / LABEL_ENCODER_FILENAME)
        metadata = read_json(model_dir / METADATA_FILENAME) or {}

        background = background_labels = None
        background_path = model_dir / BACKGROUND_FILENAME
        if background_path.exists():
            try:
                payload = np.load(background_path, allow_pickle=True)
                background = np.asarray(payload["X"], dtype=np.float64)
                background_labels = np.asarray(payload["y"])
            except Exception as exc:  # pragma: no cover - optional artefact
                logger.warning("XAI background could not be loaded: %s", exc)

        expected = len(pipeline.output_features)
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and int(n_features) != expected:
            raise ModelError(
                f"Model expects {n_features} features but the preprocessing "
                f"pipeline produces {expected}.",
                hint="The artefacts are from different training runs; re-train.",
            )

        predictor = cls(
            model=model,
            pipeline=pipeline,
            codec=codec,
            risk_scorer=RiskScorer.from_settings(settings),
            metadata=metadata,
            benign_label=str(settings.get("inference.benign_label", "BENIGN")),
            attack_threshold=float(settings.get("inference.attack_probability_threshold", 0.6)),
            background=background,
            background_labels=background_labels,
            model_name=str(metadata.get("best_model", type(model).__name__)),
        )
        logger.info(
            "Predictor ready: model='%s', %d class(es), %d input feature(s)",
            predictor.model_name, len(predictor.classes), len(predictor.input_features),
        )
        return predictor

    @staticmethod
    def artefacts_exist(settings: Settings | None = None) -> bool:
        """Cheap check used by the dashboard before attempting to load."""
        settings = settings or get_settings()
        try:
            return (
                (settings.path("model_dir") / BEST_MODEL_FILENAME).exists()
                and (settings.path("preprocessing_dir") / "preprocessing_pipeline.joblib").exists()
                and (settings.path("preprocessing_dir") / LABEL_ENCODER_FILENAME).exists()
            )
        except Exception:  # pragma: no cover - defensive
            return False

    # ---------------------------------------------------------- transform
    def transform(self, records: Sequence[Mapping[str, float]] | pd.DataFrame) -> np.ndarray:
        """Apply the persisted preprocessing to raw flow records."""
        frame = self._as_frame(records)
        return self.pipeline.transform(frame)

    def _as_frame(self, records) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            frame = records
        elif isinstance(records, Mapping):
            frame = pd.DataFrame([dict(records)])
        else:
            frame = pd.DataFrame([dict(r) for r in records])
        if frame.empty:
            return pd.DataFrame(columns=self.input_features)
        return align_frame(frame, self.input_features, strict=True)

    # ------------------------------------------------------------ predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities for an already-transformed matrix."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        elif hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X), dtype=np.float64)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            proba = _softmax(scores)
        else:  # pragma: no cover - no such estimator is configured
            codes = np.asarray(self.model.predict(X)).astype(int)
            proba = np.zeros((len(codes), len(self.classes)), dtype=np.float64)
            proba[np.arange(len(codes)), codes] = 1.0
        if proba.shape[1] != len(self.classes):
            raise ModelError(
                f"Model returned {proba.shape[1]} probabilities for "
                f"{len(self.classes)} classes; artefacts are inconsistent."
            )
        return proba

    def predict(
        self,
        records: Sequence[Mapping[str, float]] | pd.DataFrame,
        *,
        keep_transformed: bool = True,
    ) -> list[PredictionResult]:
        """Classify raw flow records end to end."""
        frame = self._as_frame(records)
        if frame.empty:
            return []
        X = self.pipeline.transform(frame)
        proba = self.predict_proba(X)
        raw_records = frame.to_dict(orient="records")

        results: list[PredictionResult] = []
        for index in range(len(frame)):
            row = proba[index]
            argmax_code = int(np.argmax(row))
            argmax_label = self.codec.label_of(argmax_code)
            argmax_confidence = float(row[argmax_code])

            final_code, final_label, final_confidence, thresholded = self._apply_threshold(
                row, argmax_code, argmax_label, argmax_confidence
            )
            is_attack = final_label != self.benign_label
            record = {k: _to_float(v) for k, v in raw_records[index].items()}
            risk = self.risk_scorer.score(
                final_label,
                final_confidence,
                packets_per_second=record.get("flow_packets_s", 0.0),
                is_attack=is_attack,
            )
            results.append(
                PredictionResult(
                    predicted_label=final_label,
                    predicted_code=final_code,
                    is_attack=is_attack,
                    confidence=final_confidence,
                    probabilities={
                        name: float(row[i]) for i, name in enumerate(self.classes)
                    },
                    risk=risk,
                    argmax_label=argmax_label,
                    argmax_confidence=argmax_confidence,
                    threshold_applied=thresholded,
                    features=record,
                    transformed=X[index].copy() if keep_transformed else None,
                    model_name=self.model_name,
                )
            )
        return results

    def predict_one(self, record: Mapping[str, float]) -> PredictionResult:
        """Convenience wrapper for a single flow."""
        results = self.predict([dict(record)])
        if not results:
            raise ModelError("Prediction produced no result for the supplied record.")
        return results[0]

    def _apply_threshold(
        self, row: np.ndarray, argmax_code: int, argmax_label: str, argmax_confidence: float
    ) -> tuple[int, str, float, bool]:
        if argmax_label == self.benign_label:
            return argmax_code, argmax_label, argmax_confidence, False
        if argmax_confidence >= self.attack_threshold:
            return argmax_code, argmax_label, argmax_confidence, False
        if self._benign_code >= 0:
            benign_confidence = float(row[self._benign_code])
            return self._benign_code, self.benign_label, benign_confidence, True
        return argmax_code, argmax_label, argmax_confidence, False

    # --------------------------------------------------------- XAI support
    def proba_function(self):
        """Callable ``f(X_transformed) -> probabilities``, for LIME/SHAP."""

        def _predict(matrix: np.ndarray) -> np.ndarray:
            return self.predict_proba(np.asarray(matrix, dtype=np.float64))

        return _predict

    def background_matrix(self, max_rows: int = 200) -> np.ndarray:
        """Background sample in the transformed space (for SHAP/LIME)."""
        if self.background is None or len(self.background) == 0:
            raise ModelError(
                "No XAI background sample was saved with the model.",
                hint="Re-run training to regenerate models/trained/xai_background.npz.",
            )
        if len(self.background) <= max_rows:
            return self.background
        step = max(1, len(self.background) // max_rows)
        return self.background[::step][:max_rows]

    # ------------------------------------------------------------ describe
    def describe(self) -> dict[str, Any]:
        """Summary rendered on the dashboard's Settings / Model pages."""
        return {
            "model_name": self.model_name,
            "estimator": type(self.model).__name__,
            "classes": self.classes,
            "n_input_features": len(self.input_features),
            "n_output_features": len(self.output_features),
            "attack_probability_threshold": self.attack_threshold,
            "benign_label": self.benign_label,
            "trained_at": self.metadata.get("created_at", "unknown"),
            "feature_policy": self.metadata.get("feature_policy", {}),
            "risk_policy": self.risk_scorer.describe(),
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<Predictor model={self.model_name} classes={len(self.classes)}>"


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.clip(exponent.sum(axis=1, keepdims=True), 1e-12, None)


def _to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(number) or np.isinf(number):
        return 0.0
    return number


def batched(records: Iterable[Mapping[str, float]], size: int = 256):
    """Yield lists of at most ``size`` records (used by the live monitor)."""
    batch: list[Mapping[str, float]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = ["Predictor", "PredictionResult", "batched"]
