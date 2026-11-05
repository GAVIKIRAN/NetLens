"""Candidate model construction and data-driven model selection (Phase 5).

Nothing here assumes that Random Forest or XGBoost wins.  Every enabled model
is trained, scored on the **validation** split, and the winner is chosen from
those measurements.  The test split is touched exactly once, afterwards, to
report the honest generalisation estimate of the already-selected model.

Tie-breaking: if two models are within ``model_selection.tie_break_margin`` of
each other on the primary metric, the one with the lower per-record inference
latency wins - a real-time IDS that is 0.2 % more accurate but 40x slower is
the worse engineering choice.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.training.evaluate import EvaluationResult
from src.utils.errors import ModelError
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Metric name -> attribute on :class:`~src.training.evaluate.EvaluationResult`.
METRIC_ATTRIBUTES = {
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "detection_rate": "binary_detection_rate",
}

try:  # pragma: no cover - environment dependent
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False


@dataclass
class ModelCandidate:
    """One trained candidate together with its measurements."""

    name: str
    estimator: Any = None
    fit_seconds: float = 0.0
    supports_proba: bool = True
    validation: EvaluationResult | None = None
    test: EvaluationResult | None = None
    error: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def metric(self, key: str) -> float:
        if self.validation is None:
            return float("-inf")
        attribute = METRIC_ATTRIBUTES.get(key, "f1_macro")
        return float(getattr(self.validation, attribute, 0.0))

    @property
    def latency(self) -> float:
        if self.validation is None:
            return float("inf")
        return float(self.validation.predict_latency_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fit_seconds": round(self.fit_seconds, 3),
            "supports_proba": self.supports_proba,
            "error": self.error,
            "params": self.params,
            "validation": self.validation.to_dict() if self.validation else None,
            "test": self.test.to_dict() if self.test else None,
        }


@dataclass
class SelectionResult:
    """Outcome of :func:`select_best_model`."""

    best_name: str
    primary_metric: str
    rationale: str
    table: pd.DataFrame
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.best_name,
            "primary_metric": self.primary_metric,
            "rationale": self.rationale,
            "validation_scores": self.scores,
            "comparison": self.table.to_dict(orient="records"),
        }


def _filter_params(estimator_cls, params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop parameters the installed estimator version does not accept.

    scikit-learn removes deprecated keyword arguments between releases (for
    example ``LogisticRegression(multi_class=...)``).  Silently filtering keeps
    a user's ``config.yaml`` working across versions instead of crashing.
    """
    try:
        signature = inspect.signature(estimator_cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - exotic estimators
        return dict(params)
    # XGBoost's constructor is ``(self, *, objective=..., **kwargs)``: it accepts
    # arbitrary keyword arguments, so nothing may be filtered out for it.
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(params)
    accepted = set(signature.parameters)
    filtered: dict[str, Any] = {}
    for key, value in params.items():
        if key in accepted:
            filtered[key] = value
        else:
            logger.warning(
                "%s does not accept parameter '%s' in this version; ignoring it.",
                estimator_cls.__name__, key,
            )
    return filtered


def build_candidate_models(
    training_config: Mapping[str, Any],
    *,
    random_state: int = 42,
    n_jobs: int = -1,
    n_classes: int = 2,
    use_class_weight: bool = True,
) -> dict[str, Any]:
    """Instantiate every enabled model from the ``training.models`` config."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    models_cfg = dict(training_config.get("models", {}) or {})
    models: dict[str, Any] = {}

    # ---------------------------------------------------------- Decision Tree
    cfg = models_cfg.get("decision_tree", {}) or {}
    if cfg.get("enabled", True):
        params = _filter_params(DecisionTreeClassifier, dict(cfg.get("params", {}) or {}))
        params["random_state"] = random_state
        if use_class_weight:
            params["class_weight"] = "balanced"
        models["decision_tree"] = DecisionTreeClassifier(**params)

    # ---------------------------------------------------------- Random Forest
    cfg = models_cfg.get("random_forest", {}) or {}
    if cfg.get("enabled", True):
        params = _filter_params(RandomForestClassifier, dict(cfg.get("params", {}) or {}))
        params.setdefault("n_jobs", n_jobs)
        params["random_state"] = random_state
        if use_class_weight:
            params["class_weight"] = "balanced_subsample"
        models["random_forest"] = RandomForestClassifier(**params)

    # ---------------------------------------------------- Logistic Regression
    cfg = models_cfg.get("logistic_regression", {}) or {}
    if cfg.get("enabled", True):
        params = _filter_params(LogisticRegression, dict(cfg.get("params", {}) or {}))
        # n_jobs is a no-op for LogisticRegression from scikit-learn 1.8 and
        # emits a FutureWarning, so it is never forwarded.
        params.pop("n_jobs", None)
        params["random_state"] = random_state
        if use_class_weight:
            params["class_weight"] = "balanced"
        # 'saga' is the only solver that supports every penalty and is safe for
        # large sparse-ish inputs; it also honours n_jobs for multinomial fits.
        params.setdefault("solver", "saga")
        params.setdefault("max_iter", 1000)
        models["logistic_regression"] = LogisticRegression(**params)

    # --------------------------------------------------------------- SVM
    cfg = models_cfg.get("svm", {}) or {}
    if cfg.get("enabled", False):
        params = _filter_params(LinearSVC, dict(cfg.get("params", {}) or {}))
        params["random_state"] = random_state
        if use_class_weight:
            params["class_weight"] = "balanced"
        base = LinearSVC(**params)
        # LinearSVC has no predict_proba; calibration supplies well-formed
        # probabilities, which the risk engine and SHAP/LIME both need.
        models["svm"] = CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv=3, n_jobs=n_jobs
        )

    # ----------------------------------------------------------- XGBoost
    cfg = models_cfg.get("xgboost", {}) or {}
    if cfg.get("enabled", True):
        if not XGBOOST_AVAILABLE:
            logger.warning("xgboost is not installed; skipping that candidate.")
        else:
            params = _filter_params(XGBClassifier, dict(cfg.get("params", {}) or {}))
            params.setdefault("n_jobs", n_jobs)
            params["random_state"] = random_state
            params.setdefault("tree_method", "hist")
            params["objective"] = "multi:softprob" if n_classes > 2 else "binary:logistic"
            if n_classes > 2:
                params["num_class"] = n_classes
            params.setdefault("eval_metric", "mlogloss" if n_classes > 2 else "logloss")
            models["xgboost"] = XGBClassifier(**params)

    if not models:
        raise ModelError(
            "No model is enabled in config.yaml (training.models.*.enabled).",
            hint="Enable at least one candidate and re-run training.",
        )
    logger.info("Candidate models: %s", ", ".join(models))
    return models


def select_best_model(
    candidates: Sequence[ModelCandidate],
    *,
    primary_metric: str = "macro_f1",
    tie_break_margin: float = 0.005,
) -> SelectionResult:
    """Pick the winner from measured validation performance."""
    trained = [c for c in candidates if c.estimator is not None and c.validation is not None]
    if not trained:
        raise ModelError(
            "No model finished training successfully; cannot select a best model.",
            hint="Check logs/nids.log for the per-model error messages.",
        )
    if primary_metric not in METRIC_ATTRIBUTES:
        logger.warning(
            "Unknown model_selection.primary_metric '%s'; falling back to macro_f1.",
            primary_metric,
        )
        primary_metric = "macro_f1"

    scores = {c.name: c.metric(primary_metric) for c in trained}
    top_score = max(scores.values())
    contenders = [c for c in trained if top_score - scores[c.name] <= float(tie_break_margin)]

    if len(contenders) == 1:
        winner = contenders[0]
        rationale = (
            f"'{winner.name}' had the highest validation {primary_metric} "
            f"({scores[winner.name]:.4f}) with no competitor within "
            f"{tie_break_margin:.3f}."
        )
    else:
        winner = min(contenders, key=lambda c: (c.latency, -scores[c.name]))
        others = ", ".join(
            f"{c.name}={scores[c.name]:.4f}@{c.latency:.4f}ms"
            for c in contenders
            if c.name != winner.name
        )
        rationale = (
            f"'{winner.name}' won on speed: {len(contenders)} models were within "
            f"{tie_break_margin:.3f} {primary_metric} of each other "
            f"({winner.name}={scores[winner.name]:.4f}@{winner.latency:.4f}ms vs {others}), "
            "so the lowest inference latency decided it."
        )

    rows = []
    for candidate in candidates:
        row: dict[str, Any] = {
            "model": candidate.name,
            "status": "trained" if candidate.estimator is not None else "failed",
            "error": candidate.error,
            "fit_seconds": round(candidate.fit_seconds, 2),
            "predict_latency_ms": (
                round(candidate.latency, 5) if candidate.validation else None
            ),
            "selected": candidate.name == winner.name,
        }
        if candidate.validation:
            for key, attribute in METRIC_ATTRIBUTES.items():
                row[f"val_{key}"] = round(
                    float(getattr(candidate.validation, attribute, 0.0)), 6
                )
            row["val_false_positive_rate"] = round(
                candidate.validation.binary_false_positive_rate, 6
            )
        rows.append(row)
    table = pd.DataFrame(rows)

    logger.info("Model selection: %s", rationale)
    return SelectionResult(
        best_name=winner.name,
        primary_metric=primary_metric,
        rationale=rationale,
        table=table,
        scores={k: round(float(v), 6) for k, v in scores.items()},
    )


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced per-sample weights, used for estimators without class_weight."""
    from sklearn.utils.class_weight import compute_sample_weight

    return compute_sample_weight(class_weight="balanced", y=y)


__all__ = [
    "ModelCandidate",
    "SelectionResult",
    "build_candidate_models",
    "select_best_model",
    "compute_sample_weights",
    "METRIC_ATTRIBUTES",
    "XGBOOST_AVAILABLE",
]
