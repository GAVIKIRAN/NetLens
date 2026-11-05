"""LIME explanations (Phase 8).

LIME fits a small, sparse linear model in the neighbourhood of one instance, so
it answers a different question from SHAP: *"which features, if perturbed,
would flip this particular decision?"* rather than *"how is the model's output
distributed across features overall?"*.  Showing both is the point - when they
agree, the explanation is trustworthy; when they disagree, the analyst learns
that the decision sits on a boundary.

The explainer is fitted on the same background sample that SHAP uses (the
class-stratified subset of the training data saved by the training run), in the
transformed feature space the model consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.features.contract import label_for, strip_transformer_prefix
from src.utils.errors import ExplainerError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LimeExplanation:
    """LIME output for one instance and one class."""

    class_name: str
    class_index: int
    contributions: list[dict[str, Any]]
    local_prediction: float = 0.0
    score: float = 0.0
    intercept: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_index": self.class_index,
            "local_prediction": round(float(self.local_prediction), 6),
            "r2_score": round(float(self.score), 6),
            "intercept": round(float(self.intercept), 6),
            "contributions": self.contributions,
        }


class LimeTabularExplainerWrapper:
    """Thin wrapper that keeps LIME optional and never crashes the pipeline."""

    def __init__(
        self,
        background: np.ndarray,
        feature_names: Sequence[str],
        class_names: Sequence[str],
        proba_function,
        *,
        num_samples: int = 1000,
        num_features: int = 10,
        discretize_continuous: bool = True,
        random_state: int = 42,
    ) -> None:
        self.background = np.asarray(background, dtype=np.float64)
        self.feature_names = [str(f) for f in feature_names]
        self.class_names = [str(c) for c in class_names]
        self.proba_function = proba_function
        self.num_samples = int(num_samples)
        self.num_features = int(num_features)
        self.explainer = None
        self.error = ""

        try:
            from lime.lime_tabular import LimeTabularExplainer

            self.explainer = LimeTabularExplainer(
                training_data=self.background,
                feature_names=self.feature_names,
                class_names=self.class_names,
                discretize_continuous=bool(discretize_continuous),
                mode="classification",
                random_state=int(random_state),
                sample_around_instance=True,
            )
            logger.info(
                "LIME ready (%d background rows, %d features)",
                len(self.background), len(self.feature_names),
            )
        except Exception as exc:
            self.error = f"LIME is unavailable: {exc}"
            logger.warning(self.error)

    @property
    def available(self) -> bool:
        return self.explainer is not None

    def explain(
        self,
        x: np.ndarray,
        class_index: int,
        *,
        raw_features: dict[str, float] | None = None,
        top_k: int | None = None,
    ) -> LimeExplanation:
        """Explain one instance for one class."""
        if not self.available:
            raise ExplainerError(self.error or "LIME explainer is unavailable.")

        instance = np.asarray(x, dtype=np.float64).reshape(-1)
        class_index = int(class_index)
        num_features = int(top_k or self.num_features)

        try:
            explanation = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=self.proba_function,
                labels=(class_index,),
                num_features=num_features,
                num_samples=self.num_samples,
            )
        except Exception as exc:
            raise ExplainerError(f"LIME computation failed: {exc}") from exc

        mapping = explanation.as_map().get(class_index, [])
        conditions = dict(explanation.as_list(label=class_index)) if mapping else {}
        condition_by_index: dict[int, str] = {}
        for (feature_index, _weight), condition in zip(mapping, conditions.keys()):
            condition_by_index[int(feature_index)] = str(condition)

        rows: list[dict[str, Any]] = []
        for feature_index, weight in mapping:
            index = int(feature_index)
            transformed_name = (
                self.feature_names[index] if index < len(self.feature_names) else str(index)
            )
            canonical = strip_transformer_prefix(transformed_name)
            value: float | None = None
            if raw_features is not None and canonical in raw_features:
                value = float(raw_features[canonical])
            elif index < len(instance):
                value = float(instance[index])
            rows.append(
                {
                    "feature": canonical,
                    "transformed_feature": transformed_name,
                    "label": label_for(transformed_name),
                    "value": value,
                    "contribution": float(weight),
                    "abs_contribution": float(abs(weight)),
                    "direction": "increases" if weight > 0 else "decreases",
                    "condition": condition_by_index.get(index, ""),
                }
            )
        rows.sort(key=lambda r: r["abs_contribution"], reverse=True)

        local_prediction = 0.0
        try:
            local = getattr(explanation, "local_pred", None)
            if local is not None:
                local_prediction = float(np.asarray(local).reshape(-1)[0])
        except Exception:  # pragma: no cover - version dependent
            local_prediction = 0.0

        return LimeExplanation(
            class_name=(
                self.class_names[class_index]
                if class_index < len(self.class_names)
                else str(class_index)
            ),
            class_index=class_index,
            contributions=rows,
            local_prediction=local_prediction,
            score=float(getattr(explanation, "score", 0.0) or 0.0),
            intercept=float(
                (getattr(explanation, "intercept", {}) or {}).get(class_index, 0.0)
            ),
        )


__all__ = ["LimeTabularExplainerWrapper", "LimeExplanation"]
