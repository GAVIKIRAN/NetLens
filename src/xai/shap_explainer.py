"""SHAP explanations (Phase 8).

Choosing the right explainer matters, both for correctness and for speed:

======================  ==========================  ==============  =========
Model                   Explainer                   Complexity      Inline?
======================  ==========================  ==============  =========
RandomForest, XGBoost   ``shap.TreeExplainer``      exact, fast     yes
LogisticRegression      ``shap.LinearExplainer``    exact, fast     yes
Calibrated LinearSVC    ``shap.KernelExplainer``    sampled, slow   no
======================  ==========================  ==============  =========

Explanations are computed in the **transformed** feature space (the matrix the
model actually sees) and the column names are mapped back to their canonical
contract names for display, so a contribution always refers to a feature the
analyst can recognise.

Only the fast explainers are used inline while capturing; a slow model still
gets a full explanation, but on demand from the AI Explanation page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.features.contract import label_for, strip_transformer_prefix
from src.utils.errors import ExplainerError
from src.utils.logger import get_logger

logger = get_logger(__name__)

TREE_MODELS = (
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "XGBRFClassifier",
    "LGBMClassifier",
    "HistGradientBoostingClassifier",
)
LINEAR_MODELS = ("LogisticRegression", "SGDClassifier", "RidgeClassifier")


@dataclass
class ShapExplanation:
    """SHAP output for one instance and one class."""

    class_name: str
    class_index: int
    base_value: float
    contributions: list[dict[str, Any]]
    explainer_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_index": self.class_index,
            "base_value": round(float(self.base_value), 6),
            "explainer": self.explainer_type,
            "contributions": self.contributions,
        }


class ShapExplainer:
    """Model-appropriate SHAP explainer with graceful degradation."""

    def __init__(
        self,
        model,
        background: np.ndarray,
        feature_names: Sequence[str],
        class_names: Sequence[str],
        *,
        max_evals: int = 200,
        proba_function=None,
    ) -> None:
        self.model = model
        self.background = np.asarray(background, dtype=np.float64)
        self.feature_names = [str(f) for f in feature_names]
        self.class_names = [str(c) for c in class_names]
        self.max_evals = int(max_evals)
        self.proba_function = proba_function
        self.explainer = None
        self.explainer_type = "unavailable"
        self.is_fast = False
        self.error = ""
        self._build()

    # ------------------------------------------------------------- build
    def _build(self) -> None:
        try:
            import shap
        except Exception as exc:  # pragma: no cover - optional dependency
            self.error = f"SHAP is not installed: {exc}"
            logger.warning(self.error)
            return

        estimator_name = type(self.model).__name__
        try:
            if estimator_name in TREE_MODELS:
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "TreeExplainer"
                self.is_fast = True
            elif estimator_name in LINEAR_MODELS:
                self.explainer = shap.LinearExplainer(self.model, self.background)
                self.explainer_type = "LinearExplainer"
                self.is_fast = True
            else:
                predict = self.proba_function or self.model.predict_proba
                summary = shap.kmeans(self.background, min(25, len(self.background)))
                self.explainer = shap.KernelExplainer(predict, summary)
                self.explainer_type = "KernelExplainer"
                self.is_fast = False
            logger.info(
                "SHAP ready for %s using %s (fast=%s)",
                estimator_name, self.explainer_type, self.is_fast,
            )
        except Exception as exc:
            self.error = f"Could not build a SHAP explainer for {estimator_name}: {exc}"
            self.explainer = None
            self.explainer_type = "unavailable"
            logger.warning(self.error)

    @property
    def available(self) -> bool:
        return self.explainer is not None

    # ----------------------------------------------------------- explain
    def shap_values(self, x: np.ndarray) -> np.ndarray:
        """Return SHAP values shaped ``(n_samples, n_features, n_classes)``."""
        if not self.available:
            raise ExplainerError(self.error or "SHAP explainer is unavailable.")
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        try:
            if self.explainer_type == "KernelExplainer":
                raw = self.explainer.shap_values(
                    matrix, nsamples=self.max_evals, silent=True
                )
            else:
                raw = self.explainer.shap_values(matrix, check_additivity=False)
        except TypeError:
            raw = self.explainer.shap_values(matrix)
        except Exception as exc:
            raise ExplainerError(f"SHAP computation failed: {exc}") from exc
        return self._normalise(raw, matrix.shape)

    def _normalise(self, raw: Any, shape: tuple[int, int]) -> np.ndarray:
        n_samples, n_features = shape
        n_classes = len(self.class_names)

        if isinstance(raw, list):
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in raw], axis=-1)
        else:
            stacked = np.asarray(raw, dtype=np.float64)

        if stacked.ndim == 3:
            values = stacked
        elif stacked.ndim == 2:
            # Binary / single-output model: class 1 gets +phi, class 0 gets -phi.
            if n_classes == 2:
                values = np.stack([-stacked, stacked], axis=-1)
            else:
                values = stacked.reshape(n_samples, n_features, 1)
        else:  # pragma: no cover - defensive
            raise ExplainerError(f"Unexpected SHAP output with shape {stacked.shape}.")

        if values.shape[0] != n_samples or values.shape[1] != n_features:
            # Some versions return (classes, samples, features).
            if values.shape[-1] == n_features and values.shape[-2] == n_samples:
                values = np.moveaxis(values, 0, -1)
        return values

    def base_values(self) -> np.ndarray:
        expected = getattr(self.explainer, "expected_value", 0.0)
        array = np.atleast_1d(np.asarray(expected, dtype=np.float64))
        return array

    def explain(
        self,
        x: np.ndarray,
        class_index: int,
        *,
        raw_features: dict[str, float] | None = None,
        top_k: int = 8,
    ) -> ShapExplanation:
        """Explain one instance for one class."""
        values = self.shap_values(x)
        class_index = int(class_index)
        if values.shape[2] == 1:
            class_index_used = 0
        else:
            class_index_used = min(class_index, values.shape[2] - 1)
        phi = values[0, :, class_index_used]

        bases = self.base_values()
        base = float(bases[class_index_used]) if len(bases) > class_index_used else float(bases[0])

        instance = np.asarray(x, dtype=np.float64).reshape(-1)
        contributions = build_contributions(
            phi, self.feature_names, instance, raw_features, top_k
        )
        return ShapExplanation(
            class_name=(
                self.class_names[class_index]
                if class_index < len(self.class_names)
                else str(class_index)
            ),
            class_index=class_index,
            base_value=base,
            contributions=contributions,
            explainer_type=self.explainer_type,
        )

    def global_importance(self, sample: np.ndarray | None = None, top_k: int = 20):
        """Mean |SHAP| per feature over a sample - the global feature ranking."""
        matrix = self.background if sample is None else np.asarray(sample, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if len(matrix) > 200:
            matrix = matrix[:200]
        values = self.shap_values(matrix)
        magnitude = np.abs(values).mean(axis=(0, 2))
        order = np.argsort(magnitude)[::-1][: int(top_k)]
        return [
            {
                "feature": strip_transformer_prefix(self.feature_names[i]),
                "label": label_for(self.feature_names[i]),
                "importance": float(magnitude[i]),
            }
            for i in order
        ]


def build_contributions(
    phi: np.ndarray,
    feature_names: Sequence[str],
    instance: np.ndarray,
    raw_features: dict[str, float] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """Turn a SHAP/LIME weight vector into a ranked contribution table."""
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    order = np.argsort(np.abs(phi))[::-1][: max(1, int(top_k))]
    rows: list[dict[str, Any]] = []
    for index in order:
        transformed_name = str(feature_names[index])
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
                "contribution": float(phi[index]),
                "direction": "increases" if phi[index] > 0 else "decreases",
                "abs_contribution": float(abs(phi[index])),
            }
        )
    return rows


__all__ = ["ShapExplainer", "ShapExplanation", "build_contributions", "TREE_MODELS"]
