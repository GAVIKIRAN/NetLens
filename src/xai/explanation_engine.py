"""Explanation engine (Phase 9): technical contributions -> security narrative.

Every sentence is generated from the *measured* contributions of the current
prediction.  Nothing is templated per attack class and nothing is hard-coded:
change the model, the features or the flow and the wording changes with it.

Language policy
---------------
Feature attributions describe the **model's reasoning**, not physical
causation.  A high packet rate did not *cause* a DDoS; it *contributed to the
model classifying the flow as DDoS*.  The wording throughout this module is
deliberately built around "contributed to", "pushed the model towards" and
"reduced the model's confidence", and never around "caused".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.contract import BY_NAME, describe, label_for
from src.utils.errors import ExplainerError
from src.utils.helpers import human_bytes
from src.utils.logger import get_logger

logger = get_logger(__name__)

CAUSATION_SAFE_VERBS = (
    "contributed to",
    "pushed the model towards",
    "reduced the model's confidence in",
)


# ===========================================================================
#  Value formatting
# ===========================================================================
def format_value(feature: str, value: float | None) -> str:
    """Render a feature value with a sensible unit."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "not observed"

    spec = BY_NAME.get(feature)
    unit = spec.unit if spec else ""

    if unit == "us":
        if abs(number) >= 1_000_000:
            return f"{number / 1_000_000:,.2f} s"
        if abs(number) >= 1_000:
            return f"{number / 1_000:,.2f} ms"
        return f"{number:,.0f} µs"
    if unit == "bytes":
        return human_bytes(number)
    if unit == "bytes/s":
        return f"{human_bytes(number)}/s"
    if unit in ("packets/s",):
        return f"{number:,.1f} packets/s"
    if unit in ("packets", "count"):
        return f"{number:,.0f}"
    if unit == "ratio":
        return f"{number:,.2f}"
    if unit == "bytes^2":
        return f"{number:,.0f}"
    if feature == "protocol":
        from src.features.contract import PROTOCOL_NAMES

        return PROTOCOL_NAMES.get(int(number), str(int(number)))
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:,.3f}"


# ===========================================================================
#  Narrative generation
# ===========================================================================
@dataclass
class Explanation:
    """A complete, human-readable explanation of one prediction."""

    prediction: str
    confidence: float
    risk_score: float
    risk_level: str
    headline: str = ""
    summary: str = ""
    bullet_points: list[str] = field(default_factory=list)
    positive: list[dict[str, Any]] = field(default_factory=list)
    negative: list[dict[str, Any]] = field(default_factory=list)
    caveat: str = ""
    explainer: str = ""

    def text(self) -> str:
        parts = [self.headline, "", self.summary]
        if self.bullet_points:
            parts.append("")
            parts.extend(f"- {line}" for line in self.bullet_points)
        if self.caveat:
            parts.extend(["", self.caveat])
        return "\n".join(parts).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": round(float(self.confidence), 6),
            "risk_score": round(float(self.risk_score), 2),
            "risk_level": self.risk_level,
            "headline": self.headline,
            "summary": self.summary,
            "bullet_points": self.bullet_points,
            "positive": self.positive,
            "negative": self.negative,
            "caveat": self.caveat,
            "explainer": self.explainer,
            "text": self.text(),
        }


def _phrase_for(entry: Mapping[str, Any], predicted: str) -> str:
    """One human sentence for a single feature contribution."""
    feature = str(entry.get("feature", ""))
    label = str(entry.get("label") or label_for(feature))
    value = entry.get("value")
    contribution = float(entry.get("contribution", 0.0))
    spec = describe(feature)
    rendered = format_value(feature, value)

    if contribution > 0:
        verb = "pushed the model towards"
        target = f"'{predicted}'"
    else:
        verb = "reduced the model's confidence in"
        target = f"'{predicted}'"

    descriptor = ""
    if spec is not None and spec.description:
        descriptor = spec.description.rstrip(".").lower()

    if descriptor:
        return (
            f"**{label}** = {rendered} ({descriptor}) {verb} {target} "
            f"[weight {contribution:+.4f}]"
        )
    return f"**{label}** = {rendered} {verb} {target} [weight {contribution:+.4f}]"


def build_explanation(
    prediction: str,
    confidence: float,
    contributions: Sequence[Mapping[str, Any]],
    *,
    risk_score: float = 0.0,
    risk_level: str = "LOW",
    benign_label: str = "BENIGN",
    explainer: str = "SHAP",
    top_k: int = 5,
    threshold_applied: bool = False,
) -> Explanation:
    """Compose the human-readable explanation for one prediction."""
    ranked = sorted(
        [dict(c) for c in contributions],
        key=lambda c: abs(float(c.get("contribution", 0.0))),
        reverse=True,
    )
    positive = [c for c in ranked if float(c.get("contribution", 0.0)) > 0][:top_k]
    negative = [c for c in ranked if float(c.get("contribution", 0.0)) < 0][:top_k]

    is_attack = str(prediction) != str(benign_label)
    if is_attack:
        headline = (
            f"{prediction} detected - {risk_level} risk "
            f"(risk score {risk_score:.0f}/100, model confidence {confidence:.1%})"
        )
    else:
        headline = (
            f"Traffic classified as {prediction} "
            f"(model confidence {confidence:.1%}, {risk_level} risk)"
        )

    top_labels = [str(c.get("label") or label_for(str(c.get("feature", "")))) for c in ranked[:3]]
    if top_labels:
        drivers = ", ".join(top_labels[:-1]) + (" and " + top_labels[-1] if len(top_labels) > 1 else top_labels[0])
        if len(top_labels) == 1:
            drivers = top_labels[0]
    else:
        drivers = "no individual feature"

    summary = (
        f"The model assigned this flow to **{prediction}** with "
        f"{confidence:.1%} confidence. Using {explainer}, the attribution is "
        f"dominated by {drivers}. These features **contributed to the model's "
        f"prediction**; they are statistical associations learned from "
        f"CIC-IDS2017 and are not evidence that any one of them caused the "
        f"behaviour."
    )
    if threshold_applied:
        summary += (
            " Note: the model's highest-scoring class was an attack, but its "
            "probability fell below the configured decision threshold, so the "
            "flow is reported as benign."
        )

    bullets = [_phrase_for(entry, str(prediction)) for entry in positive]
    bullets += [_phrase_for(entry, str(prediction)) for entry in negative]
    if not bullets:
        bullets = [
            "No individual feature carried a meaningful attribution weight for "
            "this flow; the decision came from the combination of many small "
            "contributions."
        ]

    caveat = (
        "Interpretation note: contributions are model attributions, not causal "
        "claims. A flow can be misclassified when live traffic differs from the "
        "CIC-IDS2017 training distribution - confirm alerts against your own "
        "network baseline before acting on them."
    )

    return Explanation(
        prediction=str(prediction),
        confidence=float(confidence),
        risk_score=float(risk_score),
        risk_level=str(risk_level),
        headline=headline,
        summary=summary,
        bullet_points=bullets,
        positive=positive,
        negative=negative,
        caveat=caveat,
        explainer=str(explainer),
    )


# ===========================================================================
#  Service that ties the predictor, SHAP, LIME and the narrative together
# ===========================================================================
class ExplanationService:
    """Builds SHAP/LIME explainers once and explains predictions on demand."""

    def __init__(self, predictor, settings) -> None:
        from src.xai.lime_explainer import LimeTabularExplainerWrapper
        from src.xai.shap_explainer import ShapExplainer

        self.predictor = predictor
        self.settings = settings
        xai_cfg = settings.section("xai")
        self.enabled = bool(xai_cfg.get("enabled", True))
        self.top_k = int(xai_cfg.get("top_k_features", 8))
        self.explain_inline = bool(xai_cfg.get("explain_alerts_inline", True))
        self.benign_label = str(settings.get("inference.benign_label", "BENIGN"))

        self.shap: ShapExplainer | None = None
        self.lime: LimeTabularExplainerWrapper | None = None
        self.errors: list[str] = []

        if not self.enabled:
            logger.info("XAI disabled via configuration.")
            return

        try:
            background = predictor.background_matrix(
                int(xai_cfg.get("background_samples", 200))
            )
        except Exception as exc:
            self.errors.append(str(exc))
            logger.warning("XAI background unavailable: %s", exc)
            return

        self.shap = ShapExplainer(
            predictor.model,
            background,
            predictor.output_features,
            predictor.classes,
            max_evals=int((xai_cfg.get("shap", {}) or {}).get("max_evals", 200)),
            proba_function=predictor.proba_function(),
        )
        if not self.shap.available:
            self.errors.append(self.shap.error)

        lime_cfg = xai_cfg.get("lime", {}) or {}
        self.lime = LimeTabularExplainerWrapper(
            background,
            predictor.output_features,
            predictor.classes,
            predictor.proba_function(),
            num_samples=int(lime_cfg.get("num_samples", 1000)),
            num_features=int(lime_cfg.get("num_features", 10)),
            discretize_continuous=bool(lime_cfg.get("discretize_continuous", True)),
            random_state=int(settings.get("training.random_state", 42)),
        )
        if not self.lime.available:
            self.errors.append(self.lime.error)

    # ------------------------------------------------------------ queries
    @property
    def shap_available(self) -> bool:
        return self.shap is not None and self.shap.available

    @property
    def lime_available(self) -> bool:
        return self.lime is not None and self.lime.available

    @property
    def inline_capable(self) -> bool:
        """Fast enough to explain every alert while capture is running?"""
        return bool(self.explain_inline and self.shap_available and self.shap.is_fast)

    # ------------------------------------------------------------ explain
    def explain_prediction(
        self,
        result,
        *,
        use_shap: bool = True,
        use_lime: bool = False,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Explain a :class:`~src.inference.predictor.PredictionResult`."""
        top_k = int(top_k or self.top_k)
        transformed = result.transformed
        if transformed is None:
            transformed = self.predictor.transform([result.features])[0]
        instance = np.asarray(transformed, dtype=np.float64).reshape(1, -1)
        class_index = int(result.predicted_code)

        payload: dict[str, Any] = {
            "prediction": result.predicted_label,
            "confidence": float(result.confidence),
            "risk_score": float(result.risk.score) if result.risk else 0.0,
            "risk_level": result.risk.level if result.risk else "LOW",
            "shap": None,
            "lime": None,
            "narrative": None,
            "errors": [],
        }

        contributions: list[dict[str, Any]] = []
        explainer_used = ""

        if use_shap and self.shap_available:
            try:
                shap_explanation = self.shap.explain(
                    instance, class_index, raw_features=result.features, top_k=top_k
                )
                payload["shap"] = shap_explanation.to_dict()
                contributions = shap_explanation.contributions
                explainer_used = shap_explanation.explainer_type
            except ExplainerError as exc:
                payload["errors"].append(f"SHAP: {exc}")
                logger.warning("SHAP explanation failed: %s", exc)
            except Exception as exc:  # pragma: no cover - library specific
                payload["errors"].append(f"SHAP: {exc}")
                logger.exception("Unexpected SHAP failure")

        if use_lime and self.lime_available:
            try:
                lime_explanation = self.lime.explain(
                    instance[0], class_index, raw_features=result.features, top_k=top_k
                )
                payload["lime"] = lime_explanation.to_dict()
                if not contributions:
                    contributions = lime_explanation.contributions
                    explainer_used = "LIME"
            except ExplainerError as exc:
                payload["errors"].append(f"LIME: {exc}")
                logger.warning("LIME explanation failed: %s", exc)
            except Exception as exc:  # pragma: no cover - library specific
                payload["errors"].append(f"LIME: {exc}")
                logger.exception("Unexpected LIME failure")

        narrative = build_explanation(
            result.predicted_label,
            result.confidence,
            contributions,
            risk_score=payload["risk_score"],
            risk_level=payload["risk_level"],
            benign_label=self.benign_label,
            explainer=explainer_used or "the model",
            top_k=min(5, top_k),
            threshold_applied=bool(getattr(result, "threshold_applied", False)),
        )
        payload["narrative"] = narrative.to_dict()
        payload["top_features"] = contributions[:top_k]
        payload["explainer"] = explainer_used
        return payload

    def quick_top_features(self, result, top_k: int | None = None) -> list[dict[str, Any]]:
        """Fast inline SHAP contributions for an alert (tree/linear models only)."""
        if not self.inline_capable:
            return []
        try:
            transformed = result.transformed
            if transformed is None:
                transformed = self.predictor.transform([result.features])[0]
            explanation = self.shap.explain(
                np.asarray(transformed).reshape(1, -1),
                int(result.predicted_code),
                raw_features=result.features,
                top_k=int(top_k or self.top_k),
            )
            return explanation.contributions
        except Exception as exc:
            logger.debug("Inline SHAP skipped: %s", exc)
            return []

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "shap_available": self.shap_available,
            "shap_explainer": self.shap.explainer_type if self.shap else "none",
            "shap_fast": bool(self.shap.is_fast) if self.shap else False,
            "lime_available": self.lime_available,
            "inline_explanations": self.inline_capable,
            "errors": self.errors,
        }


__all__ = [
    "Explanation",
    "build_explanation",
    "format_value",
    "ExplanationService",
    "CAUSATION_SAFE_VERBS",
]
