"""Risk / severity scoring (Phase 7).

Risk is **not** model confidence
--------------------------------
``confidence`` answers *"how sure is the classifier that this flow belongs to
class C?"*.  It is a property of the model.

``risk_score`` answers *"how much should a security analyst care?"*.  It is a
property of the *situation*, and deliberately combines three different things:

::

    severity   how damaging class C is if the prediction is correct   [0, 1]
    confidence how sure the model is                                  [0, 1]
    volume     how loud the flow is, log-scaled against a reference   [0, 1]

    risk_score = 100 * (w_sev * severity + w_conf * confidence + w_vol * volume)

with the weights taken from ``risk.weights`` in ``config.yaml`` (they are
validated at start-up to sum to 1.0, so the score is always in ``[0, 100]``).

A high-confidence BENIGN classification must never produce a high risk score,
so benign flows use a separate, deliberately small expression::

    risk_score = 100 * 0.20 * (1 - confidence)        -> [0, 20]  (always LOW)

which lets an *uncertain* benign verdict score slightly higher than a certain
one without ever raising an alert on its own.

Both numbers are stored, displayed and explained separately throughout the
application; neither is derived from the other on the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from src.utils.helpers import clamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"
BANDS = (LOW, MEDIUM, HIGH, CRITICAL)
BAND_ORDER = {name: index for index, name in enumerate(BANDS)}

#: Fraction of the scale a benign verdict may ever occupy.
BENIGN_RISK_CEILING = 0.20


@dataclass
class RiskAssessment:
    """Result of :meth:`RiskScorer.score`."""

    score: float
    level: str
    severity: float
    confidence: float
    volume_factor: float
    formula: str
    is_attack: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.score, 2),
            "risk_level": self.level,
            "severity_component": round(self.severity, 4),
            "confidence_component": round(self.confidence, 4),
            "volume_component": round(self.volume_factor, 4),
            "formula": self.formula,
        }


class RiskScorer:
    """Turns a prediction plus flow statistics into a documented risk score."""

    def __init__(
        self,
        severity_map: Mapping[str, float] | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        default_severity: float = 0.65,
        volume_reference_pps: float = 5000.0,
        bands: Mapping[str, float] | None = None,
        benign_label: str = "BENIGN",
    ) -> None:
        self.severity_map = {str(k): float(v) for k, v in (severity_map or {}).items()}
        weights = weights or {"severity": 0.55, "confidence": 0.30, "volume": 0.15}
        self.w_severity = float(weights.get("severity", 0.55))
        self.w_confidence = float(weights.get("confidence", 0.30))
        self.w_volume = float(weights.get("volume", 0.15))
        self.default_severity = float(default_severity)
        self.volume_reference_pps = max(1.0, float(volume_reference_pps))
        bands = bands or {"low": 25, "medium": 50, "high": 75}
        self.band_low = float(bands.get("low", 25))
        self.band_medium = float(bands.get("medium", 50))
        self.band_high = float(bands.get("high", 75))
        self.benign_label = str(benign_label)

    # ------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, settings) -> "RiskScorer":
        risk_cfg = settings.section("risk")
        return cls(
            severity_map=risk_cfg.get("severity", {}),
            weights=risk_cfg.get("weights", {}),
            default_severity=float(risk_cfg.get("default_severity", 0.65)),
            volume_reference_pps=float(risk_cfg.get("volume_reference_pps", 5000.0)),
            bands=risk_cfg.get("bands", {}),
            benign_label=str(settings.get("inference.benign_label", "BENIGN")),
        )

    # -------------------------------------------------------------- score
    def severity_of(self, label: str) -> float:
        """Severity weight of a class, falling back to ``default_severity``."""
        key = str(label)
        if key in self.severity_map:
            return clamp(self.severity_map[key])
        if key == self.benign_label:
            return 0.0
        # Try a case-insensitive match before giving up.
        for known, value in self.severity_map.items():
            if known.lower() == key.lower():
                return clamp(value)
        return clamp(self.default_severity)

    def volume_factor(self, packets_per_second: float | None) -> float:
        """Log-scaled traffic-volume factor in ``[0, 1]``."""
        try:
            pps = max(0.0, float(packets_per_second or 0.0))
        except (TypeError, ValueError):
            return 0.0
        if pps <= 0.0:
            return 0.0
        return clamp(math.log1p(pps) / math.log1p(self.volume_reference_pps))

    def band(self, score: float) -> str:
        """Map a 0-100 score onto its band name."""
        value = float(score)
        if value < self.band_low:
            return LOW
        if value < self.band_medium:
            return MEDIUM
        if value < self.band_high:
            return HIGH
        return CRITICAL

    def score(
        self,
        predicted_label: str,
        confidence: float,
        *,
        packets_per_second: float | None = None,
        is_attack: bool | None = None,
    ) -> RiskAssessment:
        """Compute the risk assessment for one prediction."""
        confidence = clamp(confidence)
        attack = (
            (str(predicted_label) != self.benign_label) if is_attack is None else bool(is_attack)
        )
        volume = self.volume_factor(packets_per_second)

        if not attack:
            raw = BENIGN_RISK_CEILING * (1.0 - confidence)
            value = 100.0 * raw
            formula = (
                f"benign verdict -> 100 * {BENIGN_RISK_CEILING} * (1 - confidence) "
                f"= 100 * {BENIGN_RISK_CEILING} * (1 - {confidence:.3f}) = {value:.2f}"
            )
            return RiskAssessment(
                score=round(value, 2),
                level=self.band(value),
                severity=0.0,
                confidence=confidence,
                volume_factor=volume,
                formula=formula,
                is_attack=False,
            )

        severity = self.severity_of(predicted_label)
        raw = (
            self.w_severity * severity
            + self.w_confidence * confidence
            + self.w_volume * volume
        )
        value = 100.0 * clamp(raw)
        formula = (
            f"100 * ({self.w_severity}*severity[{predicted_label}]={severity:.2f} + "
            f"{self.w_confidence}*confidence={confidence:.3f} + "
            f"{self.w_volume}*volume={volume:.3f}) = {value:.2f}"
        )
        return RiskAssessment(
            score=round(value, 2),
            level=self.band(value),
            severity=severity,
            confidence=confidence,
            volume_factor=volume,
            formula=formula,
            is_attack=True,
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def at_least(level: str, minimum: str) -> bool:
        """``True`` when ``level`` is at or above the ``minimum`` band."""
        return BAND_ORDER.get(str(level).upper(), 0) >= BAND_ORDER.get(
            str(minimum).upper(), 0
        )

    def describe(self) -> dict[str, Any]:
        """Machine-readable description of the scoring policy (shown in the UI)."""
        return {
            "weights": {
                "severity": self.w_severity,
                "confidence": self.w_confidence,
                "volume": self.w_volume,
            },
            "default_severity": self.default_severity,
            "volume_reference_pps": self.volume_reference_pps,
            "bands": {
                "LOW": f"score < {self.band_low}",
                "MEDIUM": f"{self.band_low} <= score < {self.band_medium}",
                "HIGH": f"{self.band_medium} <= score < {self.band_high}",
                "CRITICAL": f"score >= {self.band_high}",
            },
            "benign_rule": (
                f"benign verdicts use 100 * {BENIGN_RISK_CEILING} * (1 - confidence), "
                "so they can never leave the LOW band"
            ),
            "severity_map": dict(self.severity_map),
        }


__all__ = [
    "RiskScorer",
    "RiskAssessment",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
    "BANDS",
    "BAND_ORDER",
]
