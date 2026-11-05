"""Combine the model's prediction and behavioural findings into one IDS decision.

The model's label is never changed.  It is stored as is, next to the behavioural result
and the final decision, so an analyst can see where each came from.

Protocol compatibility comes from the training data
---------------------------------------------------
In CIC-IDS2017 every row of every attack class has a TCP window value (``Init_Win_bytes_
forward`` != -1), while 44% of BENIGN rows do not.  The model was never shown an attack
over UDP or ICMP, and it cannot see the protocol, so an attack label on such a flow has
no basis.  The label is kept for the record but does not raise an alert.  Real UDP abuse
is left to the behaviour layer.

Risk is not the probability of an attack.  Model confidence is the classifier's score for
its chosen class and is only one input, as ``RiskScorer`` documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.detection.behavior import ATTACK, SUSPICIOUS, Finding
from src.inference.risk import RiskScorer
from src.utils.logger import get_logger

logger = get_logger(__name__)

BENIGN = "BENIGN"
_ORDER = {BENIGN: 0, SUSPICIOUS: 1, ATTACK: 2}

DEFAULT_TCP_TRAINED_LABELS = (
    "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris",
    "FTP-Patator", "Heartbleed", "Infiltration", "PortScan", "Rare_Attack", "SSH-Patator",
    "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection",
)
NON_TCP_PROTOCOLS = ("UDP", "ICMP")

DEFAULT_FAMILIES = {
    "BENIGN": "Benign",
    "Bot": "Botnet",
    "DDoS": "Denial of service",
    "DoS GoldenEye": "Denial of service",
    "DoS Hulk": "Denial of service",
    "DoS Slowhttptest": "Denial of service",
    "DoS slowloris": "Denial of service",
    "FTP-Patator": "Brute force",
    "SSH-Patator": "Brute force",
    "PortScan": "Reconnaissance",
    "Rare_Attack": "Rare attack",
    "Web Attack - Brute Force": "Web attack",
    "Web Attack - XSS": "Web attack",
    "Web Attack - Sql Injection": "Web attack",
}

DEFAULT_BEHAVIOUR_SEVERITY = {
    "port_scan": 0.75,
    "host_sweep": 0.75,
    "failed_connections": 0.50,
    "repeated_attempts": 0.45,
    "connection_rate": 0.50,
    "syn_flood": 0.90,
    "traffic_burst": 0.50,
}


@dataclass
class Decision:
    """Layer 3 output for one flow."""

    final: str
    ml_label: str
    ml_family: str
    behaviour_result: str
    reason: str
    risk_score: float
    risk_level: str
    formula: str
    ml_risk_score: float
    label_compatible: bool
    raises_flow_alert: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "final_classification": self.final,
            "behaviour_result": self.behaviour_result,
            "detection_reason": self.reason,
            "ml_risk_score": round(float(self.ml_risk_score), 2),
            "ml_family": self.ml_family,
            "risk_score": round(float(self.risk_score), 2),
            "risk_level": self.risk_level,
            "formula": self.formula,
        }


class Correlator:
    """Layer 3: correlation and risk.  Pure logic, with no I/O."""

    def __init__(
        self,
        risk_scorer: RiskScorer,
        config: Mapping[str, Any] | None = None,
        *,
        alert_min_band: str = "MEDIUM",
    ) -> None:
        config = config or {}
        self.risk_scorer = risk_scorer
        self.alert_min_band = str(alert_min_band)
        self.ml_attack_min_confidence = float(config.get("ml_attack_min_confidence", 0.90))
        self.tcp_trained_labels = set(config.get("tcp_trained_labels") or DEFAULT_TCP_TRAINED_LABELS)
        self.families = {**DEFAULT_FAMILIES, **dict(config.get("families") or {})}
        self.behaviour_severity = {
            **DEFAULT_BEHAVIOUR_SEVERITY, **dict(config.get("behaviour_severity") or {})
        }
        self.corroboration_bonus = float(config.get("corroboration_bonus", 10.0))
        self.repeat_bonus_per_event = float(config.get("repeat_bonus_per_event", 3.0))
        self.repeat_bonus_cap = float(config.get("repeat_bonus_cap", 15.0))

    # ------------------------------------------------------------------ risk
    def behaviour_risk(self, finding: Finding) -> tuple[float, str]:
        """Risk of one finding: severity scaled up by how far past its threshold it is."""
        severity = float(self.behaviour_severity.get(finding.kind, 0.5))
        base = 100.0 * severity * (0.6 + 0.4 * float(finding.strength))
        bonus = min(self.repeat_bonus_cap, self.repeat_bonus_per_event * (finding.repeats - 1))
        score = min(100.0, base + bonus)
        formula = (
            f"100 * severity[{finding.kind}]={severity:.2f} * "
            f"(0.6 + 0.4*strength={finding.strength:.2f}) + repeat bonus {bonus:.0f} "
            f"= {score:.1f}"
        )
        return round(score, 2), formula

    def compatible(self, label: str, protocol_name: str | None) -> bool:
        """Could the model have learned this label for this protocol?"""
        protocol = str(protocol_name or "").upper()
        return not (protocol in NON_TCP_PROTOCOLS and label in self.tcp_trained_labels)

    # ------------------------------------------------------------ the decision
    def decide(
        self,
        *,
        ml_label: str,
        ml_is_attack: bool,
        confidence: float,
        ml_risk_score: float,
        ml_risk_formula: str,
        protocol_name: str | None,
        findings: Sequence[Finding] = (),
    ) -> Decision:
        family = self.families.get(ml_label, "Attack")
        compatible = self.compatible(ml_label, protocol_name) if ml_is_attack else True
        supported = bool(ml_is_attack and compatible)

        ml_category = BENIGN
        if supported:
            ml_category = (
                ATTACK if float(confidence) >= self.ml_attack_min_confidence else SUSPICIOUS
            )
        behaviour_category = BENIGN
        strongest: Finding | None = None
        for finding in findings:
            if strongest is None or _ORDER[finding.decision] > _ORDER[strongest.decision]:
                strongest = finding
            if _ORDER[finding.decision] > _ORDER[behaviour_category]:
                behaviour_category = finding.decision
        final = max(ml_category, behaviour_category, key=_ORDER.__getitem__)

        behaviour_result = "None"
        if findings:
            behaviour_result = "; ".join(sorted({f.result_text for f in findings}))

        reason = self._reason(
            ml_label, supported, compatible, confidence, protocol_name, findings, final
        )
        score, level, formula = self._risk(
            final, supported, ml_risk_score, ml_risk_formula, findings, compatible, confidence
        )
        return Decision(
            final=final,
            ml_label=ml_label,
            ml_family=family,
            behaviour_result=behaviour_result,
            reason=reason,
            risk_score=score,
            risk_level=level,
            formula=formula,
            ml_risk_score=float(ml_risk_score) if supported else 0.0,
            label_compatible=compatible,
            # A flow that is only part of a behavioural finding is covered by the alert
            # for that finding, so a scan does not raise one alert per probe.
            raises_flow_alert=(
                supported and RiskScorer.at_least(level, self.alert_min_band)
            ),
        )

    def _risk(self, final, supported, ml_score, ml_formula, findings, compatible, confidence):
        if final == BENIGN:
            if not compatible:
                return 0.0, self.risk_scorer.band(0.0), (
                    "model label not supported for this protocol -> risk 0"
                )
            return round(float(ml_score), 2), self.risk_scorer.band(float(ml_score)), ml_formula

        ml_part = float(ml_score) if supported else 0.0
        behaviour_part, behaviour_formula = 0.0, ""
        for finding in findings:
            value, text = self.behaviour_risk(finding)
            if value > behaviour_part:
                behaviour_part, behaviour_formula = value, text
        bonus = self.corroboration_bonus if ml_part > 0 and behaviour_part > 0 else 0.0
        score = min(100.0, max(ml_part, behaviour_part) + bonus)
        parts = []
        if ml_part:
            parts.append(f"model: {ml_formula}")
        if behaviour_part:
            parts.append(f"behaviour: {behaviour_formula}")
        formula = " | ".join(parts) + f" | final = max + corroboration {bonus:.0f} = {score:.1f}"
        return round(score, 2), self.risk_scorer.band(score), formula

    def _reason(self, label, supported, compatible, confidence, protocol, findings, final) -> str:
        parts: list[str] = []
        if not compatible:
            parts.append(
                f"Model label '{label}' not used: every '{label}' example in its training "
                f"data was TCP, and this flow is {protocol}."
            )
        elif supported:
            source = "alone" if not findings else "with behaviour"
            parts.append(f"Model labelled this flow '{label}' (score {float(confidence):.0%}, {source}).")
        for finding in findings:
            parts.append(finding.reason)
        if not parts:
            return "No model or behavioural indicator."
        if final == BENIGN:
            parts.append("No behavioural indicator supports an attack.")
        return " ".join(parts)


__all__ = [
    "BENIGN",
    "Correlator",
    "DEFAULT_TCP_TRAINED_LABELS",
    "Decision",
    "NON_TCP_PROTOCOLS",
]
