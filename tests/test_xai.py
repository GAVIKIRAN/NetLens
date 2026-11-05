"""SHAP, LIME and the human-readable explanation engine."""

from __future__ import annotations

import pytest

from src.xai.explanation_engine import (
    CAUSATION_SAFE_VERBS,
    build_explanation,
    format_value,
)


def test_value_formatting_uses_contract_units():
    assert format_value("flow_duration", 2_500_000) == "2.50 s"
    assert format_value("flow_duration", 1500) == "1.50 ms"
    assert format_value("total_length_of_fwd_packets", 2048).endswith("KB")
    assert format_value("flow_packets_s", 1234.5) == "1,234.5 packets/s"
    assert format_value("protocol", 6) == "TCP"
    assert format_value("total_fwd_packets", None) == "n/a"


def test_narrative_uses_attribution_language_not_causation():
    explanation = build_explanation(
        "DDoS",
        0.94,
        [
            {"feature": "flow_packets_s", "label": "Flow Packets/s",
             "value": 4200.0, "contribution": 0.31},
            {"feature": "flow_duration", "label": "Flow Duration",
             "value": 150000.0, "contribution": -0.12},
        ],
        risk_score=88.0,
        risk_level="CRITICAL",
        explainer="TreeExplainer",
    )
    text = explanation.text().lower()
    assert "contributed to the model's prediction" in text
    assert "caused the attack" not in text
    assert any(verb in text for verb in CAUSATION_SAFE_VERBS)
    assert "flow packets/s" in text
    assert explanation.positive and explanation.negative


def test_narrative_is_generated_from_the_supplied_contributions():
    first = build_explanation(
        "PortScan", 0.8,
        [{"feature": "syn_flag_count", "label": "SYN Flag Count",
          "value": 1, "contribution": 0.9}],
    )
    second = build_explanation(
        "PortScan", 0.8,
        [{"feature": "flow_iat_min", "label": "Flow IAT Min",
          "value": 50, "contribution": 0.9}],
    )
    assert first.text() != second.text()
    assert "SYN Flag Count" in first.text()
    assert "Flow IAT Min" in second.text()


def test_narrative_handles_no_contributions():
    explanation = build_explanation("BENIGN", 0.99, [])
    assert explanation.bullet_points
    assert "combination of many small contributions" in explanation.text()


def test_threshold_note_is_included_when_relevant():
    explanation = build_explanation(
        "BENIGN", 0.55, [], threshold_applied=True
    )
    assert "below the configured decision threshold" in explanation.text()


# ---------------------------------------------------------------------------
#  Explainers against the real trained model
# ---------------------------------------------------------------------------
def test_explanation_service_reports_its_status(explanation_service):
    status = explanation_service.status()
    assert status["enabled"] is True
    assert status["shap_available"] is True
    assert status["shap_explainer"] in ("TreeExplainer", "LinearExplainer", "KernelExplainer")


def test_shap_explains_a_prediction(explanation_service, predictor, synthetic_flows):
    result = predictor.predict_one(synthetic_flows[0].features)
    payload = explanation_service.explain_prediction(result, use_shap=True, use_lime=False)
    shap_payload = payload["shap"]
    assert shap_payload is not None
    assert shap_payload["contributions"]
    entry = shap_payload["contributions"][0]
    assert entry["feature"] in predictor.input_features
    assert entry["direction"] in ("increases", "decreases")
    assert payload["narrative"]["text"]


def test_lime_explains_a_prediction(explanation_service, predictor, synthetic_flows):
    result = predictor.predict_one(synthetic_flows[1].features)
    payload = explanation_service.explain_prediction(result, use_shap=False, use_lime=True)
    lime_payload = payload["lime"]
    assert lime_payload is not None
    assert lime_payload["contributions"]
    assert all(
        entry["feature"] in predictor.input_features
        for entry in lime_payload["contributions"]
    )


def test_shap_contributions_are_ranked_by_magnitude(explanation_service, predictor,
                                                    synthetic_flows):
    result = predictor.predict_one(synthetic_flows[2].features)
    payload = explanation_service.explain_prediction(result, use_shap=True, use_lime=False)
    magnitudes = [entry["abs_contribution"] for entry in payload["shap"]["contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_global_importance(explanation_service):
    if not explanation_service.shap_available:
        pytest.skip("SHAP unavailable")
    ranking = explanation_service.shap.global_importance(top_k=5)
    assert len(ranking) == 5
    assert all(entry["importance"] >= 0 for entry in ranking)


def test_explanations_never_crash_the_caller(explanation_service, predictor,
                                             synthetic_flows):
    """A broken explainer degrades to an empty explanation, never an exception."""
    original = explanation_service.shap
    try:
        explanation_service.shap = None
        result = predictor.predict_one(synthetic_flows[0].features)
        payload = explanation_service.explain_prediction(result, use_shap=True, use_lime=False)
        assert payload["shap"] is None
        assert payload["narrative"]["text"]
    finally:
        explanation_service.shap = original
