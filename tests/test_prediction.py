"""Training artefacts, the predictor and the risk engine."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.risk import BANDS, RiskScorer
from src.network.feature_extractor import flow_to_features
from src.utils.errors import FeatureContractError


# ===========================================================================
#  Risk scoring (no model required)
# ===========================================================================
def _scorer() -> RiskScorer:
    return RiskScorer(
        severity_map={"BENIGN": 0.0, "DDoS": 0.85, "PortScan": 0.5},
        weights={"severity": 0.55, "confidence": 0.30, "volume": 0.15},
        default_severity=0.65,
        volume_reference_pps=5000.0,
        bands={"low": 25, "medium": 50, "high": 75},
    )


def test_benign_verdicts_never_leave_the_low_band():
    scorer = _scorer()
    for confidence in (0.0, 0.5, 0.99):
        assessment = scorer.score("BENIGN", confidence, packets_per_second=100000)
        assert assessment.level == "LOW"
        assert assessment.score <= 20.0


def test_risk_is_not_confidence():
    scorer = _scorer()
    high_severity = scorer.score("DDoS", 0.60, packets_per_second=10)
    low_severity = scorer.score("PortScan", 0.99, packets_per_second=10)
    # More confident but less severe scores lower - the two quantities differ.
    assert high_severity.score > low_severity.score
    assert high_severity.confidence < low_severity.confidence


def test_risk_score_is_bounded_and_banded():
    scorer = _scorer()
    for label in ("DDoS", "PortScan", "UnknownAttack"):
        for confidence in (0.0, 0.5, 1.0):
            for pps in (0, 10, 100000):
                assessment = scorer.score(label, confidence, packets_per_second=pps)
                assert 0.0 <= assessment.score <= 100.0
                assert assessment.level in BANDS


def test_unknown_classes_use_the_default_severity():
    scorer = _scorer()
    assert scorer.severity_of("Something New") == 0.65


def test_volume_factor_is_monotonic_and_bounded():
    scorer = _scorer()
    values = [scorer.volume_factor(pps) for pps in (0, 1, 100, 5000, 10**6)]
    assert values == sorted(values)
    assert 0.0 <= values[0] and values[-1] <= 1.0


def test_band_boundaries():
    scorer = _scorer()
    assert scorer.band(24.9) == "LOW"
    assert scorer.band(25.0) == "MEDIUM"
    assert scorer.band(49.9) == "MEDIUM"
    assert scorer.band(50.0) == "HIGH"
    assert scorer.band(75.0) == "CRITICAL"


def test_at_least_ordering():
    assert RiskScorer.at_least("HIGH", "MEDIUM")
    assert not RiskScorer.at_least("LOW", "MEDIUM")
    assert RiskScorer.at_least("CRITICAL", "CRITICAL")


def test_formula_is_reported():
    assessment = _scorer().score("DDoS", 0.9, packets_per_second=100)
    assert "severity" in assessment.formula and "confidence" in assessment.formula


# ===========================================================================
#  Training artefacts
# ===========================================================================
def test_training_produced_every_artefact(settings, training_artifacts):
    model_dir = settings.path("model_dir")
    prep_dir = settings.path("preprocessing_dir")
    for name in ("best_model.joblib", "model_metadata.json",
                 "evaluation_metrics.json", "xai_background.npz"):
        assert (model_dir / name).exists(), name
    for name in ("preprocessing_pipeline.joblib", "feature_contract.json",
                 "label_encoder.joblib"):
        assert (prep_dir / name).exists(), name
    assert (settings.path("reports_dir") / "model_comparison.csv").exists()
    assert (settings.path("reports_dir") / "dataset_profile.json").exists()


def test_metrics_are_measured_not_invented(training_artifacts):
    best = training_artifacts.metrics["best_model"]
    test = best["test"]
    assert 0.0 <= test["accuracy"] <= 1.0
    assert 0.0 <= test["f1_macro"] <= 1.0
    assert test["n_samples"] > 0
    assert len(test["confusion_matrix"]) == len(test["class_names"])
    assert sum(sum(row) for row in test["confusion_matrix"]) == test["n_samples"]


def test_more_than_one_candidate_was_compared(training_artifacts):
    names = {c["name"] for c in training_artifacts.metrics["candidates"]}
    assert len(names) >= 2
    assert training_artifacts.best_model_name in names


def test_selection_used_validation_not_test(training_artifacts):
    selection = training_artifacts.metadata["selection"]
    assert selection["primary_metric"] in (
        "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "detection_rate"
    )
    assert selection["rationale"]
    assert selection["validation_scores"]


# ===========================================================================
#  Predictor
# ===========================================================================
def test_predictor_loads_and_describes_itself(predictor):
    description = predictor.describe()
    assert description["model_name"]
    assert len(description["classes"]) >= 2
    assert description["n_input_features"] == len(predictor.input_features)


def test_predict_one_returns_a_complete_result(predictor, synthetic_flows):
    result = predictor.predict_one(synthetic_flows[0].features)
    assert result.predicted_label in predictor.classes
    assert 0.0 <= result.confidence <= 1.0
    assert result.risk is not None
    assert 0.0 <= result.risk.score <= 100.0
    assert result.transformed is not None
    assert len(result.probabilities) == len(predictor.classes)
    assert result.probabilities[result.argmax_label] == pytest.approx(
        result.argmax_confidence
    )


def test_probabilities_sum_to_one(predictor, synthetic_flows):
    for flow in synthetic_flows[:5]:
        result = predictor.predict_one(flow.features)
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_batch_and_single_predictions_agree(predictor, synthetic_flows):
    records = [flow.features for flow in synthetic_flows[:6]]
    batch = predictor.predict(records)
    for index, record in enumerate(records):
        single = predictor.predict_one(record)
        assert single.predicted_label == batch[index].predicted_label
        assert single.confidence == pytest.approx(batch[index].confidence)


def test_predictor_rejects_records_missing_features(predictor, synthetic_flows):
    broken = dict(synthetic_flows[0].features)
    broken.pop("flow_duration")
    with pytest.raises(FeatureContractError):
        predictor.predict_one(broken)


def test_threshold_forces_benign_when_confidence_is_low(predictor, synthetic_flows):
    original = predictor.attack_threshold
    try:
        # Above any probability: a decision tree's pure leaves can output exactly 1.0.
        predictor.attack_threshold = 1.01
        results = predictor.predict([f.features for f in synthetic_flows])
        assert all(r.predicted_label == predictor.benign_label for r in results)
        assert any(r.threshold_applied for r in results)
    finally:
        predictor.attack_threshold = original


def test_live_flow_features_flow_straight_into_the_predictor(predictor):
    """The live path: packets -> flow -> features -> model, with no adaptation."""
    from src.network.flow_manager import FlowManager
    from src.network.feature_extractor import synthetic_packets

    manager = FlowManager(flow_timeout=1e9)
    manager.add_packets(
        synthetic_packets(src_ip="10.0.0.7", dst_ip="10.0.0.9", src_port=44444,
                          dst_port=443, count=20, interval=0.01, payload=600)
    )
    features = flow_to_features(manager.flush()[0])
    result = predictor.predict_one(features)
    assert result.predicted_label in predictor.classes
    assert np.isfinite(result.transformed).all()


def test_background_matrix_available_for_xai(predictor):
    background = predictor.background_matrix(50)
    assert background.ndim == 2
    assert background.shape[1] == len(predictor.output_features)
