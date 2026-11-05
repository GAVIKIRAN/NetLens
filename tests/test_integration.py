"""End-to-end integration.

    CIC-IDS2017-shaped CSV
        -> load -> clean -> profile -> split
        -> preprocessing pipeline -> model -> prediction
        -> risk score -> SHAP/LIME -> human-readable explanation
        -> SQLite (flow, prediction, alert)

and the two runtime paths that feed it: synthetic packets through the live flow
meter, and the DEMO replay engine.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.database.database import Database
from src.network.feature_extractor import flow_to_features, synthetic_packets
from src.network.flow_manager import FlowManager
from src.network.monitor import MODE_DEMO, MonitorService


def test_full_pipeline_from_packets_to_database(settings, predictor,
                                                explanation_service, tmp_path):
    # 1. packets -> flow -> contract features -----------------------------
    manager = FlowManager(flow_timeout=1e9)
    manager.add_packets(
        synthetic_packets(src_ip="192.168.10.20", dst_ip="203.0.113.8",
                          src_port=50505, dst_port=80, count=40,
                          interval=0.002, payload=120)
    )
    flow = manager.flush()[0]
    features = flow_to_features(flow)
    identity = flow.identity()
    assert len(features) >= 60

    # 2. preprocessing + model -------------------------------------------
    result = predictor.predict_one(features)
    assert result.predicted_label in predictor.classes
    assert np.isfinite(result.transformed).all()

    # 3. risk -------------------------------------------------------------
    assert result.risk is not None
    assert 0.0 <= result.risk.score <= 100.0
    assert result.risk.formula

    # 4. XAI --------------------------------------------------------------
    payload = explanation_service.explain_prediction(result, use_shap=True, use_lime=True)
    assert payload["narrative"]["text"]
    assert payload["shap"] is not None

    # 5. persistence ------------------------------------------------------
    database = Database(tmp_path / "integration.db")
    try:
        flow_id = database.insert_flow(identity, features, mode="TEST")
        prediction_payload = result.to_dict()
        prediction_payload["formula"] = result.risk.formula
        prediction_payload["top_features"] = payload["top_features"]
        prediction_payload["explanation"] = payload["narrative"]["text"]
        prediction_id = database.insert_prediction(flow_id, prediction_payload, mode="TEST")
        database.insert_alert(prediction_id, flow_id, identity, prediction_payload,
                              mode="TEST")

        stored = database.get_prediction(prediction_id)
        assert stored["prediction"] == result.predicted_label
        assert stored["features"]["flow_duration"] == pytest.approx(
            features["flow_duration"]
        )
        assert stored["explanation"]
        assert database.counts()["alerts"] == 1
    finally:
        database.close()


def test_demo_mode_runs_end_to_end(settings, predictor, demo_file, tmp_path):
    database = Database(tmp_path / "demo.db")
    monitor = MonitorService(settings, database=database, predictor=predictor)
    try:
        monitor.start(MODE_DEMO, replay_rate=60.0)
        deadline = time.time() + 12
        while time.time() < deadline and monitor.stats.flows_processed < 20:
            time.sleep(0.25)
        monitor.stop()

        assert monitor.stats.flows_processed >= 10
        assert monitor.stats.errors == 0
        counts = database.counts()
        assert counts["flows"] == monitor.stats.flows_processed
        assert counts["predictions"] == monitor.stats.flows_processed

        recent = database.recent_predictions(10)
        assert not recent.empty
        assert recent["ground_truth"].notna().any()  # demo labels are stored
        assert monitor.status()["mode"] == "IDLE"
        assert monitor.status()["running"] is False
    finally:
        monitor.stop()
        database.close()


def test_demo_burst_classifies_without_starting_capture(settings, predictor, demo_file,
                                                        tmp_path):
    database = Database(tmp_path / "burst.db")
    monitor = MonitorService(settings, database=database, predictor=predictor)
    try:
        processed = monitor.demo_burst(15)
        assert processed == 15
        assert database.counts()["predictions"] == 15
        assert not monitor.is_running
    finally:
        database.close()


def test_monitor_refuses_to_start_without_a_model(settings, tmp_path):
    from src.config.settings import DEFAULT_CONFIG_PATH, Settings
    from src.utils.errors import ModelError

    empty = Settings.load(
        DEFAULT_CONFIG_PATH,
        overrides={
            "paths": {
                "model_dir": str(tmp_path / "no_models"),
                "preprocessing_dir": str(tmp_path / "no_prep"),
                "database_path": str(tmp_path / "empty.db"),
                "log_dir": str(tmp_path / "logs"),
            },
            "logging": {"console": False, "file": False},
        },
        use_env=False,
    )
    empty.ensure_directories()
    monitor = MonitorService(empty)
    with pytest.raises(ModelError):
        monitor.start(MODE_DEMO)


def test_artefacts_are_self_consistent(settings, predictor):
    """The persisted pipeline, label encoder and model must agree."""
    metadata = predictor.metadata
    assert metadata["classes"] == predictor.classes
    assert metadata["input_features"] == predictor.input_features
    assert metadata["output_features"] == predictor.output_features
    n_features = getattr(predictor.model, "n_features_in_", None)
    if n_features is not None:
        assert int(n_features) == len(predictor.output_features)


def test_pipeline_is_reproducible(settings, predictor, synthetic_flows):
    """Two identical records must always produce identical output."""
    record = synthetic_flows[0].features
    first = predictor.predict_one(record)
    second = predictor.predict_one(dict(record))
    assert first.predicted_label == second.predicted_label
    assert first.confidence == pytest.approx(second.confidence)
    assert first.risk.score == pytest.approx(second.risk.score)
