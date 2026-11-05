"""SQLite persistence."""

from __future__ import annotations

import threading

import pytest

from src.database.database import Database
from src.database.schema import TABLES

IDENTITY = {
    "src_ip": "192.168.10.5", "dst_ip": "203.0.113.9", "src_port": 51000,
    "dst_port": 443, "protocol": 6, "protocol_name": "TCP",
    "duration_seconds": 1.25, "packets": 12, "bytes": 4096,
    "start_time": 100.0, "end_time": 101.25, "termination": "fin",
}
PAYLOAD = {
    "model": "xgboost", "prediction": "DDoS", "is_attack": True, "confidence": 0.93,
    "argmax_label": "DDoS", "argmax_confidence": 0.93, "threshold_applied": False,
    "risk_score": 88.4, "risk_level": "CRITICAL", "formula": "100 * (...)",
    "probabilities": {"BENIGN": 0.07, "DDoS": 0.93},
    "top_features": [{"feature": "flow_packets_s", "contribution": 0.41}],
    "explanation": "test explanation", "explainer": "TreeExplainer",
}


def test_schema_is_created(database):
    counts = database.counts()
    assert set(counts) == set(TABLES)
    assert all(value == 0 for value in counts.values())


def test_insert_and_read_back(database):
    flow_id = database.insert_flow(IDENTITY, {"flow_duration": 1_250_000.0}, mode="DEMO")
    prediction_id = database.insert_prediction(flow_id, PAYLOAD, mode="DEMO",
                                               ground_truth="DDoS")
    alert_id = database.insert_alert(prediction_id, flow_id, IDENTITY, PAYLOAD, mode="DEMO")
    assert flow_id > 0 and prediction_id > 0 and alert_id > 0

    record = database.get_prediction(prediction_id)
    assert record["prediction"] == "DDoS"
    assert record["ground_truth"] == "DDoS"
    assert record["features"]["flow_duration"] == 1_250_000.0
    assert record["probabilities"]["DDoS"] == 0.93
    assert record["top_features"][0]["feature"] == "flow_packets_s"


def test_statistics_and_distributions(database):
    flow_id = database.insert_flow(IDENTITY, {}, mode="DEMO")
    database.insert_prediction(flow_id, PAYLOAD, mode="DEMO")
    benign = dict(PAYLOAD, prediction="BENIGN", is_attack=False, risk_level="LOW",
                  risk_score=0.5)
    database.insert_prediction(flow_id, benign, mode="DEMO")

    stats = database.statistics()
    assert stats["total_flows"] == 2
    assert stats["attack_flows"] == 1
    assert stats["attack_percentage"] == 50.0

    distribution = database.attack_distribution()
    assert list(distribution["attack_type"]) == ["DDoS"]
    assert not database.class_distribution().empty
    assert not database.timeline("minute").empty
    assert not database.protocol_distribution().empty
    assert not database.top_talkers().empty


def test_alert_filtering(database):
    flow_id = database.insert_flow(IDENTITY, {}, mode="DEMO")
    prediction_id = database.insert_prediction(flow_id, PAYLOAD, mode="DEMO")
    database.insert_alert(prediction_id, flow_id, IDENTITY, PAYLOAD, mode="DEMO")
    database.insert_alert(
        prediction_id, flow_id, IDENTITY,
        dict(PAYLOAD, risk_level="MEDIUM", prediction="PortScan"), mode="LIVE",
    )
    assert len(database.recent_alerts(10)) == 2
    assert len(database.recent_alerts(10, severities=["CRITICAL"])) == 1
    assert len(database.recent_alerts(10, attack_types=["PortScan"])) == 1
    assert len(database.recent_alerts(10, mode="LIVE")) == 1


def test_acknowledge_and_update_explanation(database):
    flow_id = database.insert_flow(IDENTITY, {}, mode="DEMO")
    prediction_id = database.insert_prediction(flow_id, PAYLOAD, mode="DEMO")
    alert_id = database.insert_alert(prediction_id, flow_id, IDENTITY, PAYLOAD)
    database.acknowledge_alert(alert_id, True)
    assert int(database.recent_alerts(1).iloc[0]["acknowledged"]) == 1

    database.update_prediction_explanation(prediction_id, "new text", [{"feature": "x"}], "LIME")
    record = database.get_prediction(prediction_id)
    assert record["explanation"] == "new text"
    assert record["explainer"] == "LIME"


def test_system_events(database):
    database.log_event("INFO", "test", "hello", {"k": 1})
    events = database.recent_events(5)
    assert len(events) == 1
    assert events.iloc[0]["component"] == "test"


def test_clear_and_prune(database):
    flow_id = database.insert_flow(IDENTITY, {}, mode="DEMO")
    database.insert_prediction(flow_id, PAYLOAD, mode="DEMO")
    assert database.prune(0) == 0                     # disabled
    assert database.prune(3650) == 0                  # nothing is that old
    database.clear()
    assert all(value == 0 for value in database.counts().values())


def test_writes_from_another_thread(database):
    errors: list[Exception] = []

    def writer():
        try:
            for _ in range(20):
                flow_id = database.insert_flow(IDENTITY, {}, mode="DEMO")
                database.insert_prediction(flow_id, PAYLOAD, mode="DEMO")
        except Exception as exc:  # pragma: no cover - the assertion reports it
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    assert database.counts()["predictions"] == 60


def test_no_payload_is_stored(database):
    """Data minimisation: the schema has no column that could hold packet bytes."""
    columns = database.query("PRAGMA table_info(flows)")["name"].tolist()
    for forbidden in ("payload", "packet_bytes", "raw", "content"):
        assert forbidden not in columns
