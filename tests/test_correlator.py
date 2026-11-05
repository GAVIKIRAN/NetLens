"""Layer 3 decisions, the monitor path, the database migration and the padding fix."""

from __future__ import annotations

import sqlite3
import time

import pytest

from src.database.database import Database
from src.detection.behavior import ATTACK, PORT_SCAN, SUSPICIOUS, Finding
from src.detection.correlator import BENIGN, Correlator
from src.inference.risk import RiskScorer
from src.network.flow_manager import SYN, TCP, PacketInfo
from src.network.monitor import MODE_PCAP, MonitorService
from src.simulation.scenarios import SCENARIOS, build_packets

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
A, B = "192.168.1.44", "192.168.1.35"


def _scan(decision=ATTACK, strength=1.0, repeats=1):
    return Finding(PORT_SCAN, decision, A, B, 0, "Port scan: 120 ports.", strength, repeats=repeats)


@pytest.fixture
def correlator():
    return Correlator(RiskScorer(severity_map={"DoS Hulk": 0.75}))


def _decide(correlator, *, label="BENIGN", attack=False, confidence=0.99, risk=0.0,
            protocol="TCP", findings=()):
    return correlator.decide(
        ml_label=label, ml_is_attack=attack, confidence=confidence, ml_risk_score=risk,
        ml_risk_formula="ml formula", protocol_name=protocol, findings=findings,
    )


def test_benign_with_no_behaviour_stays_benign(correlator):
    decision = _decide(correlator)
    assert decision.final == BENIGN and not decision.raises_flow_alert


def test_scan_overrides_decision_but_not_the_model_label(correlator):
    decision = _decide(correlator, findings=[_scan()])
    assert decision.final == ATTACK
    assert decision.ml_label == "BENIGN"
    assert "Port scan" in decision.behaviour_result
    assert "Port scan: 120 ports." in decision.reason


def test_flow_in_a_finding_does_not_raise_its_own_alert(correlator):
    assert not _decide(correlator, findings=[_scan()]).raises_flow_alert


def test_tcp_attack_label_on_udp_is_recorded_but_not_used(correlator):
    decision = _decide(correlator, label="DoS Slowhttptest", attack=True, confidence=0.84,
                       risk=62.0, protocol="UDP")
    assert decision.final == BENIGN
    assert decision.ml_label == "DoS Slowhttptest"
    assert decision.risk_score == 0.0 and not decision.label_compatible
    assert "UDP" in decision.reason


def test_udp_with_behavioural_evidence_is_not_excused(correlator):
    decision = _decide(correlator, label="DDoS", attack=True, risk=80.0, protocol="UDP",
                       findings=[_scan(SUSPICIOUS)])
    assert decision.final == SUSPICIOUS


def test_unknown_protocol_is_not_treated_as_a_mismatch(correlator):
    decision = _decide(correlator, label="DoS Hulk", attack=True, risk=80.0, protocol="OTHER")
    assert decision.final == ATTACK and decision.label_compatible


def test_low_score_model_label_alone_is_only_suspicious(correlator):
    decision = _decide(correlator, label="DoS Hulk", attack=True, confidence=0.7, risk=60.0)
    assert decision.final == SUSPICIOUS and decision.raises_flow_alert


def test_agreement_adds_a_stated_bonus(correlator):
    alone = _decide(correlator, label="DoS Hulk", attack=True, risk=60.0)
    both = _decide(correlator, label="DoS Hulk", attack=True, risk=60.0, findings=[_scan()])
    assert both.risk_score > alone.risk_score
    assert "corroboration 10" in both.formula


def test_repeating_raises_risk_up_to_the_cap(correlator):
    one, _ = correlator.behaviour_risk(_scan(strength=0.5))
    two, _ = correlator.behaviour_risk(_scan(strength=0.5, repeats=2))
    many, _ = correlator.behaviour_risk(_scan(strength=0.5, repeats=500))
    assert one < two < many
    assert many - one == pytest.approx(15.0)


def test_risk_formula_is_shown(correlator):
    assert "severity[port_scan]" in _decide(correlator, findings=[_scan()]).formula


# ------------------------------------------------------------------ monitor path
@pytest.fixture
def monitor(settings, predictor, database):
    return MonitorService(settings, database=database, predictor=predictor)


def test_scan_of_lone_syn_probes_reaches_database_and_alerts(monitor, database):
    monitor.process_packets(build_packets(BY_NAME["scan_filtered"], 1))
    events = database.recent_behaviour_events()
    assert (events["kind"] == PORT_SCAN).any()
    assert (events["decision"] == ATTACK).any()
    alerts = database.recent_alerts()
    assert alerts["attack_type"].str.contains("behaviour").any()
    assert alerts["prediction_id"].isna().any()
    assert monitor.status()["behaviour"]["connection_attempts"] > 200


def test_ssdp_is_not_alerted_on_whatever_the_model_says(monitor):
    monitor.process_packets(build_packets(BY_NAME["normal_ssdp"], 1))
    assert monitor.stats.flows_processed > 0
    assert all(row["final"] == BENIGN for row in monitor.recent)
    assert monitor.stats.alerts_raised == 0


def test_model_output_stays_in_its_own_column(monitor, database):
    monitor.process_packets(build_packets(BY_NAME["scan_closed"], 1))
    frame = database.recent_predictions(1000)
    assert not frame.empty
    assert {"final_classification", "behaviour_result", "detection_reason"} <= set(frame.columns)
    assert frame["prediction"].notna().all()


def test_pcap_mode_runs_every_layer_on_the_worker(monitor, database, tmp_path):
    from scapy.all import Ether, IP, TCP as ScapyTCP, wrpcap

    frames = []
    for index in range(40):
        frame = Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / IP(src=A, dst=B) / ScapyTCP(
            sport=40000 + index, dport=index + 1, flags="S"
        )
        frame.time = 1_700_000_000.0 + index * 0.4
        frames.append(frame)
    path = tmp_path / "scan.pcap"
    wrpcap(str(path), frames)

    monitor.start(MODE_PCAP, pcap_path=path)
    deadline = time.time() + 60
    while monitor.is_running and time.time() < deadline:
        time.sleep(0.05)
    assert not monitor.is_running
    assert (database.recent_behaviour_events()["kind"] == PORT_SCAN).any()


def test_full_queue_drops_and_counts_instead_of_blocking(settings, predictor, database):
    service = MonitorService(settings, database=database, predictor=predictor)
    service._worker_thread = object()  # pretend a worker exists, so nothing drains
    for _ in range(service._queue.maxsize + 5):
        service._submit("flow", {}, {})
    service._worker_thread = None
    assert service.stats.records_dropped == 5


# ---------------------------------------------------------------- database
def test_old_database_is_migrated_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, flow_id INTEGER,
            timestamp TEXT NOT NULL, model TEXT, prediction TEXT NOT NULL,
            is_attack INTEGER NOT NULL DEFAULT 0, confidence REAL NOT NULL DEFAULT 0,
            argmax_label TEXT, argmax_confidence REAL, threshold_applied INTEGER NOT NULL DEFAULT 0,
            risk_score REAL NOT NULL DEFAULT 0, risk_level TEXT NOT NULL DEFAULT 'LOW',
            risk_formula TEXT, probabilities_json TEXT, top_features_json TEXT, explanation TEXT,
            explainer TEXT, mode TEXT NOT NULL DEFAULT 'LIVE', ground_truth TEXT);
        INSERT INTO predictions (timestamp, prediction, is_attack) VALUES ('t', 'DDoS', 1);
        """
    )
    connection.commit()
    connection.close()

    store = Database(path)
    frame = store.recent_predictions()
    assert frame.iloc[0]["prediction"] == "DDoS"
    assert frame.iloc[0]["final_classification"] is None
    assert store.statistics()["attack_flows"] == 1  # old rows fall back to is_attack
    store.close()
    store = Database(path)  # migrating twice is safe
    store.close()


def test_final_decision_drives_the_attack_count(database):
    flow = database.insert_flow({"src_ip": A}, {"flow_duration": 1.0})
    database.insert_prediction(flow, {"prediction": "DoS Slowhttptest", "is_attack": True,
                                      "final_classification": "BENIGN"})
    assert database.statistics()["attack_flows"] == 0


def test_unmeasured_feature_is_stored_as_null_not_zero(database):
    flow = database.insert_flow({"src_ip": A}, {"flow_bytes_s": float("nan"), "flow_duration": 5.0})
    payload = database.insert_prediction(flow, {"prediction": "BENIGN"})
    record = database.get_prediction(payload)
    assert record["features"]["flow_bytes_s"] is None
    assert record["features"]["flow_duration"] == 5.0


# -------------------------------------------------------------- feature fixes
def test_ethernet_padding_is_not_counted_as_payload():
    from scapy.all import Ether, IP, TCP as ScapyTCP

    from src.network.feature_extractor import packet_to_info

    base = Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / IP(src=B, dst=A) / ScapyTCP(
        sport=81, dport=5, flags="RA", window=0
    )
    assert packet_to_info(Ether(bytes(base) + b"\x00" * 6)).payload_len == 0
    assert packet_to_info(Ether(bytes(base / b"abc") + b"\x00" * 3)).payload_len == 3


def test_missing_feature_is_an_error_not_a_zero():
    from src.network.feature_extractor import features_to_array

    with pytest.raises(KeyError):
        features_to_array({"a": 1.0}, ["a", "b"])


def test_feature_order_does_not_depend_on_input_order(predictor):
    import numpy as np

    from src.network.feature_extractor import flow_to_features
    from src.network.flow_manager import Flow

    flow = Flow(PacketInfo(timestamp=0.0, src_ip=A, dst_ip=B, src_port=1, dst_port=2,
                           protocol=TCP, flags=SYN, window=1024, header_len=20))
    features = flow_to_features(flow)
    reordered = dict(reversed(list(features.items())))
    assert np.array_equal(predictor.transform([features]), predictor.transform([reordered]),
                          equal_nan=True)
