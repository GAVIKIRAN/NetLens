"""Behaviour detection: each normal scenario stays quiet and each pattern is found."""

from __future__ import annotations

import pytest

from src.detection.behavior import (
    ATTACK,
    CONNECTION_RATE,
    FAILED_CONNECTIONS,
    HOST_SWEEP,
    PORT_SCAN,
    REPEATED_ATTEMPTS,
    SUSPICIOUS,
    SYN_FLOOD,
    BehaviorDetector,
)
from src.network.flow_manager import ACK, RST, SYN, TCP, PacketInfo
from src.simulation.scenarios import SCENARIOS, build_packets

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
A, B = "192.168.1.44", "192.168.1.35"


def _run(packets, config=None):
    found = []
    detector = BehaviorDetector(config, on_finding=found.append)
    for packet in packets:
        detector.observe(packet)
    if packets:
        detector.tick(packets[-1].timestamp + 4.0)
    return detector, found


def _kinds(found):
    return {finding.kind for finding in found}


def _syn(ts, sport, dport, dst=B):
    return PacketInfo(timestamp=ts, src_ip=A, dst_ip=dst, src_port=sport, dst_port=dport,
                      protocol=TCP, flags=SYN, window=64240, header_len=40)


@pytest.mark.parametrize(
    "name",
    [
        "normal_http", "normal_https", "normal_dns", "normal_ssdp",
        "many_answered_ports", "flaky_service", "browser_burst", "large_download",
    ],
)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_normal_traffic_raises_nothing(name, seed):
    _, found = _run(build_packets(BY_NAME[name], seed))
    assert found == []


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_filtered_port_scan_is_detected_from_single_syn_probes(seed):
    _, found = _run(build_packets(BY_NAME["scan_filtered"], seed))
    scans = [f for f in found if f.kind == PORT_SCAN]
    assert scans and scans[-1].decision == ATTACK
    assert scans[-1].dst_ip == B and scans[-1].src_ip == A


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_closed_port_scan_is_detected(seed):
    _, found = _run(build_packets(BY_NAME["scan_closed"], seed))
    assert any(f.kind == PORT_SCAN and f.decision == ATTACK for f in found)


def test_small_scan_is_only_suspicious_and_reason_gives_the_count():
    _, found = _run(build_packets(BY_NAME["scan_small"], 1))
    scan = next(f for f in found if f.kind == PORT_SCAN)
    assert scan.decision == SUSPICIOUS
    assert "unique destination ports" in scan.reason
    assert str(scan.indicators["unique_ports"]) in scan.reason


def test_repeated_failures_are_found_but_not_called_a_scan():
    _, found = _run(build_packets(BY_NAME["repeated_failures"], 1))
    assert FAILED_CONNECTIONS in _kinds(found)
    assert PORT_SCAN not in _kinds(found)


def test_repeated_successes_are_found():
    _, found = _run(build_packets(BY_NAME["repeated_successes"], 1))
    assert REPEATED_ATTEMPTS in _kinds(found)


def test_high_connection_rate_is_found_but_not_a_scan():
    _, found = _run(build_packets(BY_NAME["high_connection_rate"], 1))
    assert CONNECTION_RATE in _kinds(found)
    assert PORT_SCAN not in _kinds(found)


def test_syn_flood_pattern_is_found():
    _, found = _run(build_packets(BY_NAME["syn_flood_pattern"], 1))
    assert any(f.kind == SYN_FLOOD and f.decision == ATTACK for f in found)


def test_host_sweep_is_found():
    packets = [_syn(i * 0.5, 40000 + i, 445, dst=f"192.168.1.{i + 1}") for i in range(40)]
    _, found = _run(packets)
    assert HOST_SWEEP in _kinds(found)
    assert PORT_SCAN not in _kinds(found)


def test_answered_ports_are_not_a_scan_even_when_there_are_many():
    packets = []
    for index in range(60):
        packets.append(_syn(index * 0.1, 40000 + index, 1000 + index))
        packets.append(PacketInfo(timestamp=index * 0.1 + 0.001, src_ip=B, dst_ip=A,
                                  src_port=1000 + index, dst_port=40000 + index, protocol=TCP,
                                  flags=SYN | ACK, window=65535))
    _, found = _run(packets)
    assert PORT_SCAN not in _kinds(found)


def test_probes_are_not_failures_until_the_timeout_passes():
    found = []
    detector = BehaviorDetector(on_finding=found.append)
    for index in range(30):
        detector.observe(_syn(index * 0.01, 40000 + index, index + 1))
    detector.tick(0.5)
    assert found == []
    detector.tick(10.0)
    assert any(f.kind == PORT_SCAN for f in found)


def test_retransmitted_syn_is_one_attempt():
    detector, _ = _run([_syn(0.0, 40000, 80), _syn(1.0, 40000, 80), _syn(3.0, 40000, 80)])
    assert detector.attempts_seen == 1


def test_thresholds_come_from_config():
    packets = [_syn(i * 0.1, 40000 + i, i + 1) for i in range(30)]
    _, found = _run(packets, {"port_scan": {"min_unique_ports": 50}})
    assert PORT_SCAN not in _kinds(found)
    _, found = _run(packets, {"port_scan": {"min_unique_ports": 10, "attack_unique_ports": 20}})
    assert any(f.kind == PORT_SCAN and f.decision == ATTACK for f in found)


def test_old_attempts_leave_the_window():
    packets = [_syn(i * 15.0, 40000 + i, i + 1) for i in range(60)]
    _, found = _run(packets)
    assert PORT_SCAN not in _kinds(found)


def test_a_finding_matches_only_flows_in_the_pattern():
    detector, found = _run(build_packets(BY_NAME["scan_filtered"], 1))
    last_seen = found[-1].last_seen
    assert detector.findings_for(A, B, 1234, at=last_seen)
    assert not detector.findings_for(A, "8.8.8.8", 443, at=last_seen)
    assert not detector.findings_for("192.168.1.99", B, 1234, at=last_seen)


def test_tracked_sources_are_limited():
    detector = BehaviorDetector({"max_sources": 10})
    for index in range(100):
        detector.observe(PacketInfo(timestamp=float(index), src_ip=f"10.0.0.{index}", dst_ip=B,
                                    src_port=1, dst_port=80, protocol=TCP, flags=ACK))
    assert detector.stats()["tracked_sources"] == 10
    assert detector.stats()["sources_evicted"] == 90


def test_consumer_error_does_not_reach_capture():
    def explode(_finding):
        raise RuntimeError("boom")

    detector = BehaviorDetector(on_finding=explode)
    for index in range(40):
        detector.observe(_syn(index * 0.1, 40000 + index, index + 1))
    detector.tick(100.0)


def test_rst_from_the_scanner_does_not_count_as_refused():
    packets = [
        _syn(0.0, 40000, 80),
        PacketInfo(timestamp=0.01, src_ip=B, dst_ip=A, src_port=80, dst_port=40000, protocol=TCP,
                   flags=SYN | ACK, window=65535),
        PacketInfo(timestamp=0.02, src_ip=A, dst_ip=B, src_port=40000, dst_port=80, protocol=TCP,
                   flags=RST | ACK, window=0),
    ]
    detector, _ = _run(packets)
    assert detector.stats()["pending_attempts"] == 0
