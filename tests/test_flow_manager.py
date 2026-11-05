"""Flow assembly: keys, direction, termination, expiry and safety valves."""

from __future__ import annotations

import pytest

from src.network.feature_extractor import flow_to_features, synthetic_packets
from src.network.flow_manager import (
    ACK, FIN, FlowManager, PSH, PacketInfo, RST, RunningStats, SYN, TCP, UDP,
)


def _packet(ts, src="10.0.0.1", dst="10.0.0.2", sp=1000, dp=80, flags=0, payload=100,
            protocol=TCP, window=8192):
    return PacketInfo(timestamp=ts, src_ip=src, dst_ip=dst, src_port=sp, dst_port=dp,
                      protocol=protocol, payload_len=payload, frame_len=payload + 54,
                      header_len=20 if protocol == TCP else 8, flags=flags, window=window)


def test_running_stats_matches_the_sample_convention():
    stats = RunningStats()
    for value in (1, 2, 3, 4):
        stats.add(value)
    assert stats.n == 4
    assert stats.mean == pytest.approx(2.5)
    assert stats.variance == pytest.approx(1.6666666, rel=1e-5)   # ddof = 1
    assert stats.minimum == 1 and stats.maximum == 4
    single = RunningStats()
    single.add(7)
    assert single.std == 0.0 and single.variance == 0.0


def test_flow_key_is_direction_agnostic():
    forward = _packet(0.0)
    backward = _packet(0.1, src="10.0.0.2", dst="10.0.0.1", sp=80, dp=1000)
    assert forward.flow_key() == backward.flow_key()


def test_both_directions_join_one_flow():
    manager = FlowManager(flow_timeout=1e9)
    manager.add_packet(_packet(0.0, flags=SYN))
    manager.add_packet(_packet(0.1, src="10.0.0.2", dst="10.0.0.1", sp=80, dp=1000, flags=ACK))
    assert manager.active_flows == 1
    flow = manager.flush()[0]
    assert flow.fwd.packets == 1 and flow.bwd.packets == 1


def test_rst_terminates_the_flow_immediately():
    completed = []
    manager = FlowManager(flow_timeout=1e9, on_flow_complete=completed.append)
    manager.add_packet(_packet(0.0, flags=SYN))
    manager.add_packet(_packet(0.05, flags=RST | ACK))
    assert len(completed) == 1
    assert completed[0].termination_reason == "rst"
    assert manager.active_flows == 0


def test_fin_in_both_directions_terminates_the_flow():
    completed = []
    manager = FlowManager(flow_timeout=1e9, on_flow_complete=completed.append)
    manager.add_packet(_packet(0.0, flags=SYN))
    manager.add_packet(_packet(0.1, flags=FIN | ACK))
    assert not completed
    manager.add_packet(
        _packet(0.2, src="10.0.0.2", dst="10.0.0.1", sp=80, dp=1000, flags=FIN | ACK)
    )
    assert len(completed) == 1
    assert completed[0].termination_reason == "fin"


def test_timeout_expires_a_flow():
    completed = []
    manager = FlowManager(flow_timeout=2.0, on_flow_complete=completed.append)
    manager.add_packet(_packet(0.0, flags=PSH | ACK))
    assert manager.expire(now=1.0) == []
    expired = manager.expire(now=5.0)
    assert len(expired) == 1
    assert expired[0].termination_reason == "timeout"
    assert completed == expired


def test_timeout_is_measured_from_flow_start_not_last_packet():
    """CICFlowMeter: currentTimestamp - flow.getFlowStartTime() > flowTimeOut."""
    completed = []
    manager = FlowManager(flow_timeout=10.0, on_flow_complete=completed.append)
    for ts in (0.0, 4.0, 8.0):
        manager.add_packet(_packet(ts, flags=PSH | ACK))
    # The flow was active 2.5 s ago, but it started 10.5 s ago.
    assert len(manager.expire(now=10.5)) == 1


def test_a_packet_past_the_timeout_starts_a_new_flow():
    completed = []
    manager = FlowManager(flow_timeout=10.0, on_flow_complete=completed.append)
    for ts in (0.0, 6.0, 10.5):
        manager.add_packet(_packet(ts, flags=PSH | ACK))
    assert len(completed) == 1
    assert completed[0].packet_count == 2
    assert flow_to_features(completed[0])["flow_duration"] == pytest.approx(6_000_000.0)
    assert manager.active_flows == 1


def test_single_packet_flows_are_dropped_on_timeout_and_flush_paths():
    completed = []
    manager = FlowManager(flow_timeout=1.0, on_flow_complete=completed.append,
                          min_packets_to_emit=2)
    manager.add_packet(_packet(0.0, sp=1, flags=SYN))
    manager.add_packet(_packet(0.0, sp=2, flags=SYN))
    assert manager.expire(now=5.0) == []
    manager.add_packet(_packet(6.0, sp=3, flags=SYN))
    assert manager.flush() == []
    assert completed == []
    assert manager.stats()["discarded_flows"] == 3


def test_single_packet_flow_closed_by_rst_is_kept():
    completed = []
    manager = FlowManager(flow_timeout=1e9, on_flow_complete=completed.append,
                          min_packets_to_emit=2)
    manager.add_packet(_packet(0.0, flags=RST))
    assert len(completed) == 1


def test_first_packet_counts_twice_in_flow_wide_length_stats():
    """Published rows: 2 forward packets of 6 bytes give mean 6 and average size 9."""
    manager = FlowManager(flow_timeout=1e9)
    manager.add_packet(_packet(0.0, payload=6, flags=PSH | ACK))
    manager.add_packet(_packet(0.1, payload=6, flags=PSH | ACK))
    features = flow_to_features(manager.flush()[0])
    assert features["packet_length_mean"] == pytest.approx(6.0)
    assert features["average_packet_size"] == pytest.approx(9.0)
    assert features["total_length_of_fwd_packets"] == 12
    assert features["fwd_packet_length_mean"] == pytest.approx(6.0)


def test_act_data_pkt_fwd_skips_the_first_packet():
    manager = FlowManager(flow_timeout=1e9)
    for ts in (0.0, 0.1, 0.2):
        manager.add_packet(_packet(ts, payload=50, flags=PSH | ACK))
    assert flow_to_features(manager.flush()[0])["act_data_pkt_fwd"] == 2


def test_fwd_psh_and_urg_only_look_at_the_first_packet():
    manager = FlowManager(flow_timeout=1e9)
    manager.add_packet(_packet(0.0, flags=ACK))
    for ts in (0.1, 0.2, 0.3):
        manager.add_packet(_packet(ts, flags=PSH | ACK | 0x20))
    features = flow_to_features(manager.flush()[0])
    assert features["fwd_psh_flags"] == 0
    assert features["fwd_urg_flags"] == 0
    assert features["psh_flag_count"] == 3

    manager = FlowManager(flow_timeout=1e9)
    manager.add_packet(_packet(0.0, flags=PSH | ACK))
    manager.add_packet(_packet(0.1, flags=PSH | ACK))
    assert flow_to_features(manager.flush()[0])["fwd_psh_flags"] == 1


def test_final_active_burst_is_not_recorded():
    """A flow with no gap above the threshold has zero Active time, as published."""
    manager = FlowManager(flow_timeout=1e9, activity_timeout=5.0)
    for ts in (0.0, 0.1, 0.5, 1.0):
        manager.add_packet(_packet(ts, flags=PSH | ACK))
    features = flow_to_features(manager.flush()[0])
    assert features["active_mean"] == 0.0
    assert features["idle_mean"] == 0.0


def test_max_active_flows_evicts_the_oldest():
    manager = FlowManager(flow_timeout=1e9, max_active_flows=3)
    for index in range(6):
        manager.add_packet(_packet(float(index), sp=1000 + index, flags=PSH | ACK))
    assert manager.active_flows <= 3
    assert manager.evicted_count >= 2


def test_udp_flows_are_supported():
    manager = FlowManager(flow_timeout=1e9)
    manager.add_packet(_packet(0.0, protocol=UDP, dp=53, window=-1))
    manager.add_packet(_packet(0.01, protocol=UDP, dp=53, window=-1))
    flow = manager.flush()[0]
    features = flow_to_features(flow)
    assert features["protocol"] == float(UDP)
    assert features["fwd_header_length"] == 16  # two 8-byte UDP headers


def test_subflow_counting_uses_the_one_second_gap():
    manager = FlowManager(flow_timeout=1e9, subflow_timeout=1.0)
    for ts in (0.0, 0.1, 0.2, 5.0, 5.1, 12.0):
        manager.add_packet(_packet(ts, flags=PSH | ACK, payload=100))
    flow = manager.flush()[0]
    assert flow.subflow_count == 3
    features = flow_to_features(flow)
    # The published Subflow columns equal the totals even across gaps above 1 s.
    assert features["subflow_fwd_packets"] == 6
    assert features["subflow_fwd_bytes"] == 600


def test_active_idle_uses_the_five_second_threshold():
    manager = FlowManager(flow_timeout=1e9, activity_timeout=5.0)
    for ts in (0.0, 0.5, 1.0, 20.0, 20.5):
        manager.add_packet(_packet(ts, flags=PSH | ACK))
    flow = manager.flush()[0]
    features = flow_to_features(flow)
    assert features["idle_max"] == pytest.approx(19_000_000.0, rel=1e-6)
    # Only the burst before the gap is recorded (0 to 1 s); the final one never is.
    assert features["active_max"] == pytest.approx(1_000_000.0, rel=1e-6)


def test_flag_counters():
    manager = FlowManager(flow_timeout=1e9)
    manager.add_packet(_packet(0.0, flags=SYN))
    manager.add_packet(_packet(0.1, flags=SYN | ACK))
    manager.add_packet(_packet(0.2, flags=PSH | ACK))
    flow = manager.flush()[0]
    features = flow_to_features(flow)
    assert features["syn_flag_count"] == 2
    assert features["ack_flag_count"] == 2
    assert features["psh_flag_count"] == 1


def test_manager_statistics():
    manager = FlowManager(flow_timeout=1e9)
    manager.add_packets(synthetic_packets(src_ip="1.1.1.1", dst_ip="2.2.2.2",
                                          src_port=1, dst_port=2, count=8))
    stats = manager.stats()
    assert stats["packets"] == 8
    assert stats["active_flows"] == 1
    manager.flush()
    assert manager.stats()["completed_flows"] == 1


def test_callback_errors_do_not_kill_the_manager():
    def explode(_flow):
        raise RuntimeError("boom")

    manager = FlowManager(flow_timeout=1e9, on_flow_complete=explode)
    manager.add_packet(_packet(0.0, flags=SYN))
    manager.add_packet(_packet(0.1, flags=RST))
    assert manager.active_flows == 0  # survived the raising callback
