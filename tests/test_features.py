"""The feature contract and the live flow-feature extractor."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features import contract
from src.network.feature_extractor import flow_to_features, synthetic_packets
from src.network.flow_manager import FlowManager, TCP
from src.utils.errors import FeatureContractError


def test_contract_has_no_duplicate_names():
    names = [spec.name for spec in contract.FEATURES]
    assert len(names) == len(set(names))


def test_live_and_excluded_sets_are_disjoint():
    assert set(contract.LIVE_FEATURES).isdisjoint(set(contract.EXCLUDED))


def test_cic_aliases_map_back_to_canonical_names():
    assert contract.canonical_name(" Flow Bytes/s") == "flow_bytes_s"
    assert contract.canonical_name("Init_Win_bytes_forward") == "init_win_bytes_forward"
    assert contract.canonical_name(" Total Length of Fwd Packets") == "total_length_of_fwd_packets"
    assert contract.canonical_name(" Fwd Header Length.1") == "fwd_header_length_1"


def test_leaky_and_degenerate_columns_are_excluded():
    for name in ("flow_id", "source_ip", "destination_ip", "source_port",
                 "timestamp", "cwe_flag_count", "fwd_header_length_1",
                 "fwd_avg_bytes_bulk", "bwd_avg_bulk_rate",
                 "bwd_psh_flags", "bwd_urg_flags"):
        assert name in contract.EXCLUDED, f"{name} should be excluded"
        assert contract.EXCLUDED[name], f"{name} needs an exclusion reason"


def test_unverified_columns_are_off_by_default():
    resolved = contract.resolve_features(contract.LIVE_FEATURES)
    for name in contract.UNVERIFIED_FEATURES:
        assert name not in resolved.all
        assert name in resolved.disabled
    enabled = contract.resolve_features(contract.LIVE_FEATURES, use_unverified_columns=True)
    assert set(contract.UNVERIFIED_FEATURES) <= set(enabled.numeric)


def test_every_column_has_a_derivability_classification():
    allowed = {contract.DERIVABLE, contract.APPROXIMATION, contract.NOT_DERIVABLE}
    for spec in contract.FEATURES:
        assert contract.derivability(spec.name) in allowed
    assert contract.derivability("flow_iat_mean") == contract.DERIVABLE
    assert contract.derivability("subflow_fwd_packets") == contract.APPROXIMATION
    assert contract.derivability("fwd_avg_bytes_bulk") == contract.NOT_DERIVABLE


def test_destination_port_is_off_by_default():
    resolved = contract.resolve_features(
        contract.LIVE_FEATURES, use_protocol=True, use_destination_port=False
    )
    assert "destination_port" not in resolved.all
    assert "destination_port" in resolved.disabled
    assert "protocol" in resolved.categorical


def test_missing_protocol_column_is_reported_not_fatal():
    columns = [c for c in contract.LIVE_FEATURES if c != "protocol"]
    resolved = contract.resolve_features(columns, use_protocol=True)
    assert "protocol" in resolved.missing
    assert "protocol" not in resolved.all
    assert len(resolved.numeric) > 50


def test_resolve_rejects_a_non_cicids_header():
    with pytest.raises(FeatureContractError):
        contract.resolve_features(["a", "b", "c"])


def test_align_frame_enforces_order_and_raises_on_missing():
    expected = ["flow_duration", "total_fwd_packets", "flow_packets_s"]
    frame = pd.DataFrame([{"flow_packets_s": 3.0, "total_fwd_packets": 2.0,
                           "flow_duration": 1.0, "extra": 9.0}])
    aligned = contract.align_frame(frame, expected)
    assert list(aligned.columns) == expected
    assert aligned.iloc[0]["flow_duration"] == 1.0
    with pytest.raises(FeatureContractError):
        contract.align_frame(frame.drop(columns=["flow_duration"]), expected)


def test_port_bucket():
    assert contract.port_bucket(22) == "well_known"
    assert contract.port_bucket(8080) == "registered"
    assert contract.port_bucket(54321) == "dynamic"
    assert contract.port_bucket(None) == "dynamic"


def test_label_for_handles_transformer_prefixes():
    assert contract.label_for("num__flow_packets_s") == "Flow Packets/s"
    assert contract.label_for("proto__protocol_6") == "Protocol = TCP"


def _one_flow(**kwargs):
    manager = FlowManager(flow_timeout=1e9)
    packets = synthetic_packets(
        src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=50000, dst_port=443, **kwargs
    )
    manager.add_packets(packets)
    flows = manager.flush()
    assert flows
    return flows[0]


def test_extractor_emits_every_contract_feature():
    features = flow_to_features(_one_flow(count=10, interval=0.05, payload=400))
    assert set(features) == set(contract.LIVE_FEATURES)


def test_flow_duration_is_microseconds():
    features = flow_to_features(_one_flow(count=5, interval=0.1, payload=100))
    # 4 gaps of 0.1 s = 0.4 s = 400000 microseconds
    assert features["flow_duration"] == pytest.approx(400_000.0, rel=1e-6)
    assert features["flow_iat_mean"] == pytest.approx(100_000.0, rel=1e-6)


def test_counts_and_bytes_match_the_packets():
    flow = _one_flow(count=10, interval=0.02, payload=250)
    features = flow_to_features(flow)
    assert features["total_fwd_packets"] + features["total_backward_packets"] == 10
    total_bytes = (
        features["total_length_of_fwd_packets"] + features["total_length_of_bwd_packets"]
    )
    assert total_bytes == 250 * 10
    assert features["packet_length_mean"] == pytest.approx(250.0)
    # The first packet is counted twice in the numerator, as in CICFlowMeter.
    assert features["average_packet_size"] == pytest.approx(275.0)


def test_rate_features_are_nan_for_zero_duration_flows():
    """CIC-IDS2017 stored Infinity here; we emit NaN so the imputer handles it."""
    features = flow_to_features(_one_flow(count=3, interval=0.0, payload=100))
    assert math.isnan(features["flow_bytes_s"])
    assert math.isnan(features["flow_packets_s"])
    assert features["fwd_packets_s"] == 0.0


def test_no_infinite_values_are_ever_emitted():
    for interval in (0.0, 1e-9, 0.5):
        features = flow_to_features(_one_flow(count=6, interval=interval, payload=64))
        for name, value in features.items():
            assert not math.isinf(value), f"{name} is infinite"


def test_down_up_ratio_uses_integer_division_like_cicflowmeter():
    flow = _one_flow(count=11, interval=0.01, payload=100)
    features = flow_to_features(flow)
    expected = int(features["total_backward_packets"]) // int(features["total_fwd_packets"])
    assert features["down_up_ratio"] == float(expected)


def test_init_window_sentinel_when_no_tcp_window_is_seen():
    from src.network.flow_manager import PacketInfo

    manager = FlowManager(flow_timeout=1e9)
    manager.add_packet(
        PacketInfo(timestamp=0.0, src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=1,
                   dst_port=2, protocol=17, payload_len=10, header_len=8)
    )
    features = flow_to_features(manager.flush()[0])
    assert features["init_win_bytes_forward"] == contract.UNKNOWN_WINDOW
    assert features["init_win_bytes_backward"] == contract.UNKNOWN_WINDOW


def test_records_to_frame_orders_columns():
    order = list(contract.LIVE_FEATURES)
    record = contract.empty_record()
    frame = contract.records_to_frame([record, record], order)
    assert list(frame.columns) == order
    assert len(frame) == 2
