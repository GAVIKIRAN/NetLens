"""Packet parsing and flow -> feature-vector conversion (Phases 3 & 6).

Two responsibilities:

``packet_to_info``
    Turns a Scapy packet into the protocol-independent
    :class:`~src.network.flow_manager.PacketInfo` the flow manager consumes.
    IPv4, IPv6, TCP, UDP and ICMP are handled; anything else is ignored.

``flow_to_features``
    Turns a completed :class:`~src.network.flow_manager.Flow` into a dictionary
    whose keys are **exactly** the canonical names in
    :mod:`src.features.contract`.  Units follow CIC-IDS2017: durations and
    inter-arrival times are in microseconds, rates are per second, and lengths
    are transport-payload bytes.

One deliberate detail: when a flow lasts zero microseconds (a single packet, or
several packets with identical timestamps) CICFlowMeter emitted ``Infinity``
for ``Flow Bytes/s`` and ``Flow Packets/s``.  Those became NaN when this
project cleaned the CSVs, and were then filled by the pipeline's median
imputer.  The live extractor therefore emits ``NaN`` for exactly those two
features in exactly that situation, so a zero-duration flow is treated the same
way at inference time as it was at training time.  ``Fwd/Bwd Packets/s`` are
emitted as ``0.0`` instead, which is what CICFlowMeter does for them.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.features.contract import LIVE_FEATURES, UNKNOWN_WINDOW
from src.network.flow_manager import ICMP, TCP, UDP, Flow, PacketInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MISSING_RATE = float("nan")


# ===========================================================================
#  Packet -> PacketInfo
# ===========================================================================
def packet_to_info(packet: Any, timestamp: float | None = None) -> PacketInfo | None:
    """Convert a Scapy packet into a :class:`PacketInfo`, or ``None`` to skip."""
    try:
        from scapy.layers.inet import ICMP as ScapyICMP, IP, TCP as ScapyTCP, UDP as ScapyUDP
        from scapy.layers.inet6 import IPv6
        from scapy.packet import Padding
    except Exception:  # pragma: no cover - scapy missing
        return None

    if packet is None:
        return None

    when = float(timestamp if timestamp is not None else getattr(packet, "time", 0.0) or 0.0)

    if packet.haslayer(IP):
        network = packet[IP]
        src_ip, dst_ip = str(network.src), str(network.dst)
        protocol = int(network.proto)
    elif packet.haslayer(IPv6):
        network = packet[IPv6]
        src_ip, dst_ip = str(network.src), str(network.dst)
        protocol = int(network.nh)
    else:
        return None

    frame_len = int(len(packet))
    # Short Ethernet frames are padded to 60 bytes.  Scapy keeps that padding in the
    # layer chain, but it is not payload: CICFlowMeter counts the IP-declared length.
    padding = packet.getlayer(Padding)
    padding_len = len(bytes(padding)) if padding is not None else 0
    src_port = dst_port = 0
    header_len = 0
    flags = 0
    window = UNKNOWN_WINDOW
    payload_len = 0

    if packet.haslayer(ScapyTCP):
        transport = packet[ScapyTCP]
        src_port = int(transport.sport)
        dst_port = int(transport.dport)
        header_len = int(transport.dataofs or 5) * 4
        try:
            flags = int(transport.flags)
        except (TypeError, ValueError):  # scapy FlagValue
            flags = int(getattr(transport.flags, "value", 0))
        window = int(transport.window or 0)
        payload_len = len(bytes(transport.payload)) - padding_len
        protocol = TCP
    elif packet.haslayer(ScapyUDP):
        transport = packet[ScapyUDP]
        src_port = int(transport.sport)
        dst_port = int(transport.dport)
        header_len = 8
        payload_len = len(bytes(transport.payload)) - padding_len
        protocol = UDP
    elif packet.haslayer(ScapyICMP):
        transport = packet[ScapyICMP]
        header_len = 8
        payload_len = len(bytes(transport.payload)) - padding_len
        protocol = ICMP
    else:
        payload_len = len(bytes(network.payload)) - padding_len

    return PacketInfo(
        timestamp=when,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        payload_len=max(0, int(payload_len)),
        frame_len=frame_len,
        header_len=int(header_len),
        flags=int(flags),
        window=int(window),
    )


# ===========================================================================
#  Flow -> feature dictionary
# ===========================================================================
def flow_to_features(flow: Flow) -> dict[str, float]:
    """Convert a completed flow into a contract-compliant feature dictionary."""
    duration_us = float(flow.duration_us)
    duration_s = float(flow.duration_seconds)

    fwd, bwd = flow.fwd, flow.bwd
    total_packets = int(fwd.packets + bwd.packets)
    total_bytes = float(fwd.bytes_total + bwd.bytes_total)

    if duration_s > 0:
        flow_bytes_s = total_bytes / duration_s
        flow_packets_s = total_packets / duration_s
        fwd_packets_s = fwd.packets / duration_s
        bwd_packets_s = bwd.packets / duration_s
    else:
        # See the module docstring: this mirrors how CIC-IDS2017 encoded
        # zero-duration flows, so training and inference agree.
        flow_bytes_s = _MISSING_RATE
        flow_packets_s = _MISSING_RATE
        fwd_packets_s = 0.0
        bwd_packets_s = 0.0

    # Integer division, as in CICFlowMeter's getDownUpRatio.
    down_up_ratio = float(bwd.packets // fwd.packets) if fwd.packets > 0 else 0.0

    features: dict[str, float] = {
        # ------------------------------------------------------- metadata
        "protocol": float(flow.protocol),
        "destination_port": float(flow.dst_port),

        # --------------------------------------------------------- volume
        "flow_duration": duration_us,
        "total_fwd_packets": float(fwd.packets),
        "total_backward_packets": float(bwd.packets),
        "total_length_of_fwd_packets": float(fwd.bytes_total),
        "total_length_of_bwd_packets": float(bwd.bytes_total),

        # ---------------------------------------------------- packet size
        "fwd_packet_length_max": float(fwd.lengths.maximum if fwd.packets else 0.0),
        "fwd_packet_length_min": float(fwd.lengths.minimum if fwd.packets else 0.0),
        "fwd_packet_length_mean": float(fwd.lengths.mean),
        "fwd_packet_length_std": float(fwd.lengths.std),
        "bwd_packet_length_max": float(bwd.lengths.maximum if bwd.packets else 0.0),
        "bwd_packet_length_min": float(bwd.lengths.minimum if bwd.packets else 0.0),
        "bwd_packet_length_mean": float(bwd.lengths.mean),
        "bwd_packet_length_std": float(bwd.lengths.std),

        # ----------------------------------------------------------- rate
        "flow_bytes_s": float(flow_bytes_s),
        "flow_packets_s": float(flow_packets_s),
        "fwd_packets_s": float(fwd_packets_s),
        "bwd_packets_s": float(bwd_packets_s),

        # ------------------------------------------------------------ IAT
        "flow_iat_mean": float(flow.flow_iat.mean),
        "flow_iat_std": float(flow.flow_iat.std),
        "flow_iat_max": float(flow.flow_iat.maximum if flow.flow_iat.n else 0.0),
        "flow_iat_min": float(flow.flow_iat.minimum if flow.flow_iat.n else 0.0),
        "fwd_iat_total": float(fwd.iat.total),
        "fwd_iat_mean": float(fwd.iat.mean),
        "fwd_iat_std": float(fwd.iat.std),
        "fwd_iat_max": float(fwd.iat.maximum if fwd.iat.n else 0.0),
        "fwd_iat_min": float(fwd.iat.minimum if fwd.iat.n else 0.0),
        "bwd_iat_total": float(bwd.iat.total),
        "bwd_iat_mean": float(bwd.iat.mean),
        "bwd_iat_std": float(bwd.iat.std),
        "bwd_iat_max": float(bwd.iat.maximum if bwd.iat.n else 0.0),
        "bwd_iat_min": float(bwd.iat.minimum if bwd.iat.n else 0.0),

        # ------------------------------------------------------ TCP flags
        "fwd_psh_flags": float(fwd.psh_flags),
        "fwd_urg_flags": float(fwd.urg_flags),
        "fin_flag_count": float(flow.flag_counts["fin"]),
        "syn_flag_count": float(flow.flag_counts["syn"]),
        "rst_flag_count": float(flow.flag_counts["rst"]),
        "psh_flag_count": float(flow.flag_counts["psh"]),
        "ack_flag_count": float(flow.flag_counts["ack"]),
        "urg_flag_count": float(flow.flag_counts["urg"]),
        "ece_flag_count": float(flow.flag_counts["ece"]),

        # ------------------------------------------------- header / window
        "fwd_header_length": float(fwd.header_bytes),
        "bwd_header_length": float(bwd.header_bytes),
        "init_win_bytes_forward": float(
            fwd.init_window if fwd.init_window is not None else UNKNOWN_WINDOW
        ),
        "init_win_bytes_backward": float(
            bwd.init_window if bwd.init_window is not None else UNKNOWN_WINDOW
        ),
        "act_data_pkt_fwd": float(fwd.data_packets),
        "min_seg_size_forward": float(fwd.min_header if fwd.packets else 0.0),

        # ------------------------------------------------ aggregate sizes
        "min_packet_length": float(flow.lengths.minimum if flow.lengths.n else 0.0),
        "max_packet_length": float(flow.lengths.maximum if flow.lengths.n else 0.0),
        "packet_length_mean": float(flow.lengths.mean),
        "packet_length_std": float(flow.lengths.std),
        "packet_length_variance": float(flow.lengths.variance),
        "down_up_ratio": down_up_ratio,
        # getAvgPacketSize divides flowLengthStats.getSum(), which holds the first
        # packet twice, by the real packet count, so this is not packet_length_mean.
        "average_packet_size": float(flow.lengths.total / total_packets) if total_packets else 0.0,
        "avg_fwd_segment_size": float(fwd.lengths.mean),
        "avg_bwd_segment_size": float(bwd.lengths.mean),

        # -------------------------------------------------------- subflow
        # The published columns equal the totals, even for flows with gaps above
        # 1 s, so dividing by flow.subflow_count here would be out of distribution.
        "subflow_fwd_packets": float(fwd.packets),
        "subflow_fwd_bytes": float(fwd.bytes_total),
        "subflow_bwd_packets": float(bwd.packets),
        "subflow_bwd_bytes": float(bwd.bytes_total),

        # -------------------------------------------------- active / idle
        "active_mean": float(flow.active.mean),
        "active_std": float(flow.active.std),
        "active_max": float(flow.active.maximum if flow.active.n else 0.0),
        "active_min": float(flow.active.minimum if flow.active.n else 0.0),
        "idle_mean": float(flow.idle.mean),
        "idle_std": float(flow.idle.std),
        "idle_max": float(flow.idle.maximum if flow.idle.n else 0.0),
        "idle_min": float(flow.idle.minimum if flow.idle.n else 0.0),
    }

    # Guard against any accidental inf leaking through; NaN is intentional and
    # is handled by the pipeline's imputer, inf is not.
    for key, value in features.items():
        if isinstance(value, float) and math.isinf(value):
            features[key] = _MISSING_RATE

    missing = [name for name in LIVE_FEATURES if name not in features]
    if missing:  # pragma: no cover - guarded by tests
        raise KeyError(f"flow_to_features is missing contract feature(s): {missing}")
    return features


def features_to_array(features: Mapping[str, float], order: list[str]) -> np.ndarray:
    """Order a feature dictionary into a 1-D array (used by tests/benchmarks).

    A missing feature is an error, never a silent zero.
    """
    missing = [name for name in order if name not in features]
    if missing:
        raise KeyError(f"Missing feature(s): {missing}")
    return np.array([float(features[name]) for name in order], dtype=np.float64)


def synthetic_packets(
    *,
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: int = TCP,
    count: int = 10,
    start_time: float = 0.0,
    interval: float = 0.01,
    payload: int = 500,
    flags: int = 0x18,
) -> list[PacketInfo]:
    """Build a deterministic packet sequence.

    Used by the unit tests and by the safe demo tooling so that flow assembly
    can be exercised **without capturing or generating any real traffic**.
    """
    packets: list[PacketInfo] = []
    for index in range(int(count)):
        forward = index % 2 == 0
        packets.append(
            PacketInfo(
                timestamp=start_time + index * interval,
                src_ip=src_ip if forward else dst_ip,
                dst_ip=dst_ip if forward else src_ip,
                src_port=src_port if forward else dst_port,
                dst_port=dst_port if forward else src_port,
                protocol=protocol,
                payload_len=payload,
                frame_len=payload + 54,
                header_len=20 if protocol == TCP else 8,
                flags=flags if protocol == TCP else 0,
                window=8192 if protocol == TCP else -1,
            )
        )
    return packets


__all__ = [
    "packet_to_info",
    "flow_to_features",
    "features_to_array",
    "synthetic_packets",
]
