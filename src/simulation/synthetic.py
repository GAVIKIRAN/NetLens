"""Safe synthetic traffic generation for DEMO mode (Phase 12).

**Nothing in this module touches the network.**  It builds in-memory
:class:`~src.network.flow_manager.PacketInfo` sequences and feeds them through
the *same* flow meter the live capture path uses, so the resulting records are
internally consistent by construction (rates really are bytes over duration,
subflow counts really come from the 1 s gap rule, and so on).

This is what makes the project demonstrable without attacking anything: the
"attack" flows are numbers in a Python list, not packets on a wire.

The behavioural profiles below are *shapes* - a port scan is a handful of
one-way SYNs, a DDoS is a flood of tiny packets at a very high rate - chosen so
that the demo exercises the whole pipeline.  They are labelled ``synthetic``
in the generated file, and the dashboard always shows the model's prediction
next to the synthetic ground-truth label rather than assuming they agree.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.network.feature_extractor import flow_to_features
from src.network.flow_manager import ACK, FIN, FlowManager, PSH, PacketInfo, RST, SYN, TCP, UDP
from src.utils.logger import get_logger

logger = get_logger(__name__)

INTERNAL_PREFIX = "192.168.10."
EXTERNAL_HOSTS = ("172.16.0.1", "203.0.113.7", "198.51.100.42", "203.0.113.99")


@dataclass
class FlowProfile:
    """Behavioural template for one traffic class."""

    label: str
    protocol: int = TCP
    packets: tuple[int, int] = (10, 60)
    payload: tuple[int, int] = (80, 1400)
    interval: tuple[float, float] = (0.005, 0.4)
    backward_ratio: float = 0.8
    backward_payload_scale: float = 1.0
    dst_ports: Sequence[int] = (80, 443)
    flags: int = PSH | ACK
    first_flags: int = SYN
    close_flags: int = FIN | ACK
    reset_probability: float = 0.0
    idle_gap: tuple[float, float] | None = None
    description: str = ""


#: Behavioural profiles for the demo generator.
PROFILES: dict[str, FlowProfile] = {
    "BENIGN": FlowProfile(
        label="BENIGN",
        packets=(12, 90),
        payload=(60, 1400),
        interval=(0.01, 0.6),
        backward_ratio=0.9,
        dst_ports=(80, 443, 53, 22, 3389, 8080),
        description="Ordinary bidirectional client/server conversation.",
    ),
    "DDoS": FlowProfile(
        label="DDoS",
        packets=(200, 900),
        payload=(0, 64),
        interval=(0.00005, 0.0008),
        backward_ratio=0.02,
        dst_ports=(80, 443),
        flags=ACK,
        description="Very high packet rate, tiny payloads, almost no replies.",
    ),
    "DoS Hulk": FlowProfile(
        label="DoS Hulk",
        packets=(120, 400),
        payload=(200, 900),
        interval=(0.0005, 0.006),
        backward_ratio=0.25,
        dst_ports=(80,),
        flags=PSH | ACK,
        description="Sustained HTTP request flood from a single source.",
    ),
    "DoS slowloris": FlowProfile(
        label="DoS slowloris",
        packets=(20, 70),
        payload=(4, 40),
        interval=(1.5, 6.0),
        backward_ratio=0.1,
        dst_ports=(80,),
        flags=PSH | ACK,
        idle_gap=(4.0, 9.0),
        description="Long-lived connection trickling tiny partial requests.",
    ),
    "PortScan": FlowProfile(
        label="PortScan",
        packets=(1, 3),
        payload=(0, 0),
        interval=(0.0002, 0.004),
        backward_ratio=0.35,
        dst_ports=tuple(range(20, 1024, 7)),
        flags=SYN,
        close_flags=RST | ACK,
        reset_probability=0.7,
        description="Single SYN per port, frequently answered with RST.",
    ),
    "SSH-Patator": FlowProfile(
        label="SSH-Patator",
        packets=(20, 60),
        payload=(20, 120),
        interval=(0.02, 0.25),
        backward_ratio=1.0,
        dst_ports=(22,),
        flags=PSH | ACK,
        description="Repeated small credential-exchange bursts.",
    ),
    "FTP-Patator": FlowProfile(
        label="FTP-Patator",
        packets=(16, 50),
        payload=(10, 90),
        interval=(0.03, 0.3),
        backward_ratio=1.0,
        dst_ports=(21,),
        flags=PSH | ACK,
        description="Repeated small credential-exchange bursts over FTP.",
    ),
    "Bot": FlowProfile(
        label="Bot",
        packets=(6, 20),
        payload=(40, 260),
        interval=(2.0, 12.0),
        backward_ratio=0.8,
        dst_ports=(8080, 443, 6667),
        flags=PSH | ACK,
        idle_gap=(6.0, 20.0),
        description="Low-and-slow periodic beaconing to a controller.",
    ),
    "Web Attack - Brute Force": FlowProfile(
        label="Web Attack - Brute Force",
        packets=(25, 80),
        payload=(180, 700),
        interval=(0.05, 0.4),
        backward_ratio=0.9,
        dst_ports=(80, 8080),
        flags=PSH | ACK,
        description="Repeated form POSTs against a web login endpoint.",
    ),
}


@dataclass
class SyntheticFlow:
    """One generated flow: its identity, its features and its true label."""

    identity: dict[str, Any]
    features: dict[str, float]
    label: str
    profile: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class SyntheticFlowGenerator:
    """Builds coherent flow records by simulating packets, never sending any."""

    def __init__(
        self,
        seed: int = 42,
        profiles: dict[str, FlowProfile] | None = None,
        jitter: float = 0.0,
    ) -> None:
        """``jitter`` perturbs each profile's ranges per flow (0.0 = exact
        profiles, 0.4 = +/-40 %).  A non-zero jitter makes the classes overlap,
        which is what a realistic dataset looks like."""
        self.random = random.Random(seed)
        self.profiles = profiles or PROFILES
        self.jitter = max(0.0, float(jitter))

    def _jittered(self, profile: FlowProfile) -> FlowProfile:
        """Return a copy of ``profile`` with randomly widened ranges."""
        if self.jitter <= 0.0:
            return profile
        rnd = self.random

        def scale(bounds: tuple[float, float], integer: bool = False):
            factor = rnd.uniform(1.0 - self.jitter, 1.0 + self.jitter)
            low, high = bounds[0] * factor, bounds[1] * factor
            if integer:
                low, high = max(1, int(round(low))), max(1, int(round(high)))
                return (min(low, high), max(low, high))
            return (min(low, high), max(low, high))

        return FlowProfile(
            label=profile.label,
            protocol=profile.protocol,
            packets=scale(profile.packets, integer=True),
            payload=scale(profile.payload, integer=True) if profile.payload[1] > 0 else profile.payload,
            interval=scale(profile.interval),
            backward_ratio=max(
                0.0,
                profile.backward_ratio * rnd.uniform(1.0 - self.jitter, 1.0 + self.jitter),
            ),
            backward_payload_scale=profile.backward_payload_scale,
            dst_ports=profile.dst_ports,
            flags=profile.flags,
            first_flags=profile.first_flags,
            close_flags=profile.close_flags,
            reset_probability=profile.reset_probability,
            idle_gap=profile.idle_gap,
            description=profile.description,
        )

    # ------------------------------------------------------------ packets
    def _packets(self, profile: FlowProfile, start_time: float) -> list[PacketInfo]:
        rnd = self.random
        src_ip = f"{INTERNAL_PREFIX}{rnd.randint(2, 250)}"
        dst_ip = rnd.choice(EXTERNAL_HOSTS)
        if profile.label in ("DDoS", "DoS Hulk", "PortScan", "DoS slowloris"):
            src_ip, dst_ip = dst_ip, f"{INTERNAL_PREFIX}{rnd.randint(2, 60)}"
        src_port = rnd.randint(1024, 65535)
        dst_port = int(rnd.choice(list(profile.dst_ports)))

        total = rnd.randint(*profile.packets)
        packets: list[PacketInfo] = []
        timestamp = start_time
        header = 20 if profile.protocol == TCP else 8
        window = rnd.choice([8192, 29200, 64240, 65535])

        for index in range(total):
            forward = index == 0 or rnd.random() > profile.backward_ratio / (
                1.0 + profile.backward_ratio
            )
            if index == 0:
                flags = profile.first_flags
                payload = 0
            elif index == total - 1 and profile.protocol == TCP:
                flags = (
                    profile.close_flags
                    if rnd.random() > profile.reset_probability
                    else RST | ACK
                )
                payload = 0
            else:
                flags = profile.flags
                payload = rnd.randint(*profile.payload)
                if not forward:
                    payload = int(payload * profile.backward_payload_scale)

            packets.append(
                PacketInfo(
                    timestamp=timestamp,
                    src_ip=src_ip if forward else dst_ip,
                    dst_ip=dst_ip if forward else src_ip,
                    src_port=src_port if forward else dst_port,
                    dst_port=dst_port if forward else src_port,
                    protocol=profile.protocol,
                    payload_len=max(0, payload),
                    frame_len=max(0, payload) + header + 34,
                    header_len=header,
                    flags=flags if profile.protocol == TCP else 0,
                    window=window if profile.protocol == TCP else -1,
                )
            )
            gap = rnd.uniform(*profile.interval)
            if profile.idle_gap and rnd.random() < 0.25:
                gap = rnd.uniform(*profile.idle_gap)
            timestamp += gap
        return packets

    # -------------------------------------------------------------- flows
    def generate_flow(self, label: str, start_time: float = 0.0) -> SyntheticFlow:
        """Generate one labelled flow record."""
        profile = self._jittered(self.profiles.get(label, self.profiles["BENIGN"]))
        packets = self._packets(profile, start_time)
        # The real 120 s rule applies, so simulated durations match live ones.
        manager = FlowManager(flow_timeout=120.0, activity_timeout=5.0, subflow_timeout=1.0)
        # A flow can close mid-stream (RST/FIN), so collect both the flows the
        # manager emitted while ingesting and whatever is still open at the end.
        flows = list(manager.add_packets(packets))
        flows.extend(manager.flush())
        if not flows:  # pragma: no cover - defensive
            raise RuntimeError(f"Synthetic generation produced no flow for '{label}'.")
        flow = max(flows, key=lambda f: f.packet_count)
        return SyntheticFlow(
            identity=flow.identity(),
            features=flow_to_features(flow),
            label=profile.label,
            profile=profile.description,
        )

    def generate(
        self,
        count: int,
        *,
        attack_ratio: float = 0.3,
        labels: Sequence[str] | None = None,
        start_time: float = 0.0,
    ) -> list[SyntheticFlow]:
        """Generate ``count`` flows with the requested benign/attack mix."""
        attack_labels = [
            name for name in (labels or self.profiles.keys()) if name != "BENIGN"
        ]
        if not attack_labels:
            attack_labels = ["DDoS"]
        flows: list[SyntheticFlow] = []
        timestamp = float(start_time)
        for _ in range(int(count)):
            if self.random.random() < float(attack_ratio):
                label = self.random.choice(attack_labels)
            else:
                label = "BENIGN"
            flow = self.generate_flow(label, start_time=timestamp)
            flows.append(flow)
            timestamp += self.random.uniform(0.05, 1.5)
        logger.info("Generated %d synthetic flow record(s)", len(flows))
        return flows


def synthetic_to_rows(flows: Iterable[SyntheticFlow]) -> list[dict[str, Any]]:
    """Flatten synthetic flows into CSV-ready rows."""
    rows: list[dict[str, Any]] = []
    for flow in flows:
        row: dict[str, Any] = dict(flow.features)
        row["label"] = flow.label
        row["src_ip"] = flow.identity.get("src_ip", "")
        row["dst_ip"] = flow.identity.get("dst_ip", "")
        row["src_port"] = flow.identity.get("src_port", 0)
        row["dst_port"] = flow.identity.get("dst_port", 0)
        row["protocol_name"] = flow.identity.get("protocol_name", "")
        row["packets"] = flow.identity.get("packets", 0)
        row["bytes"] = flow.identity.get("bytes", 0)
        row["duration_seconds"] = flow.identity.get("duration_seconds", 0.0)
        row["source"] = "synthetic"
        rows.append(row)
    return rows


__all__ = [
    "FlowProfile",
    "PROFILES",
    "SyntheticFlow",
    "SyntheticFlowGenerator",
    "synthetic_to_rows",
    "INTERNAL_PREFIX",
    "EXTERNAL_HOSTS",
]
