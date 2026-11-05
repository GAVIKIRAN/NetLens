"""Behaviour-based detection: patterns that only appear across many connections.

The XGBoost model scores one flow at a time.  A port scan is not visible that way: each
probe looks like a short failed connection, and a probe to a firewalled port is a single
SYN, which never becomes a flow at all because flows with one packet are not emitted
(CIC-IDS2017 has no such rows for the model to learn from).

So this detector reads **packets**, before flow assembly, and keeps per-source sliding
windows of TCP connection attempts.  An attempt ends in one of three ways:

    open       the target answered SYN-ACK
    refused    the target answered RST
    unanswered no reply within ``unanswered_after_seconds`` (typically a firewall)

No finding rests on one indicator.  Contacting many ports is normal if they answer, so a
port scan also needs most attempts to fail.  All thresholds come from
``detection.behavior`` in ``config.yaml``.  Time comes from packet timestamps, so a saved
capture behaves the same as live traffic.

Costs are bounded: memory by ``max_sources``, ``max_attempts_per_source`` and
``max_pending``, CPU by evaluating each source at most once per second of traffic.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.network.flow_manager import ACK, RST, SYN, TCP, PacketInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUSPICIOUS = "SUSPICIOUS"
ATTACK = "ATTACK"

PORT_SCAN = "port_scan"
HOST_SWEEP = "host_sweep"
FAILED_CONNECTIONS = "failed_connections"
REPEATED_ATTEMPTS = "repeated_attempts"
CONNECTION_RATE = "connection_rate"
SYN_FLOOD = "syn_flood"
TRAFFIC_BURST = "traffic_burst"

KIND_LABELS = {
    PORT_SCAN: "Port scan",
    HOST_SWEEP: "Host sweep",
    FAILED_CONNECTIONS: "Repeated failed connections",
    REPEATED_ATTEMPTS: "Repeated connection attempts",
    CONNECTION_RATE: "High connection rate",
    SYN_FLOOD: "SYN flood pattern",
    TRAFFIC_BURST: "Abnormal traffic burst",
}

_OPEN, _REFUSED, _PENDING = "open", "refused", "pending"

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "window_seconds": 120.0,
    "rate_window_seconds": 10.0,
    "unanswered_after_seconds": 3.0,
    "finding_hold_seconds": 120.0,
    "alert_cooldown_seconds": 60.0,
    "max_sources": 5000,
    "max_attempts_per_source": 5000,
    "max_pending": 50000,
    "port_scan": {"min_unique_ports": 25, "attack_unique_ports": 100, "min_failure_ratio": 0.6},
    "host_sweep": {"min_unique_hosts": 20, "attack_unique_hosts": 100, "min_failure_ratio": 0.8},
    "failed_connections": {"min_failures": 20, "min_failure_ratio": 0.8},
    "repeated_attempts": {"min_attempts": 60},
    "connection_rate": {"min_attempts_per_second": 20.0},
    "syn_flood": {"min_syn_per_second": 100.0, "min_unanswered_ratio": 0.8},
    "traffic_burst": {"min_packets_per_second": 3000.0, "max_mean_payload_bytes": 100.0},
}


@dataclass
class Finding:
    """One behavioural detection, kept separate from any model prediction."""

    kind: str
    decision: str
    src_ip: str
    dst_ip: str = ""
    dst_port: int = 0
    reason: str = ""
    strength: float = 0.5
    indicators: dict[str, Any] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    repeats: int = 1
    last_emitted: float = 0.0

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def result_text(self) -> str:
        state = "detected" if self.decision == ATTACK else "suspected"
        return f"{self.label} {state}"

    def matches(self, src_ip: str, dst_ip: str, dst_port: int) -> bool:
        """Does this flow belong to the detected pattern?

        A shared source address is not enough, so a scanner's unrelated traffic is not
        relabelled.  Rate and burst findings describe a whole host and match no flow.
        """
        if self.kind == PORT_SCAN:
            return src_ip == self.src_ip and dst_ip == self.dst_ip
        if self.kind == HOST_SWEEP:
            return src_ip == self.src_ip and int(dst_port) == self.dst_port
        if self.kind in (FAILED_CONNECTIONS, REPEATED_ATTEMPTS):
            return (
                src_ip == self.src_ip and dst_ip == self.dst_ip and int(dst_port) == self.dst_port
            )
        if self.kind == SYN_FLOOD:
            return dst_ip == self.dst_ip and int(dst_port) == self.dst_port
        return False


@dataclass(slots=True)
class _Attempt:
    ts: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    outcome: str = _PENDING


class _SourceState:
    __slots__ = ("attempts", "buckets", "last_eval", "dirty")

    def __init__(self, max_attempts: int) -> None:
        self.attempts: deque[_Attempt] = deque(maxlen=max_attempts)
        self.buckets: deque[list[float]] = deque(maxlen=16)  # [second, packets, payload bytes]
        self.last_eval = float("-inf")
        self.dirty = False


class BehaviorDetector:
    """Sliding-window behaviour analysis over captured packets."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        on_finding: Callable[[Finding], None] | None = None,
    ) -> None:
        merged: dict[str, Any] = {}
        for key, default in DEFAULT_THRESHOLDS.items():
            value = (config or {}).get(key)
            if isinstance(default, dict):
                merged[key] = {**default, **(value or {})}
            else:
                merged[key] = default if value is None else value
        self.config = merged
        self.window = float(merged["window_seconds"])
        self.rate_window = float(merged["rate_window_seconds"])
        self.unanswered_after = float(merged["unanswered_after_seconds"])
        self.hold = float(merged["finding_hold_seconds"])
        self.cooldown = float(merged["alert_cooldown_seconds"])
        self.max_sources = int(merged["max_sources"])
        self.max_attempts = int(merged["max_attempts_per_source"])
        self.max_pending = int(merged["max_pending"])

        self.on_finding = on_finding
        self._lock = threading.RLock()
        self._sources: "OrderedDict[str, _SourceState]" = OrderedDict()
        self._pending: dict[tuple, _Attempt] = {}
        self._findings: dict[tuple, Finding] = {}
        self._clock = 0.0
        self.packets_seen = 0
        self.attempts_seen = 0
        self.sources_evicted = 0

    # -------------------------------------------------------------- ingest
    def observe(self, packet: PacketInfo) -> None:
        """Take one packet.  Findings are delivered through ``on_finding``."""
        emitted: list[Finding] = []
        with self._lock:
            now = float(packet.timestamp)
            self._clock = max(self._clock, now)
            self.packets_seen += 1
            state = self._source(packet.src_ip)
            self._count_packet(state, now, packet.payload_len)
            if packet.protocol == TCP:
                self._observe_tcp(packet, state, now)
            if now - state.last_eval >= 1.0:
                emitted = self._evaluate_source(packet.src_ip, state, self._clock)
        self._deliver(emitted)

    def _observe_tcp(self, packet: PacketInfo, state: _SourceState, now: float) -> None:
        flags = int(packet.flags or 0)
        if flags & SYN and not flags & ACK:
            key = (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port)
            known = self._pending.get(key)
            if known is not None and now - known.ts <= self.window:
                return  # a retransmission of the same attempt
            attempt = _Attempt(now, packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port)
            state.attempts.append(attempt)
            state.dirty = True
            self.attempts_seen += 1
            self._pending[key] = attempt
            if len(self._pending) > self.max_pending:
                self._pending.pop(next(iter(self._pending)))
        elif flags & (SYN | RST):
            # A reply travels from the target back to the attempt's source.
            key = (packet.dst_ip, packet.dst_port, packet.src_ip, packet.src_port)
            attempt = self._pending.get(key)
            if attempt is None:
                return
            attempt.outcome = _REFUSED if flags & RST else _OPEN
            del self._pending[key]
            source = self._sources.get(attempt.src_ip)
            if source is not None:
                source.dirty = True

    def tick(self, now: float | None = None) -> None:
        """Re-evaluate without new packets, so unanswered probes can age into failures."""
        emitted: list[Finding] = []
        with self._lock:
            if now is not None:
                self._clock = max(self._clock, float(now))
            clock = self._clock
            for src_ip, state in list(self._sources.items()):
                if state.dirty or (state.attempts and clock - state.last_eval >= 1.0):
                    emitted.extend(self._evaluate_source(src_ip, state, clock))
            self._prune_findings(clock)
            self._prune_pending(clock)
        self._deliver(emitted)

    def _source(self, src_ip: str) -> _SourceState:
        state = self._sources.get(src_ip)
        if state is None:
            state = _SourceState(self.max_attempts)
            self._sources[src_ip] = state
            if len(self._sources) > self.max_sources:
                self._sources.popitem(last=False)
                self.sources_evicted += 1
        else:
            self._sources.move_to_end(src_ip)
        return state

    @staticmethod
    def _count_packet(state: _SourceState, now: float, payload: int) -> None:
        second = float(int(now))
        if state.buckets and state.buckets[-1][0] == second:
            state.buckets[-1][1] += 1
            state.buckets[-1][2] += max(0, int(payload))
        else:
            state.buckets.append([second, 1.0, float(max(0, int(payload)))])

    # ---------------------------------------------------------- evaluation
    def _failed(self, attempt: _Attempt, now: float) -> bool:
        if attempt.outcome == _REFUSED:
            return True
        return attempt.outcome == _PENDING and now - attempt.ts >= self.unanswered_after

    def _resolved(self, attempt: _Attempt, now: float) -> bool:
        return attempt.outcome != _PENDING or now - attempt.ts >= self.unanswered_after

    def _evaluate_source(self, src_ip: str, state: _SourceState, now: float) -> list[Finding]:
        state.last_eval = now
        state.dirty = False
        cutoff = now - self.window
        while state.attempts and state.attempts[0].ts < cutoff:
            state.attempts.popleft()

        emitted: list[Finding] = []
        emitted += self._check_burst(src_ip, state, now)
        if not state.attempts:
            return emitted

        by_target: dict[str, list[_Attempt]] = {}
        by_port: dict[int, list[_Attempt]] = {}
        by_service: dict[tuple[str, int], list[_Attempt]] = {}
        for attempt in state.attempts:
            by_target.setdefault(attempt.dst_ip, []).append(attempt)
            by_port.setdefault(attempt.dst_port, []).append(attempt)
            by_service.setdefault((attempt.dst_ip, attempt.dst_port), []).append(attempt)

        for dst_ip, attempts in by_target.items():
            emitted += self._check_port_scan(src_ip, dst_ip, attempts, now)
        for dst_port, attempts in by_port.items():
            emitted += self._check_host_sweep(src_ip, dst_port, attempts, now)
        for (dst_ip, dst_port), attempts in by_service.items():
            emitted += self._check_service(src_ip, dst_ip, dst_port, attempts, now)
        emitted += self._check_rate(src_ip, state, now)
        return emitted

    def _ratio(self, attempts: list[_Attempt], now: float) -> tuple[int, int, float]:
        resolved = sum(1 for a in attempts if self._resolved(a, now))
        failed = sum(1 for a in attempts if self._failed(a, now))
        return resolved, failed, (failed / resolved if resolved else 0.0)

    def _check_port_scan(self, src_ip, dst_ip, attempts, now) -> list[Finding]:
        cfg = self.config[PORT_SCAN]
        ports = {a.dst_port for a in attempts}
        if len(ports) < int(cfg["min_unique_ports"]):
            return []
        resolved, failed, ratio = self._ratio(attempts, now)
        if not resolved or ratio < float(cfg["min_failure_ratio"]):
            return []
        attack_at = int(cfg["attack_unique_ports"])
        refused = sum(1 for a in attempts if a.outcome == _REFUSED)
        opened = sum(1 for a in attempts if a.outcome == _OPEN)
        decision = ATTACK if len(ports) >= attack_at else SUSPICIOUS
        lead = "Port scan detected" if decision == ATTACK else "Port scan suspected"
        reason = (
            f"{lead}: {len(ports)} unique destination ports on {dst_ip} contacted within "
            f"{self.window:.0f} s; {failed} of {resolved} finished attempts failed "
            f"({refused} refused, {failed - refused} unanswered, {opened} answered)."
        )
        return self._record(
            PORT_SCAN, decision, src_ip, dst_ip, 0, reason, len(ports) / attack_at,
            {"unique_ports": len(ports), "attempts": len(attempts), "failed": failed,
             "resolved": resolved, "failure_ratio": round(ratio, 3), "refused": refused,
             "answered": opened},
            now,
        )

    def _check_host_sweep(self, src_ip, dst_port, attempts, now) -> list[Finding]:
        cfg = self.config[HOST_SWEEP]
        hosts = {a.dst_ip for a in attempts}
        if len(hosts) < int(cfg["min_unique_hosts"]):
            return []
        resolved, failed, ratio = self._ratio(attempts, now)
        if not resolved or ratio < float(cfg["min_failure_ratio"]):
            return []
        attack_at = int(cfg["attack_unique_hosts"])
        decision = ATTACK if len(hosts) >= attack_at else SUSPICIOUS
        state = "detected" if decision == ATTACK else "suspected"
        reason = (
            f"Host sweep {state}: {len(hosts)} hosts contacted on port {dst_port} within "
            f"{self.window:.0f} s; {failed} of {resolved} finished attempts failed."
        )
        return self._record(
            HOST_SWEEP, decision, src_ip, "", dst_port, reason, len(hosts) / attack_at,
            {"unique_hosts": len(hosts), "failed": failed, "resolved": resolved,
             "failure_ratio": round(ratio, 3)},
            now,
        )

    def _check_service(self, src_ip, dst_ip, dst_port, attempts, now) -> list[Finding]:
        emitted: list[Finding] = []
        cfg = self.config[FAILED_CONNECTIONS]
        resolved, failed, ratio = self._ratio(attempts, now)
        minimum = int(cfg["min_failures"])
        if failed >= minimum and ratio >= float(cfg["min_failure_ratio"]):
            reason = (
                f"Repeated failed connections: {failed} of {resolved} finished attempts to "
                f"{dst_ip}:{dst_port} failed within {self.window:.0f} s."
            )
            emitted += self._record(
                FAILED_CONNECTIONS, SUSPICIOUS, src_ip, dst_ip, dst_port, reason,
                failed / (2 * minimum),
                {"failed": failed, "resolved": resolved, "failure_ratio": round(ratio, 3)},
                now,
            )
        minimum = int(self.config[REPEATED_ATTEMPTS]["min_attempts"])
        if len(attempts) >= minimum:
            reason = (
                f"Repeated connection attempts: {len(attempts)} new connections to "
                f"{dst_ip}:{dst_port} within {self.window:.0f} s."
            )
            emitted += self._record(
                REPEATED_ATTEMPTS, SUSPICIOUS, src_ip, dst_ip, dst_port, reason,
                len(attempts) / (2 * minimum), {"attempts": len(attempts)}, now,
            )
        # A flood is judged on attempts old enough to have had time to be answered.
        cfg = self.config[SYN_FLOOD]
        upper = now - self.unanswered_after
        lower = upper - self.rate_window
        aged = [a for a in attempts if lower <= a.ts <= upper]
        rate = len(aged) / self.rate_window
        minimum_rate = float(cfg["min_syn_per_second"])
        if rate >= minimum_rate:
            unanswered = sum(1 for a in aged if a.outcome == _PENDING)
            share = unanswered / len(aged)
            if share >= float(cfg["min_unanswered_ratio"]):
                reason = (
                    f"SYN flood pattern: {rate:.0f} new connections per second to "
                    f"{dst_ip}:{dst_port}, with {share:.0%} never answered."
                )
                emitted += self._record(
                    SYN_FLOOD, ATTACK, src_ip, dst_ip, dst_port, reason,
                    rate / (2 * minimum_rate),
                    {"syn_per_second": round(rate, 1), "unanswered_ratio": round(share, 3)},
                    now,
                )
        return emitted

    def _check_rate(self, src_ip, state, now) -> list[Finding]:
        minimum = float(self.config[CONNECTION_RATE]["min_attempts_per_second"])
        recent = 0
        for attempt in reversed(state.attempts):
            if attempt.ts < now - self.rate_window:
                break
            recent += 1
        rate = recent / self.rate_window
        if rate < minimum:
            return []
        reason = (
            f"High connection rate: {rate:.0f} new connections per second from {src_ip} "
            f"over the last {self.rate_window:.0f} s."
        )
        return self._record(
            CONNECTION_RATE, SUSPICIOUS, src_ip, "", 0, reason, rate / (2 * minimum),
            {"attempts_per_second": round(rate, 1)}, now,
        )

    def _check_burst(self, src_ip, state, now) -> list[Finding]:
        """Many tiny packets.  A large download also has a high rate, but big packets."""
        cfg = self.config[TRAFFIC_BURST]
        seconds = [b for b in state.buckets if now - b[0] <= 5.0]
        if len(seconds) < 3:
            return []
        packets = sum(b[1] for b in seconds)
        rate = packets / len(seconds)
        mean_payload = sum(b[2] for b in seconds) / packets
        minimum = float(cfg["min_packets_per_second"])
        if rate < minimum or mean_payload > float(cfg["max_mean_payload_bytes"]):
            return []
        reason = (
            f"Abnormal traffic burst: {rate:.0f} packets per second from {src_ip} with a mean "
            f"payload of {mean_payload:.0f} bytes."
        )
        return self._record(
            TRAFFIC_BURST, SUSPICIOUS, src_ip, "", 0, reason, rate / (2 * minimum),
            {"packets_per_second": round(rate, 1), "mean_payload": round(mean_payload, 1)}, now,
        )

    # ------------------------------------------------------------ findings
    def _record(
        self, kind, decision, src_ip, dst_ip, dst_port, reason, strength, indicators, now
    ) -> list[Finding]:
        key = (kind, src_ip, dst_ip, int(dst_port))
        strength = min(1.0, max(0.0, float(strength)))
        finding = self._findings.get(key)
        if finding is None:
            finding = Finding(
                kind, decision, src_ip, dst_ip, int(dst_port), reason, strength,
                dict(indicators), first_seen=now, last_seen=now, last_emitted=now,
            )
            self._findings[key] = finding
            return [finding]

        upgraded = decision == ATTACK and finding.decision != ATTACK
        finding.decision = ATTACK if upgraded else finding.decision
        finding.reason = reason
        finding.strength = max(finding.strength, strength)
        finding.indicators = dict(indicators)
        finding.last_seen = now
        if upgraded or now - finding.last_emitted >= self.cooldown:
            finding.last_emitted = now
            finding.repeats += 1
            return [finding]
        return []

    def _deliver(self, findings: list[Finding]) -> None:
        if not findings or self.on_finding is None:
            return
        for finding in findings:
            try:
                self.on_finding(finding)
            except Exception:  # a consumer must never stop the capture thread
                logger.exception("on_finding callback raised")

    def _prune_findings(self, now: float) -> None:
        for key in [k for k, f in self._findings.items() if now - f.last_seen > self.hold]:
            del self._findings[key]

    def _prune_pending(self, now: float) -> None:
        stale = [k for k, a in self._pending.items() if now - a.ts > self.window]
        for key in stale:
            del self._pending[key]

    # ------------------------------------------------------------- queries
    def findings_for(
        self, src_ip: str, dst_ip: str, dst_port: int, at: float | None = None
    ) -> list[Finding]:
        """Active findings that this flow is part of."""
        with self._lock:
            when = self._clock if at is None else float(at)
            return [
                f for f in self._findings.values()
                if when - f.last_seen <= self.hold
                and f.first_seen - self.hold <= when
                and f.matches(src_ip, dst_ip, int(dst_port or 0))
            ]

    def active_findings(self) -> list[Finding]:
        with self._lock:
            return list(self._findings.values())

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "packets": self.packets_seen,
                "connection_attempts": self.attempts_seen,
                "tracked_sources": len(self._sources),
                "pending_attempts": len(self._pending),
                "active_findings": len(self._findings),
                "sources_evicted": self.sources_evicted,
            }


__all__ = [
    "ATTACK",
    "SUSPICIOUS",
    "BehaviorDetector",
    "Finding",
    "KIND_LABELS",
    "PORT_SCAN",
    "HOST_SWEEP",
    "FAILED_CONNECTIONS",
    "REPEATED_ATTEMPTS",
    "CONNECTION_RATE",
    "SYN_FLOOD",
    "TRAFFIC_BURST",
]
