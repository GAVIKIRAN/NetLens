"""Packet-level traffic scenarios for testing detection.  Nothing is transmitted.

Each scenario returns :class:`~src.network.flow_manager.PacketInfo` objects in memory.
The list includes deliberately hard cases: normal traffic that resembles an attack, and
scans built to fall below the thresholds, because a test set of easy cases proves little.

The values approximate traffic on a home network.  They are a check of the logic, not a
measurement of real-world accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.network.flow_manager import ACK, FIN, PSH, RST, SYN, TCP, UDP, PacketInfo

BENIGN, SUSPICIOUS, ATTACK = "BENIGN", "SUSPICIOUS", "ATTACK"
ATTACKER = "192.168.1.44"
VICTIM = "192.168.1.35"
START = 1_700_000_000.0


def _tcp(ts, src, dst, sport, dport, flags, payload=0, window=64240, header=20) -> PacketInfo:
    return PacketInfo(
        timestamp=float(ts), src_ip=src, dst_ip=dst, src_port=int(sport), dst_port=int(dport),
        protocol=TCP, payload_len=int(payload), frame_len=int(payload) + 54, header_len=header,
        flags=int(flags), window=int(window),
    )


def _udp(ts, src, dst, sport, dport, payload) -> PacketInfo:
    return PacketInfo(
        timestamp=float(ts), src_ip=src, dst_ip=dst, src_port=int(sport), dst_port=int(dport),
        protocol=UDP, payload_len=int(payload), frame_len=int(payload) + 42, header_len=8,
        flags=0, window=-1,
    )


def _connection(rng, t, client, server, dport, *, exchanges=3, size=800) -> list[PacketInfo]:
    """A complete TCP connection: handshake, request and response, orderly close."""
    sport = int(rng.integers(49152, 65535))
    rtt = float(rng.uniform(0.002, 0.04))
    out = [
        _tcp(t, client, server, sport, dport, SYN, header=32),
        _tcp(t + rtt, server, client, dport, sport, SYN | ACK, window=65535, header=32),
        _tcp(t + 2 * rtt, client, server, sport, dport, ACK),
    ]
    now = t + 2 * rtt
    for _ in range(exchanges):
        now += float(rng.uniform(0.001, 0.05))
        out.append(_tcp(now, client, server, sport, dport, PSH | ACK, int(rng.integers(80, 500))))
        now += rtt
        out.append(_tcp(now, server, client, dport, sport, PSH | ACK,
                        int(rng.integers(size // 2, size * 2)), window=65535))
    now += float(rng.uniform(0.01, 0.1))
    out += [
        _tcp(now, client, server, sport, dport, FIN | ACK),
        _tcp(now + rtt, server, client, dport, sport, FIN | ACK, window=65535),
        _tcp(now + 2 * rtt, client, server, sport, dport, ACK),
    ]
    return out


def _public_host(rng) -> str:
    return f"{int(rng.choice([34, 104, 142, 151, 172]))}.{rng.integers(1, 250)}.{rng.integers(1, 250)}.{rng.integers(1, 250)}"


# ------------------------------------------------------------------ normal traffic
def normal_http(rng) -> list[PacketInfo]:
    out = []
    for index in range(int(rng.integers(5, 15))):
        out += _connection(rng, START + index * rng.uniform(0.5, 3), VICTIM, _public_host(rng), 80)
    return out


def normal_https(rng) -> list[PacketInfo]:
    out = []
    for index in range(int(rng.integers(5, 15))):
        out += _connection(rng, START + index * rng.uniform(0.5, 3), VICTIM, _public_host(rng), 443,
                           exchanges=int(rng.integers(3, 12)), size=1200)
    return out


def normal_dns(rng) -> list[PacketInfo]:
    out = []
    for index in range(int(rng.integers(10, 40))):
        t = START + index * float(rng.uniform(0.2, 2))
        sport = int(rng.integers(49152, 65535))
        out += [
            _udp(t, VICTIM, "192.168.1.1", sport, 53, int(rng.integers(30, 60))),
            _udp(t + float(rng.uniform(0.005, 0.05)), "192.168.1.1", VICTIM, 53, sport,
                 int(rng.integers(60, 300))),
        ]
    return out


def normal_ssdp(rng) -> list[PacketInfo]:
    """The discovery exchange that the model labelled DoS Slowhttptest."""
    out = []
    for round_index in range(int(rng.integers(2, 6))):
        t = START + round_index * 60.0
        sport = int(rng.integers(49152, 65535))
        for retry in range(4):
            out.append(_udp(t + retry, VICTIM, "239.255.255.250", sport, 1900, 170))
        for retry in range(4):
            out.append(_udp(t + 0.1 + retry, "192.168.1.128", VICTIM, 1900, sport, 338))
    return out


def many_answered_ports(rng) -> list[PacketInfo]:
    """A client using many ports on one server.  They all answer, so this isn't a scan."""
    out = []
    for index, port in enumerate(rng.choice(np.arange(1000, 9000), 40, replace=False)):
        out += _connection(rng, START + index * 0.5, VICTIM, "192.168.1.10", int(port))
    return out


def flaky_service(rng) -> list[PacketInfo]:
    """A client retrying a service that is down, a handful of times."""
    out = []
    for index in range(int(rng.integers(3, 12))):
        t = START + index * 2.0
        sport = int(rng.integers(49152, 65535))
        out += [
            _tcp(t, VICTIM, "192.168.1.10", sport, 8080, SYN, header=32),
            _tcp(t + 0.002, "192.168.1.10", VICTIM, 8080, sport, RST | ACK, window=0),
        ]
    return out


def browser_burst(rng) -> list[PacketInfo]:
    """A heavy web page: dozens of connections to many hosts in a few seconds."""
    hosts = [_public_host(rng) for _ in range(15)]
    out = []
    for _ in range(int(rng.integers(30, 60))):
        out += _connection(rng, START + float(rng.uniform(0, 5)), VICTIM, str(rng.choice(hosts)), 443)
    return out


def large_download(rng) -> list[PacketInfo]:
    """A fast transfer: a very high packet rate, but with large packets."""
    server, sport = _public_host(rng), int(rng.integers(49152, 65535))
    out = [
        _tcp(START, VICTIM, server, sport, 443, SYN, header=32),
        _tcp(START + 0.01, server, VICTIM, 443, sport, SYN | ACK, window=65535, header=32),
        _tcp(START + 0.02, VICTIM, server, sport, 443, ACK),
    ]
    for index in range(30000):
        t = START + 0.03 + index * 0.0002
        out.append(_tcp(t, server, VICTIM, 443, sport, ACK, 1460, window=65535))
        if index % 2 == 0:
            out.append(_tcp(t + 0.00005, VICTIM, server, sport, 443, ACK))
    return out


# ------------------------------------------------------------------- attack traffic
def _ports(rng, count: int) -> list[int]:
    return [int(p) for p in rng.choice(np.arange(1, 10000), count, replace=False)]


def _probe(rng, t, port, outcome) -> list[PacketInfo]:
    sport = int(rng.integers(32768, 60999))
    rtt = float(rng.uniform(0.002, 0.01))
    out = [_tcp(t, ATTACKER, VICTIM, sport, port, SYN, window=64240, header=40)]
    if outcome == "closed":
        out.append(_tcp(t + rtt, VICTIM, ATTACKER, port, sport, RST | ACK, window=0))
    elif outcome == "open":
        out += [
            _tcp(t + rtt, VICTIM, ATTACKER, port, sport, SYN | ACK, window=65535, header=32),
            _tcp(t + 2 * rtt, ATTACKER, VICTIM, sport, port, ACK, window=502, header=32),
            _tcp(t + 2 * rtt + 0.001, ATTACKER, VICTIM, sport, port, RST | ACK, window=0),
        ]
    return out


def scan_filtered(rng) -> list[PacketInfo]:
    """Like ``nmap -sT -T2`` against a firewall: nearly all probes get no reply."""
    out = []
    for index, port in enumerate(_ports(rng, int(rng.integers(250, 300)))):
        outcome = "open" if index % 80 == 0 else "filtered"
        out += _probe(rng, START + index * 0.4, port, outcome)
    return out


def scan_closed(rng) -> list[PacketInfo]:
    """A fast scan of a host with no firewall: closed ports answer with RST."""
    out = []
    for index, port in enumerate(_ports(rng, int(rng.integers(200, 400)))):
        out += _probe(rng, START + index * 0.01, port, "closed")
    return out


def scan_small(rng) -> list[PacketInfo]:
    """A short scan, above the suspicious threshold but below the attack threshold."""
    out = []
    for index, port in enumerate(_ports(rng, int(rng.integers(30, 80)))):
        out += _probe(rng, START + index * 0.4, port, "filtered")
    return out


def scan_below_threshold(rng) -> list[PacketInfo]:
    """Known miss: too few ports to tell from a retrying client."""
    out = []
    for index, port in enumerate(_ports(rng, int(rng.integers(10, 20)))):
        out += _probe(rng, START + index * 0.4, port, "filtered")
    return out


def scan_too_slow(rng) -> list[PacketInfo]:
    """Known miss: like ``nmap -T1``, slow enough to never fill the 120 s window."""
    out = []
    for index, port in enumerate(_ports(rng, 60)):
        out += _probe(rng, START + index * 15.0, port, "filtered")
    return out


def repeated_failures(rng) -> list[PacketInfo]:
    out = []
    for index in range(int(rng.integers(30, 60))):
        out += _probe(rng, START + index * 1.0, 22, "closed")
    return out


def repeated_successes(rng) -> list[PacketInfo]:
    """Many completed connections to one service, as a password-guessing tool makes."""
    out = []
    for index in range(int(rng.integers(70, 100))):
        out += _connection(rng, START + index * 1.0, ATTACKER, VICTIM, 22, exchanges=2, size=200)
    return out


def high_connection_rate(rng) -> list[PacketInfo]:
    hosts = [f"192.168.1.{n}" for n in range(50, 60)]
    out = []
    for index in range(int(rng.integers(300, 400))):
        out += _connection(rng, START + index * 0.03, ATTACKER, str(rng.choice(hosts)), 80, exchanges=1)
    return out


def syn_flood_pattern(rng) -> list[PacketInfo]:
    """Packets in memory only: no flood is generated on any network."""
    out = []
    for index in range(int(rng.integers(2500, 3500))):
        out.append(_tcp(START + index * 0.005, ATTACKER, VICTIM, 1024 + index % 60000, 80, SYN))
    return out


@dataclass(frozen=True)
class Scenario:
    name: str
    truth: str
    build: Callable[[np.random.Generator], list[PacketInfo]]
    note: str = ""


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("normal_http", BENIGN, normal_http),
    Scenario("normal_https", BENIGN, normal_https),
    Scenario("normal_dns", BENIGN, normal_dns),
    Scenario("normal_ssdp", BENIGN, normal_ssdp),
    Scenario("many_answered_ports", BENIGN, many_answered_ports, "hard negative"),
    Scenario("flaky_service", BENIGN, flaky_service, "hard negative"),
    Scenario("browser_burst", BENIGN, browser_burst, "hard negative"),
    Scenario("large_download", BENIGN, large_download, "hard negative"),
    Scenario("scan_filtered", ATTACK, scan_filtered),
    Scenario("scan_closed", ATTACK, scan_closed),
    Scenario("scan_small", SUSPICIOUS, scan_small),
    Scenario("repeated_failures", SUSPICIOUS, repeated_failures),
    Scenario("repeated_successes", SUSPICIOUS, repeated_successes),
    Scenario("high_connection_rate", SUSPICIOUS, high_connection_rate),
    Scenario("syn_flood_pattern", ATTACK, syn_flood_pattern),
    Scenario("scan_below_threshold", SUSPICIOUS, scan_below_threshold, "expected miss"),
    Scenario("scan_too_slow", ATTACK, scan_too_slow, "expected miss"),
)


def build_packets(scenario: Scenario, seed: int) -> list[PacketInfo]:
    packets = scenario.build(np.random.default_rng(seed))
    return sorted(packets, key=lambda packet: packet.timestamp)


__all__ = ["ATTACK", "BENIGN", "SUSPICIOUS", "SCENARIOS", "Scenario", "build_packets"]
