"""The capture path from real packet bytes, without opening a network interface.

Packets are built with Scapy and written to a file; nothing is transmitted.  They
are read back through the same parser that live capture uses, so these tests
cover everything except the driver itself.
"""

from __future__ import annotations

import time

import pytest

from src.database.database import Database
from src.network import interfaces as interfaces_module
from src.network.feature_extractor import flow_to_features
from src.network.flow_manager import FlowManager
from src.network.interfaces import InterfaceInfo, rank_interfaces
from src.network.monitor import MODE_LIVE, MODE_PCAP, MonitorService
from src.network.packet_capture import read_pcap
from src.utils.errors import CaptureError

CLIENT, SERVER = "192.168.1.35", "93.184.216.34"
# Fixed addresses: without them Scapy looks the MAC up by sending an ARP request.
FRAME = {"src": "02:00:00:00:00:01", "dst": "02:00:00:00:00:02"}


def _stamped(packet, when):
    packet.time = when
    return packet


@pytest.fixture
def pcap_file(tmp_path):
    from scapy.all import DNS, DNSQR, Ether, IP, TCP, UDP, wrpcap

    def forward(flags, seq, payload=b"", window=64240):
        return Ether(**FRAME) / IP(src=CLIENT, dst=SERVER) / TCP(
            sport=50000, dport=443, flags=flags, seq=seq, window=window
        ) / payload

    def backward(flags, seq, payload=b"", window=65535):
        return Ether(**FRAME) / IP(src=SERVER, dst=CLIENT) / TCP(
            sport=443, dport=50000, flags=flags, seq=seq, window=window
        ) / payload

    packets = [
        _stamped(forward("S", 1), 100.0),
        _stamped(backward("SA", 1), 100.1),
        _stamped(forward("A", 2), 100.2),
        _stamped(forward("PA", 2, b"x" * 300), 100.3),
        _stamped(backward("PA", 2, b"y" * 1000), 100.4),
        _stamped(forward("FA", 302), 100.5),
        _stamped(backward("FA", 1002), 100.6),
        _stamped(Ether(**FRAME) / IP(src=CLIENT, dst="1.1.1.1") / UDP(sport=53000, dport=53)
                 / DNS(qd=DNSQR(qname="example.com")), 101.0),
        _stamped(Ether(**FRAME) / IP(src="1.1.1.1", dst=CLIENT) / UDP(sport=53, dport=53000)
                 / DNS(qd=DNSQR(qname="example.com")), 101.1),
    ]
    path = tmp_path / "sample.pcap"
    wrpcap(str(path), packets)
    return path


def test_real_packets_become_correct_flow_features(pcap_file):
    completed = []
    manager = FlowManager(on_flow_complete=completed.append, min_packets_to_emit=2)
    assert read_pcap(pcap_file, manager.add_packet) == 9
    manager.flush()

    assert len(completed) == 2
    tcp = next(flow for flow in completed if flow.protocol == 6)
    assert tcp.termination_reason == "fin"
    assert tcp.src_ip == CLIENT

    features = flow_to_features(tcp)
    assert features["total_fwd_packets"] == 4
    assert features["total_backward_packets"] == 3
    assert features["total_length_of_fwd_packets"] == 300
    assert features["total_length_of_bwd_packets"] == 1000
    assert features["flow_duration"] == pytest.approx(600_000.0, rel=1e-3)
    assert features["init_win_bytes_forward"] == 64240
    assert features["init_win_bytes_backward"] == 65535
    assert features["act_data_pkt_fwd"] == 1

    udp = next(flow for flow in completed if flow.protocol == 17)
    assert flow_to_features(udp)["init_win_bytes_forward"] == -1


def test_real_packets_reach_prediction_and_database(settings, predictor, pcap_file, tmp_path):
    database = Database(tmp_path / "capture.db")
    monitor = MonitorService(settings, database=database, predictor=predictor)
    try:
        monitor.start(MODE_PCAP, pcap_path=pcap_file)
        deadline = time.time() + 15
        while time.time() < deadline and monitor.is_running:
            time.sleep(0.1)
        assert monitor.stats.errors == 0
        assert monitor.stats.packets_seen == 9
        assert monitor.stats.flows_processed == 2
        assert database.counts()["predictions"] == 2
    finally:
        monitor.stop()
        database.close()


def test_live_mode_explains_a_missing_driver(settings, predictor, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.network.monitor.capture_driver_status", lambda: (False, "Npcap not found")
    )
    database = Database(tmp_path / "live.db")
    monitor = MonitorService(settings, database=database, predictor=predictor)
    try:
        with pytest.raises(CaptureError, match="driver"):
            monitor.start(MODE_LIVE)
        assert not monitor.is_running
    finally:
        database.close()


def test_building_the_sample_transmits_nothing(tmp_path, monkeypatch):
    """Any attempt to look up a MAC or send a frame fails this test."""
    import scapy.layers.l2 as l2
    import scapy.sendrecv as sendrecv

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the test tried to use the network")

    monkeypatch.setattr(l2, "getmacbyip", forbidden)
    for name in ("send", "sendp", "sr", "sr1", "srp", "srp1"):
        monkeypatch.setattr(sendrecv, name, forbidden)
    from scapy.all import Ether, IP, TCP, wrpcap

    frame = Ether(**FRAME) / IP(src=CLIENT, dst=SERVER) / TCP(sport=50000, dport=443, flags="S")
    wrpcap(str(tmp_path / "one.pcap"), [frame])


def test_rank_interfaces_keeps_real_adapters_with_the_default_first():
    adapters = [
        InterfaceInfo("Wi-Fi 3", ip="169.254.223.10"),
        InterfaceInfo("Hyper-V", ip="172.20.0.1"),
        InterfaceInfo("Wi-Fi", ip="192.168.1.35"),
        InterfaceInfo("Filter-0000", ip=""),
        InterfaceInfo("Loopback", ip="127.0.0.1", is_loopback=True),
    ]
    assert [i.name for i in rank_interfaces(adapters, "Wi-Fi")] == ["Wi-Fi", "Hyper-V"]


def test_rank_interfaces_never_returns_an_empty_list():
    adapters = [InterfaceInfo("Filter-0000", ip="")]
    assert rank_interfaces(adapters, None) == adapters


def test_driver_status_reports_a_missing_driver(monkeypatch):
    monkeypatch.setattr(interfaces_module.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", r"Z:\does-not-exist")
    ok, message = interfaces_module.capture_driver_status()
    assert not ok
    assert "npcap.com" in message
