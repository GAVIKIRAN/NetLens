"""Capture-path smoke test - no model or dashboard required.

Captures for a few seconds (or replays a PCAP) and prints the flows that were
assembled together with a handful of extracted features.  Use it to confirm
that Npcap/permissions/interface selection work before starting the dashboard.

    python scripts/test_live_capture.py --duration 15
    python scripts/test_live_capture.py --interface "Wi-Fi" --duration 20
    python scripts/test_live_capture.py --pcap captures\\sample.pcap
    python scripts/test_live_capture.py --simulate      (no network at all)
"""

from __future__ import annotations

import argparse
import sys
import time

from _bootstrap import bootstrap

PREVIEW_FEATURES = (
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "syn_flag_count",
    "packet_length_mean",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the capture pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--pcap", default=None, help="replay a saved capture instead")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="use in-memory synthetic packets; touches no network interface",
    )
    parser.add_argument("--limit", type=int, default=10, help="flows to print")
    args = parser.parse_args()

    settings = bootstrap(args.config)

    from src.network.feature_extractor import flow_to_features
    from src.network.flow_manager import FlowManager
    from src.network.packet_capture import PacketCapture, read_pcap
    from src.utils.errors import NIDSError

    network_cfg = settings.section("network")
    completed = []

    manager = FlowManager(
        flow_timeout=float(network_cfg.get("flow_timeout", 120.0)),
        activity_timeout=float(network_cfg.get("activity_timeout", 5.0)),
        subflow_timeout=float(network_cfg.get("subflow_timeout", 1.0)),
        on_flow_complete=completed.append,
        min_packets_to_emit=int(network_cfg.get("min_packets_per_flow", 2)),
    )

    try:
        if args.simulate:
            from src.simulation.synthetic import SyntheticFlowGenerator

            print("\nSimulated mode: generating in-memory packets (no capture).\n")
            generator = SyntheticFlowGenerator(seed=7)
            flows = generator.generate(args.limit, attack_ratio=0.5)
            for flow in flows:
                print(
                    f"  {flow.label:<26s} {flow.identity['src_ip']}:{flow.identity['src_port']}"
                    f" -> {flow.identity['dst_ip']}:{flow.identity['dst_port']}"
                    f"  packets={flow.identity['packets']:<5d} bytes={flow.identity['bytes']:<8d}"
                )
            print()
            for name in PREVIEW_FEATURES:
                print(f"    {name:<28s} {flows[0].features[name]:,.3f}")
            print()
            return 0

        if args.pcap:
            print(f"\nReplaying {args.pcap} ...")
            count = read_pcap(args.pcap, manager.add_packet)
            manager.flush()
            print(f"  {count:,} packet(s) read, {len(completed)} flow(s) assembled.")
        else:
            capture = PacketCapture(
                on_packet=manager.add_packet,
                interface=args.interface if args.interface else network_cfg.get("interface"),
                bpf_filter=str(network_cfg.get("bpf_filter", "ip") or ""),
                duration=args.duration,
            )
            print(
                f"\nCapturing for {args.duration:.0f}s on "
                f"{args.interface or network_cfg.get('interface') or '<scapy default>'} ..."
            )
            capture.start()
            deadline = time.time() + args.duration
            while time.time() < deadline and capture.is_running:
                time.sleep(0.5)
                manager.expire()
            capture.stop()
            manager.flush()
            print(f"  Captured {capture.packet_count:,} packet(s).")
            if capture.last_error:
                print(f"  [ERROR] {capture.last_error}", file=sys.stderr)
                return 2
    except NIDSError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 2

    if not completed:
        print("\n  No flows completed. Try a longer duration or generate some traffic")
        print("  by browsing the web while the capture runs.\n")
        return 0

    print(f"\n  {len(completed)} flow(s) assembled. First {min(args.limit, len(completed))}:\n")
    for flow in completed[: args.limit]:
        identity = flow.identity()
        print(
            f"  {identity['protocol_name']:<5s} "
            f"{identity['src_ip']}:{identity['src_port']} -> "
            f"{identity['dst_ip']}:{identity['dst_port']}  "
            f"packets={identity['packets']:<6d} bytes={identity['bytes']:<9d} "
            f"duration={identity['duration_seconds']:.3f}s ({identity['termination']})"
        )

    features = flow_to_features(completed[0])
    print("\n  Extracted features (first flow):")
    for name in PREVIEW_FEATURES:
        print(f"    {name:<28s} {features[name]:,.3f}")
    print(f"\n  {len(features)} features extracted per flow.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
