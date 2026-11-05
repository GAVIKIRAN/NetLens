"""Print every capture interface Scapy can see on this machine.

Copy the exact 'name' value into ``network.interface`` in config/config.yaml
(or set the NIDS_NETWORK__INTERFACE environment variable).

    python scripts/list_interfaces.py
"""

from __future__ import annotations

from _bootstrap import bootstrap


def main() -> int:
    bootstrap()

    from src.network.interfaces import default_interface, list_interfaces, scapy_available

    available, message = scapy_available()
    if not available:
        print(f"\n[WARNING] {message}")
        print("Install it with: pip install scapy\n")
        return 2

    interfaces = list_interfaces()
    if not interfaces:
        print("\nNo interfaces were reported.")
        print("On Windows this usually means Npcap is not installed:")
        print("  https://npcap.com  (enable 'WinPcap API-compatible Mode')\n")
        return 2

    print(f"\n{len(interfaces)} capture interface(s):\n")
    print(f"  {'NAME':<42s} {'IPv4':<16s} {'MAC':<18s} DESCRIPTION")
    print("  " + "-" * 110)
    for interface in interfaces:
        print(
            f"  {interface.name[:42]:<42s} {interface.ip[:16]:<16s} "
            f"{interface.mac[:18]:<18s} {interface.description[:40]}"
        )
    print(f"\n  Scapy default: {default_interface()}")
    print("\n  Live capture also needs Administrator rights on Windows.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
