"""Demo attack self-test — generate a REAL port-scan against your OWN host so a
live NIDS capture can detect it. For authorised testing of your own system only.

Safety: the target is restricted to loopback (127.0.0.1 / ::1) or a private-LAN
address (10/8, 172.16/12, 192.168/16). Public / internet addresses are refused,
so this cannot be pointed at machines you do not own.

Two modes, both TCP (the only protocol this NIDS detects):
  * scan  (default) — connect to many ports in sequence. Closed ports refuse, which
           the behaviour layer counts as failed: 100+ ports, 60%+ failed => ATTACK
           (port_scan rule).
  * burst            — connect repeatedly, fast, to ONE closed port. Trips the
           failed_connections (>=20 failed, >=80%), repeated_attempts (>=60) and
           connection_rate (>=20/s) rules.

Single machine (capture on the loopback adapter):
    python scripts/demo_attack_selftest.py                     # scans 127.0.0.1
    python scripts/demo_attack_selftest.py --mode burst

From a SECOND device you own, aimed at the monitor over Wi-Fi (capture on Wi-Fi):
    python demo_attack_selftest.py --target 192.168.1.34

NOT supported by this NIDS, so do not expect a detection: UDP/ICMP scans or floods
(the model was trained only on TCP attacks; the behaviour layer tracks TCP only).
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import time


def _check_target(target: str) -> str:
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(target))
    except (OSError, ValueError) as exc:
        sys.exit(f"Refusing: cannot resolve '{target}' to an IP ({exc}).")
    if not (ip.is_loopback or ip.is_private):
        sys.exit(
            f"Refusing: {ip} is not loopback or a private-LAN address. "
            "This tool only tests your own host / LAN."
        )
    return str(ip)


def main() -> int:
    p = argparse.ArgumentParser(description="TCP attack self-test against your own host.")
    p.add_argument("--target", default="127.0.0.1", help="your own host (loopback or private LAN)")
    p.add_argument("--mode", choices=("scan", "burst"), default="scan",
                   help="scan = many ports (port_scan); burst = one port, fast (rate/failed rules)")
    p.add_argument("--start-port", type=int, default=3000)
    p.add_argument("--ports", type=int, default=130, help="scan: distinct ports (>=100 => ATTACK)")
    p.add_argument("--count", type=int, default=120, help="burst: connection attempts to one port")
    p.add_argument("--timeout", type=float, default=0.25, help="per-connection timeout (s)")
    p.add_argument("--delay", type=float, default=0.02, help="scan: delay between connections (s)")
    args = p.parse_args()

    ip = _check_target(args.target)

    def attempt(port: int, timeout: float) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            return s.connect_ex((ip, port)) == 0
        except OSError:
            return False
        finally:
            s.close()

    opened = failed = 0
    if args.mode == "scan":
        print(f"Target {ip} — scanning {args.ports} ports from {args.start_port}. "
              "Authorised self-test only.\n")
        for i in range(args.ports):
            if attempt(args.start_port + i, args.timeout):
                opened += 1
            else:
                failed += 1
            if (i + 1) % 25 == 0:
                print(f"  {i + 1:3d} ports  open={opened}  failed={failed}")
            time.sleep(args.delay)
        hint = "a PORT SCAN (ATTACK at >=100 ports, >=60% failed)."
    else:  # burst — one port, as fast as possible
        port = args.start_port
        print(f"Target {ip}:{port} — {args.count} rapid connection attempts. "
              "Authorised self-test only.\n")
        burst_timeout = min(args.timeout, 0.05)  # keep the SYN rate high for connection_rate
        start = time.time()
        for i in range(args.count):
            if attempt(port, burst_timeout):
                opened += 1
            else:
                failed += 1
            if (i + 1) % 30 == 0:
                print(f"  {i + 1:3d} attempts  open={opened}  failed={failed}")
        rate = args.count / max(time.time() - start, 1e-6)
        print(f"  rate ~{rate:.0f} attempts/s")
        hint = ("REPEATED FAILED CONNECTIONS / high connection rate "
                "(>=20 failed & >=80%, or >=20/s, or >=60 attempts).")

    total = opened + failed
    print(f"\nDone. {total} connections — open={opened}, failed={failed} "
          f"({failed / total:.0%} failed).")
    print(f"Expect the Live Monitor behaviour layer to flag {hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
