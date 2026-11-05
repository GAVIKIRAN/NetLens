"""Network interface discovery.

Windows interface names are unpredictable ("Ethernet", "Wi-Fi",
``\\Device\\NPF_{GUID}``), so nothing in this project hard-codes an adapter.
:func:`list_interfaces` returns everything Scapy can see, with the friendly
name, the description, the IPv4 address and the MAC, and
:func:`resolve_interface` accepts any of those spellings.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.errors import CaptureError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InterfaceInfo:
    """One capture-capable network interface."""

    name: str
    description: str = ""
    ip: str = ""
    mac: str = ""
    index: int | None = None
    is_loopback: bool = False

    def display(self) -> str:
        parts = [self.name]
        if self.description and self.description != self.name:
            parts.append(f"({self.description})")
        if self.ip:
            parts.append(f"[{self.ip}]")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "ip": self.ip,
            "mac": self.mac,
            "index": self.index,
            "is_loopback": self.is_loopback,
            "display": self.display(),
        }


def scapy_available() -> tuple[bool, str]:
    """Return ``(available, message)`` without raising."""
    try:
        import scapy.all  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"Scapy is not usable: {exc}"
    return True, "ok"


def list_interfaces() -> list[InterfaceInfo]:
    """Enumerate capture interfaces. Returns an empty list instead of raising."""
    ok, message = scapy_available()
    if not ok:
        logger.warning("Interface enumeration skipped: %s", message)
        return []

    interfaces: list[InterfaceInfo] = []
    system = platform.system().lower()

    if system == "windows":
        try:
            from scapy.arch.windows import get_windows_if_list

            for entry in get_windows_if_list():
                ips = entry.get("ips", []) or []
                ipv4 = next((ip for ip in ips if ":" not in str(ip)), "")
                interfaces.append(
                    InterfaceInfo(
                        name=str(entry.get("name") or entry.get("description") or ""),
                        description=str(entry.get("description", "")),
                        ip=str(ipv4),
                        mac=str(entry.get("mac", "")),
                        index=entry.get("index"),
                        is_loopback="loopback" in str(entry.get("description", "")).lower(),
                    )
                )
        except Exception as exc:  # pragma: no cover - Windows only
            logger.warning("Windows interface enumeration failed: %s", exc)

    if not interfaces:
        try:
            from scapy.all import conf, get_if_addr, get_if_hwaddr, get_if_list

            for name in get_if_list():
                ip = mac = ""
                try:
                    ip = get_if_addr(name)
                except Exception:
                    pass
                try:
                    mac = get_if_hwaddr(name)
                except Exception:
                    pass
                interfaces.append(
                    InterfaceInfo(
                        name=str(name),
                        description=str(name),
                        ip="" if ip in ("0.0.0.0", None) else str(ip),
                        mac=str(mac or ""),
                        is_loopback=str(name).startswith("lo"),
                    )
                )
            _ = conf  # keep the import meaningful for older scapy versions
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Interface enumeration failed: %s", exc)

    logger.info("Found %d capture interface(s)", len(interfaces))
    return interfaces


def capture_driver_status() -> tuple[bool, str]:
    """Look for the packet-capture driver without opening a device."""
    if sys.platform.startswith("win"):
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        for path in (
            system_root / "System32" / "Npcap" / "wpcap.dll",
            system_root / "System32" / "wpcap.dll",
        ):
            if path.exists():
                return True, f"Npcap found at {path}"
        return False, "Npcap not found; install it from https://npcap.com and reboot"
    library = ctypes.util.find_library("pcap")
    if library:
        return True, f"libpcap found ({library})"
    return False, "libpcap not found; install libpcap (e.g. apt install libpcap0.8)"


def rank_interfaces(
    interfaces: list[InterfaceInfo], preferred: str | None = None
) -> list[InterfaceInfo]:
    """Keep adapters that are really on a network, with the default one first.

    Windows reports dozens of virtual adapters and driver filter entries.  An
    adapter with no address, a loopback or a self-assigned 169.254.x.x address is
    not carrying your traffic, so it only makes the list misleading.  If nothing
    qualifies, the full list is returned rather than an empty one.
    """
    usable = [
        interface
        for interface in interfaces
        if interface.ip
        and not interface.is_loopback
        and not interface.ip.startswith(("169.254.", "127."))
    ]
    if not usable:
        return list(interfaces)
    return sorted(usable, key=lambda interface: interface.name != preferred)


def default_interface() -> str | None:
    """Scapy's own default interface name, or ``None``."""
    try:
        from scapy.all import conf

        value = getattr(conf, "iface", None)
        if value is None:
            return None
        name = getattr(value, "name", None)
        return str(name if name else value)
    except Exception:  # pragma: no cover - environment dependent
        return None


def resolve_interface(requested: str | None) -> str | None:
    """Resolve a user-supplied interface string to something Scapy accepts.

    Accepts the friendly name, the description or the IPv4 address.  ``None``
    means "let Scapy pick its default", which is the shipped configuration.
    """
    if requested in (None, "", "auto", "default"):
        return None
    wanted = str(requested).strip()
    interfaces = list_interfaces()
    if not interfaces:
        return wanted  # nothing to validate against; hand it to Scapy as-is

    for interface in interfaces:
        if wanted.lower() == interface.name.lower():
            return interface.name
    for interface in interfaces:
        if wanted.lower() == interface.description.lower() or wanted == interface.ip:
            return interface.name
    for interface in interfaces:
        if wanted.lower() in interface.description.lower():
            return interface.name

    available = ", ".join(i.name for i in interfaces[:12])
    raise CaptureError(
        f"Network interface '{requested}' was not found.",
        hint=f"Run `python scripts/list_interfaces.py`. Available: {available}",
    )


__all__ = [
    "InterfaceInfo",
    "list_interfaces",
    "default_interface",
    "capture_driver_status",
    "rank_interfaces",
    "resolve_interface",
    "scapy_available",
]
