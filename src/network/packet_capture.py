"""Live packet capture (Phase 6) - strictly read-only.

This module only ever *reads* from the network.  It never builds, injects or
transmits a packet; Scapy's sending functions are not imported anywhere in this
project.  That is a hard design constraint: the system is a defensive monitor.

Windows notes
-------------
* Scapy needs the **Npcap** driver (https://npcap.com).  Install it with
  "WinPcap API-compatible Mode" enabled.
* Capture requires **Administrator** rights.  Without them Npcap raises a
  permission error, which is translated into a clear
  :class:`~src.utils.errors.CaptureError` rather than a stack trace.
* Adapter names are unpredictable, so use ``python scripts/list_interfaces.py``
  and copy the exact name into ``network.interface``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from src.network.feature_extractor import packet_to_info
from src.network.flow_manager import PacketInfo
from src.network.interfaces import resolve_interface, scapy_available
from src.utils.errors import CaptureError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PacketHandler = Callable[[PacketInfo], None]


class PacketCapture:
    """Background Scapy sniffer that hands :class:`PacketInfo` to a callback."""

    def __init__(
        self,
        on_packet: PacketHandler,
        *,
        interface: str | None = None,
        bpf_filter: str = "ip",
        duration: float = 0.0,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.on_packet = on_packet
        self.requested_interface = interface
        self.interface: str | None = None
        self.bpf_filter = bpf_filter or None
        self.duration = float(duration or 0.0)
        self.on_error = on_error

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started_at = 0.0
        self.packet_count = 0
        self.dropped_count = 0
        self.last_error: str = ""

    # --------------------------------------------------------------- start
    def start(self) -> None:
        """Validate the environment and start the capture thread."""
        available, message = scapy_available()
        if not available:
            raise CaptureError(
                "Live capture is unavailable because Scapy could not be imported.",
                hint=f"{message}. Install it with `pip install scapy`.",
            )
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Capture already running; start() ignored.")
            return

        self.interface = resolve_interface(self.requested_interface)
        self._stop.clear()
        self.packet_count = 0
        self.last_error = ""
        self._started_at = time.time()

        self._thread = threading.Thread(
            target=self._run, name="nids-capture", daemon=True
        )
        self._thread.start()
        logger.info(
            "Packet capture started (interface=%s, filter=%s, duration=%s)",
            self.interface or "<scapy default>",
            self.bpf_filter or "<none>",
            self.duration or "unlimited",
        )

    # ---------------------------------------------------------------- stop
    def stop(self, timeout: float = 5.0) -> None:
        """Signal the sniffer to stop and wait for the thread to finish."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("Packet capture stopped after %d packet(s)", self.packet_count)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        return max(0.0, time.time() - self._started_at) if self._started_at else 0.0

    # ----------------------------------------------------------- internals
    def _run(self) -> None:
        try:
            from scapy.all import sniff
        except Exception as exc:  # pragma: no cover - guarded in start()
            self._fail(CaptureError(f"Scapy import failed: {exc}"))
            return

        kwargs: dict[str, Any] = {
            "prn": self._handle,
            "store": False,
            "stop_filter": lambda _pkt: self._stop.is_set(),
        }
        if self.interface:
            kwargs["iface"] = self.interface
        if self.bpf_filter:
            kwargs["filter"] = self.bpf_filter
        if self.duration and self.duration > 0:
            kwargs["timeout"] = self.duration

        try:
            sniff(**kwargs)
        except PermissionError as exc:
            self._fail(
                CaptureError(
                    "Permission denied while opening the capture device.",
                    hint="Run the terminal as Administrator (Windows) or with "
                         "sudo/CAP_NET_RAW (Linux).",
                )
            )
            logger.debug("Underlying permission error: %s", exc)
        except OSError as exc:
            self._fail(
                CaptureError(
                    f"The capture device could not be opened: {exc}",
                    hint="Check the interface name and that Npcap is installed.",
                )
            )
        except Exception as exc:  # pragma: no cover - driver specific
            message = str(exc).lower()
            if "npcap" in message or "winpcap" in message or "pcap" in message:
                self._fail(
                    CaptureError(
                        f"The packet-capture driver reported an error: {exc}",
                        hint="Install Npcap from https://npcap.com and reboot.",
                    )
                )
            else:
                self._fail(CaptureError(f"Packet capture failed: {exc}"))
        finally:
            self._stop.set()

    def _handle(self, packet: Any) -> None:
        try:
            info = packet_to_info(packet)
        except Exception:
            self.dropped_count += 1
            return
        if info is None:
            self.dropped_count += 1
            return
        self.packet_count += 1
        try:
            self.on_packet(info)
        except Exception:
            logger.exception("Packet handler raised; the packet was dropped")
            self.dropped_count += 1

    def _fail(self, error: Exception) -> None:
        self.last_error = str(error)
        logger.error("Capture error: %s", error)
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:  # pragma: no cover - defensive
                logger.exception("on_error callback raised")

    def stats(self) -> dict[str, Any]:
        return {
            "running": self.is_running,
            "interface": self.interface or "<scapy default>",
            "bpf_filter": self.bpf_filter or "",
            "packets": self.packet_count,
            "dropped": self.dropped_count,
            "elapsed_seconds": round(self.elapsed, 1),
            "last_error": self.last_error,
        }


def read_pcap(path: str | Path, on_packet: PacketHandler, limit: int | None = None) -> int:
    """Replay a saved capture file offline. Returns the number of packets fed.

    This is the safe way to demonstrate detection on attack traffic: analyse a
    capture that already exists (for example one of the public CIC-IDS2017
    PCAPs) instead of generating anything on a live network.
    """
    available, message = scapy_available()
    if not available:
        raise CaptureError(
            "Reading a PCAP requires Scapy.", hint=f"{message}. `pip install scapy`."
        )
    file_path = Path(path)
    if not file_path.exists():
        raise CaptureError(f"PCAP file not found: {file_path}")

    try:
        from scapy.all import PcapReader
    except Exception as exc:  # pragma: no cover - scapy internals
        raise CaptureError(f"Scapy could not provide a PCAP reader: {exc}") from exc

    count = 0
    try:
        with PcapReader(str(file_path)) as reader:
            for packet in reader:
                info = packet_to_info(packet)
                if info is None:
                    continue
                on_packet(info)
                count += 1
                if limit and count >= int(limit):
                    break
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"Could not read {file_path.name}: {exc}") from exc
    logger.info("Replayed %s packet(s) from %s", f"{count:,}", file_path.name)
    return count


__all__ = ["PacketCapture", "read_pcap", "PacketHandler"]
