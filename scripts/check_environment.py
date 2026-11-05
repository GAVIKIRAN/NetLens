"""Environment doctor - run this first if anything misbehaves.

    python scripts/check_environment.py

Reports the Python version, every required and optional package, the capture
driver (Npcap or libpcap) and capture privilege.  Exits non-zero only on a
*blocking* problem.  Missing capture support is reported but is not blocking,
because DEMO and PCAP modes do not need it.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
import sys
from pathlib import Path

from _bootstrap import bootstrap

REQUIRED = (
    "numpy", "pandas", "scipy", "sklearn", "joblib", "yaml", "matplotlib", "plotly", "streamlit",
)
OPTIONAL = ("xgboost", "shap", "lime", "scapy", "pytest")


def _check(module: str) -> tuple[bool, str]:
    try:
        loaded = importlib.import_module(module)
        return True, getattr(loaded, "__version__", "installed")
    except Exception as exc:
        return False, str(exc)


def capture_privilege() -> tuple[bool, str]:
    """Report whether this process has the rights raw capture normally needs."""
    if sys.platform.startswith("win"):
        try:
            elevated = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError) as exc:
            return False, f"could not determine ({exc})"
        if elevated:
            return True, "running as Administrator"
        return False, "not Administrator; open the terminal with 'Run as administrator'"
    if os.geteuid() == 0:
        return True, "running as root"
    return False, "not root; use sudo or setcap cap_net_raw,cap_net_admin+eip on Python"


def main() -> int:
    print("\n" + "=" * 72)
    print("  XAI-NIDS environment check")
    print("=" * 72)
    print(f"  Python   : {platform.python_version()} ({sys.executable})")
    print(f"  Platform : {platform.platform()}")

    failures = 0
    if sys.version_info < (3, 11):
        print("  [BLOCKING] Python 3.11+ is required.")
        failures += 1

    print("\n  Required packages")
    for module in REQUIRED:
        ok, detail = _check(module)
        print(f"    {'OK  ' if ok else 'MISS'}  {module:<14s} {detail[:52]}")
        failures += 0 if ok else 1

    print("\n  Optional packages")
    for module in OPTIONAL:
        ok, detail = _check(module)
        print(f"    {'OK  ' if ok else 'MISS'}  {module:<14s} {detail[:52]}")

    print("\n  Configuration and artefacts")
    try:
        settings = bootstrap()
        print(f"    OK    config           {settings.source}")
        for key in ("raw_data_dir", "model_dir", "database_path"):
            path = settings.path(key)
            marker = "OK  " if path.exists() else "MISS"
            print(f"    {marker}  {key:<16s} {path}")
        from src.inference.predictor import Predictor

        trained = Predictor.artefacts_exist(settings)
        print(f"    {'OK  ' if trained else 'MISS'}  trained model    "
              f"{'ready' if trained else 'run scripts/train_model.py'}")
        demo_file = settings.get("demo.source_file")

        demo_ready = Path(demo_file).exists()
        print(f"    {'OK  ' if demo_ready else 'MISS'}  demo data        "
              f"{'ready' if demo_ready else 'run scripts/make_demo_data.py'}")
    except Exception as exc:
        print(f"    FAIL  configuration     {exc}")
        failures += 1

    print("\n  Capture capability (needed for LIVE mode only)")
    from src.network.interfaces import capture_driver_status, default_interface

    driver_ok, driver = capture_driver_status()
    print(f"    {'OK  ' if driver_ok else 'MISS'}  driver           {driver}")
    print(f"          default interface: {default_interface() or 'unknown'}")
    privilege_ok, privilege = capture_privilege()
    # Npcap's default install lets normal users capture, so this is only a note.
    print(f"    {'OK  ' if privilege_ok else 'NOTE'}  privilege        {privilege}")
    if not privilege_ok:
        print("          Only needed if capture reports 'permission denied'.")
    try:
        from src.network.interfaces import list_interfaces, scapy_available

        ok, message = scapy_available()
        if ok:
            count = len(list_interfaces())
            print(f"    OK    scapy            {count} interface(s) visible")
        else:
            print(f"    MISS  scapy            {message}")
    except Exception as exc:
        print(f"    MISS  scapy            {exc}")
    print(f"    LIVE mode: {'ready' if driver_ok else 'not ready'}.  "
          "DEMO and PCAP modes do not need capture support.")

    print("=" * 72)
    print("  No blocking problems.\n" if failures == 0
          else f"  {failures} blocking problem(s) - see above.\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
