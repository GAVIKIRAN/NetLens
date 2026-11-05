"""Shared bootstrap for the command-line scripts.

Adds the project root to ``sys.path`` so the scripts can be executed directly
(``python scripts/train_model.py``) without installing the package, and wires
logging from ``config/config.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def bootstrap(config_path: str | None = None):
    """Load settings, configure logging, create directories. Returns Settings."""
    from src.config.settings import get_settings
    from src.utils.logger import configure_from_mapping

    settings = get_settings(config_path, reload=bool(config_path))
    configure_from_mapping(settings.section("logging"), settings.path("log_dir"))
    settings.ensure_directories()
    return settings


__all__ = ["bootstrap", "PROJECT_ROOT"]
