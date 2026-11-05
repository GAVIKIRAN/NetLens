"""Centralised logging configuration.

A single call to :func:`configure_logging` installs a rotating file handler and
an optional console handler. Every other module simply calls
:func:`get_logger(__name__)`.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Mapping

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"


def configure_logging(
    log_dir: str | Path = "logs",
    level: str = "INFO",
    console: bool = True,
    to_file: bool = True,
    filename: str = "nids.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    fmt: str = _DEFAULT_FORMAT,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger. Safe to call repeatedly (idempotent).

    Parameters
    ----------
    log_dir:
        Directory that will hold the rotating log files. Created if missing.
    level:
        Root log level name, e.g. ``"DEBUG"``.
    console:
        Also mirror records to ``stderr``.
    to_file:
        Write records to ``log_dir/filename`` with rotation.
    force:
        Re-configure even if configuration already happened.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = logging.Formatter(fmt)

    if to_file:
        try:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                path / filename,
                maxBytes=int(max_bytes),
                backupCount=int(backup_count),
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            print(f"[logger] file logging disabled: {exc}", file=sys.stderr)

    if console:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    # Third-party libraries are extremely chatty at INFO level.
    for noisy in ("matplotlib", "shap", "numba", "PIL", "urllib3", "lime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def configure_from_mapping(cfg: Mapping[str, Any], log_dir: str | Path) -> logging.Logger:
    """Configure logging from the ``logging:`` block of ``config.yaml``."""
    return configure_logging(
        log_dir=log_dir,
        level=str(cfg.get("level", "INFO")),
        console=bool(cfg.get("console", True)),
        to_file=bool(cfg.get("file", True)),
        filename=str(cfg.get("filename", "nids.log")),
        max_bytes=int(cfg.get("max_bytes", 10 * 1024 * 1024)),
        backup_count=int(cfg.get("backup_count", 5)),
        fmt=str(cfg.get("format", _DEFAULT_FORMAT)),
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "configure_from_mapping", "get_logger"]
