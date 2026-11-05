"""Small, dependency-light helpers shared across the project."""

from __future__ import annotations

import json
import math
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Characters that appear in CIC-IDS2017 headers and must be normalised away.
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def normalise_column(name: str) -> str:
    """Normalise a raw CSV column header to the project's canonical form.

    The published CIC-IDS2017 files are inconsistent: some have a leading
    space (``" Flow Duration"``), some use ``/`` (``"Flow Bytes/s"``), some use
    underscores (``"Init_Win_bytes_forward"``) and some mix capitalisation.
    Everything is lower-cased and reduced to ``snake_case``.

    >>> normalise_column(" Flow Bytes/s")
    'flow_bytes_s'
    >>> normalise_column("Init_Win_bytes_forward")
    'init_win_bytes_forward'
    >>> normalise_column("Fwd Header Length.1")
    'fwd_header_length_1'
    """
    cleaned = _NON_ALNUM.sub("_", str(name).strip())
    return cleaned.strip("_").lower()


def normalise_columns(columns: Iterable[str]) -> list[str]:
    """Normalise a whole header row, de-duplicating collisions deterministically."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        base = normalise_column(col)
        if base in seen:
            seen[base] += 1
            base = f"{base}_dup{seen[base]}"
        else:
            seen[base] = 0
        out.append(base)
    return out


def normalise_label(label: Any) -> str:
    """Canonicalise a class label.

    CIC-IDS2017 contains encoding artefacts in the web-attack labels.  Depending
    on the copy, the dash is a Windows-1252 byte (``\\x96``) or an already-damaged
    ``EF BF BD`` sequence, which reads as ``\u00ef\u00bf\u00bd`` when the file is decoded as
    latin-1.  All dash variants collapse to ``" - "`` and surrounding whitespace
    is trimmed.
    """
    text = str(label).strip()
    # Only the damaged and Unicode dashes are rewritten. ASCII hyphens are left
    # alone so that "SSH-Patator" and "FTP-Patator" keep their published
    # spelling (and keep matching the severity map in config.yaml).
    for dash in (
        "\u00ef\u00bf\u00bd", "\ufffd", "\u00e2\u0080\u0093", "\u00e2\u0080\u0094",
        "\x96", "\u2013", "\u2014", "\u2012", "\u2212",
    ):
        text = text.replace(dash, " - ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_local_datetime(value: Any) -> datetime | None:
    """Best-effort parse of a timestamp coming out of the database."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises and never returns inf/NaN."""
    try:
        if denominator == 0 or denominator is None:
            return default
        result = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp ``value`` into ``[lower, upper]``, mapping NaN to ``lower``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return lower
    if math.isnan(value):
        return lower
    return max(lower, min(upper, value))


def replace_non_finite(frame: pd.DataFrame, fill: float = np.nan) -> pd.DataFrame:
    """Replace ``+inf``/``-inf`` in every numeric column."""
    numeric_cols = frame.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return frame
    frame = frame.copy()
    frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], fill)
    return frame


def stats_block(values: Sequence[float]) -> dict[str, float]:
    """Return ``max/min/mean/std/var`` for a sequence, with 0.0 for empty input.

    ``std`` and ``var`` use the **sample** convention (``ddof=1``) and collapse
    to ``0.0`` for a single observation.  This mirrors Apache Commons Math's
    ``SummaryStatistics``, which is what CICFlowMeter used to generate
    CIC-IDS2017 - using the population convention here would introduce a small
    but systematic offset between the training data and live traffic.
    """
    if values is None or len(values) == 0:
        return {"max": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0, "var": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        v = float(arr[0])
        return {"max": v, "min": v, "mean": v, "std": 0.0, "var": 0.0}
    variance = float(np.var(arr, ddof=1))
    return {
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(math.sqrt(max(0.0, variance))),
        "var": variance,
    }


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Serialise ``payload`` to JSON, coping with numpy scalars and arrays."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default)
    return p


def read_json(path: str | Path) -> Any:
    """Load JSON, returning ``None`` when the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def human_bytes(num: float) -> str:
    """Format a byte count for display."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < step:
            return f"{num:,.1f} {unit}"
        num /= step
    return f"{num:,.1f} PB"


def format_duration(seconds: float) -> str:
    """Format a number of seconds as ``1h 02m 03s``."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@contextmanager
def timed(label: str) -> Iterator[dict[str, float]]:
    """Context manager that logs and records the wall-clock duration of a block."""
    holder: dict[str, float] = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - start
        logger.info("%s finished in %.2fs", label, holder["seconds"])


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield ``seq`` in consecutive chunks of at most ``size`` items."""
    size = max(1, int(size))
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


__all__ = [
    "normalise_column",
    "normalise_columns",
    "normalise_label",
    "utc_now_iso",
    "to_local_datetime",
    "safe_div",
    "clamp",
    "replace_non_finite",
    "stats_block",
    "ensure_dir",
    "write_json",
    "read_json",
    "human_bytes",
    "format_duration",
    "timed",
    "chunked",
]
