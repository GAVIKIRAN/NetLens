"""DEMO / SIMULATION mode (Phase 12).

The dashboard must be demonstrable - showing both *normal traffic* and
*attack detected* - **without generating a single malicious packet**.  Two safe
sources are supported, in order of preference:

1. **Real CIC-IDS2017 samples.**  If the dataset is present,
   ``scripts/make_demo_data.py`` draws a small class-stratified sample of real
   flow records and stores it as ``data/demo/demo_flows.csv``.  These are
   genuine attack *measurements* that were captured years ago in a lab; nothing
   is transmitted when they are replayed.
2. **Synthetic flow records.**  When the dataset has not been downloaded yet,
   :mod:`src.simulation.synthetic` fabricates coherent flow records in memory.

Either way the replay engine feeds records into exactly the same
predict -> risk -> explain -> persist path that live capture uses, so a demo
exercises the real system rather than a mock-up.

Because the source rows carry a ground-truth label, DEMO mode also stores that
label alongside the model's prediction.  The dashboard shows both: the demo is
a demonstration of the system, not a claim that the model is always right.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from src.features.contract import LIVE_FEATURES, canonical_name
from src.simulation.synthetic import SyntheticFlowGenerator, synthetic_to_rows
from src.utils.errors import DatasetError
from src.utils.helpers import ensure_dir, normalise_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

IDENTITY_COLUMNS = (
    "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol_name", "packets", "bytes", "duration_seconds",
)
#: Columns that must keep their own name during canonicalisation. Without this
#: guard ``src_ip`` would be rewritten to the contract's ``source_ip`` alias
#: (and ``src_port`` to ``source_port``), silently emptying the connection
#: metadata shown in the dashboard.
PRESERVED_COLUMNS = frozenset(IDENTITY_COLUMNS) | {"label", "source", "ground_truth"}


def canonicalise_columns(columns) -> list[str]:
    """Canonicalise feature headers while preserving identity column names."""
    result: list[str] = []
    for name in normalise_columns(columns):
        result.append(name if name in PRESERVED_COLUMNS else canonical_name(name))
    return result


@dataclass
class DemoRecord:
    """One replayable flow."""

    features: dict[str, float]
    identity: dict[str, Any]
    ground_truth: str = ""
    source: str = "demo"


# ===========================================================================
#  Building the demo file
# ===========================================================================
def build_demo_dataset(
    settings,
    *,
    rows: int | None = None,
    prefer_real: bool = True,
    attack_ratio: float | None = None,
    seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Create ``data/demo/demo_flows.csv``. Returns the path and a report."""
    demo_cfg = settings.section("demo")
    target_rows = int(rows or demo_cfg.get("synthetic_rows", 4000))
    ratio = float(attack_ratio if attack_ratio is not None else demo_cfg.get("attack_ratio", 0.3))
    destination = Path(demo_cfg.get("source_file"))
    ensure_dir(destination.parent)

    report: dict[str, Any] = {"target_rows": target_rows, "attack_ratio": ratio}

    frame: pd.DataFrame | None = None
    if prefer_real:
        try:
            frame = _sample_real_dataset(settings, target_rows, seed)
            report["source"] = "cic-ids2017"
            logger.info("Demo data drawn from the real CIC-IDS2017 dataset.")
        except DatasetError as exc:
            logger.info("Real dataset unavailable for demo data (%s); using synthetic.", exc)
        except Exception as exc:  # pragma: no cover - unexpected loader issue
            logger.warning("Falling back to synthetic demo data: %s", exc)

    if frame is None:
        generator = SyntheticFlowGenerator(seed=seed)
        flows = generator.generate(target_rows, attack_ratio=ratio)
        frame = pd.DataFrame(synthetic_to_rows(flows))
        report["source"] = "synthetic"

    frame = _finalise_demo_frame(frame, seed)
    frame.to_csv(destination, index=False)

    report["path"] = str(destination)
    report["rows"] = int(len(frame))
    report["class_counts"] = {
        str(k): int(v) for k, v in frame["label"].value_counts().items()
    }
    logger.info(
        "Demo dataset written: %s (%s rows, source=%s)",
        destination, f"{len(frame):,}", report["source"],
    )
    return destination, report


def _sample_real_dataset(settings, rows: int, seed: int) -> pd.DataFrame:
    """Draw a class-stratified sample of genuine CIC-IDS2017 flow records."""
    from src.data.cleaner import clean_dataset
    from src.data.loader import load_dataset

    dataset_cfg = settings.section("dataset")
    per_class = max(50, rows // 8)
    result = load_dataset(
        settings.path("raw_data_dir"),
        patterns=tuple(dataset_cfg.get("file_patterns", ["*.csv"])),
        encoding=str(dataset_cfg.get("encoding", "latin-1")),
        label_column=str(dataset_cfg.get("label_column", "Label")),
        max_rows_per_class=per_class,
        max_total_rows=rows,
        random_state=seed,
    )
    frame, _ = clean_dataset(
        result.frame,
        result.label_column,
        drop_duplicates=True,
        min_samples_per_class=1,
        benign_label=str(settings.get("inference.benign_label", "BENIGN")),
    )
    frame = frame.rename(columns={result.label_column: "label"})
    frame["source"] = "cic-ids2017"
    return frame


def _finalise_demo_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Ensure every contract feature and identity column exists."""
    frame = frame.copy()
    frame.columns = canonicalise_columns(frame.columns)
    if "label" not in frame.columns:
        frame["label"] = "UNKNOWN"

    for feature in LIVE_FEATURES:
        if feature not in frame.columns:
            frame[feature] = 0.0
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    rng = np.random.default_rng(seed)
    size = len(frame)
    if "src_ip" not in frame.columns:
        # CIC-IDS2017's own addresses are test-bed identifiers that this project
        # deliberately drops, so demo rows get clearly documented placeholders
        # drawn from RFC 5737 / RFC 1918 ranges.
        frame["src_ip"] = [f"192.168.10.{n}" for n in rng.integers(2, 250, size)]
    if "dst_ip" not in frame.columns:
        frame["dst_ip"] = [f"203.0.113.{n}" for n in rng.integers(2, 250, size)]
    if "src_port" not in frame.columns:
        frame["src_port"] = rng.integers(1024, 65535, size)
    if "dst_port" not in frame.columns:
        frame["dst_port"] = frame.get("destination_port", pd.Series([80] * size)).fillna(80)
    if "protocol_name" not in frame.columns:
        mapping = {6: "TCP", 17: "UDP", 1: "ICMP"}
        frame["protocol_name"] = (
            frame["protocol"].fillna(6).astype(int).map(mapping).fillna("OTHER")
            if "protocol" in frame.columns
            else "TCP"
        )
    if "packets" not in frame.columns:
        frame["packets"] = (
            frame["total_fwd_packets"].fillna(0) + frame["total_backward_packets"].fillna(0)
        ).astype(int)
    if "bytes" not in frame.columns:
        frame["bytes"] = (
            frame["total_length_of_fwd_packets"].fillna(0)
            + frame["total_length_of_bwd_packets"].fillna(0)
        ).astype(int)
    if "duration_seconds" not in frame.columns:
        frame["duration_seconds"] = frame["flow_duration"].fillna(0) / 1_000_000.0
    if "source" not in frame.columns:
        frame["source"] = "demo"

    ordered = list(LIVE_FEATURES) + [c for c in IDENTITY_COLUMNS if c in frame.columns]
    ordered += ["label", "source"]
    ordered = [c for i, c in enumerate(ordered) if c not in ordered[:i]]
    return frame[ordered].sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ===========================================================================
#  Replaying the demo file
# ===========================================================================
class DemoReplaySource:
    """Iterates over demo records at a configurable rate."""

    def __init__(
        self,
        path: str | Path,
        *,
        rate: float = 4.0,
        loop: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.path = Path(path)
        self.rate = max(0.1, float(rate))
        self.loop = bool(loop)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._records: list[DemoRecord] = []
        self._stop = threading.Event()
        self.load()

    def load(self) -> None:
        """Read the demo CSV into memory."""
        if not self.path.exists():
            raise DatasetError(
                f"Demo data file not found: {self.path}",
                hint="Run `python scripts/make_demo_data.py` to create it.",
            )
        try:
            frame = pd.read_csv(self.path)
        except Exception as exc:
            raise DatasetError(f"Could not read the demo file: {exc}") from exc
        frame.columns = canonicalise_columns(frame.columns)
        if self.shuffle:
            frame = frame.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        self._records = []
        for row in frame.to_dict(orient="records"):
            features = {
                name: _finite(row.get(name, 0.0)) for name in LIVE_FEATURES
            }
            duration = float(row.get("duration_seconds", 0.0) or 0.0)
            identity = {
                "src_ip": str(row.get("src_ip", "")),
                "dst_ip": str(row.get("dst_ip", "")),
                "src_port": int(float(row.get("src_port", 0) or 0)),
                "dst_port": int(float(row.get("dst_port", 0) or 0)),
                "protocol": int(float(row.get("protocol", 6) or 6)),
                "protocol_name": str(row.get("protocol_name", "TCP")),
                "packets": int(float(row.get("packets", 0) or 0)),
                "bytes": int(float(row.get("bytes", 0) or 0)),
                "duration_seconds": duration,
                "start_time": 0.0,
                "end_time": duration,
                "termination": "replay",
            }
            self._records.append(
                DemoRecord(
                    features=features,
                    identity=identity,
                    ground_truth=str(row.get("label", "")),
                    source=str(row.get("source", "demo")),
                )
            )
        logger.info("Loaded %s demo record(s) from %s", f"{len(self._records):,}", self.path)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.ground_truth] = counts.get(record.ground_truth, 0) + 1
        return counts

    def stop(self) -> None:
        self._stop.set()

    def stream(self) -> Iterator[DemoRecord]:
        """Yield records, pacing them at ``rate`` flows per second."""
        if not self._records:
            return
        delay = 1.0 / self.rate
        while not self._stop.is_set():
            for record in self._records:
                if self._stop.is_set():
                    return
                yield record
                time.sleep(delay)
            if not self.loop:
                return

    def take(self, count: int) -> list[DemoRecord]:
        """Return ``count`` records immediately (no pacing) - used by tests."""
        if not self._records:
            return []
        out: list[DemoRecord] = []
        while len(out) < count:
            out.extend(self._records[: count - len(out)])
            if not self.loop and len(out) >= len(self._records):
                break
        return out[:count]


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isinf(number):
        return float("nan")
    return number


def load_demo_records(path: str | Path, limit: int | None = None) -> list[DemoRecord]:
    """Convenience loader used by the tests and the batch scripts."""
    source = DemoReplaySource(path, loop=False, shuffle=False)
    records = source._records
    return records[: int(limit)] if limit else records


__all__ = [
    "canonicalise_columns",
    "DemoRecord",
    "DemoReplaySource",
    "build_demo_dataset",
    "load_demo_records",
    "IDENTITY_COLUMNS",
]
