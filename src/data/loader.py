"""CIC-IDS2017 dataset discovery and memory-safe loading.

The published dataset ships as several CSV files whose names differ between
mirrors (``MachineLearningCVE/``, ``GeneratedLabelledFlows/TrafficLabelling/``,
Kaggle re-uploads, ...).  Nothing here assumes a filename: the loader
recursively discovers every CSV under the configured directory and identifies
the label column by content, not by position.

Loading strategy
----------------
CIC-IDS2017 has ~2.83 M flow records.  Materialising all of them as float64
needs well over 1.5 GB, which is not realistic on a student laptop, so the
loader performs a **two-pass stratified subsample**:

1. *Pass 1* reads only the label column of every file and counts the classes.
   This is cheap (one column) and tells us how the classes are distributed
   across files.
2. *Pass 2* re-reads the files in chunks and keeps a seeded random subset of
   each class, with the per-class quota spread across files **in proportion to
   where that class actually occurs**.  Sampling proportionally matters: naive
   "take the first N rows" would draw the entire BENIGN quota from Monday's
   capture and silently bias the model.

The result is a balanced-but-honest subsample whose exact composition is
reported by :meth:`LoadResult.summary`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.features.contract import canonical_name
from src.utils.errors import DatasetError
from src.utils.helpers import normalise_columns, normalise_label
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 200_000
_LABEL_CANDIDATES = ("label", "class", "attack", "attack_type", "category")


@dataclass
class LoadResult:
    """Everything the training pipeline needs to know about the loaded data."""

    frame: pd.DataFrame
    label_column: str
    files: list[Path] = field(default_factory=list)
    total_rows_available: int = 0
    rows_loaded: int = 0
    class_counts_full: dict[str, int] = field(default_factory=dict)
    class_counts_loaded: dict[str, int] = field(default_factory=dict)
    skipped_files: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "files": [str(p) for p in self.files],
            "skipped_files": self.skipped_files,
            "label_column": self.label_column,
            "total_rows_available": self.total_rows_available,
            "rows_loaded": self.rows_loaded,
            "n_features": max(0, self.frame.shape[1] - 1),
            "class_counts_full": self.class_counts_full,
            "class_counts_loaded": self.class_counts_loaded,
        }


def discover_csv_files(
    directory: str | Path, patterns: Sequence[str] = ("*.csv", "*.CSV")
) -> list[Path]:
    """Recursively find candidate CSV files under ``directory``."""
    root = Path(directory)
    if not root.exists():
        raise DatasetError(
            f"Dataset directory does not exist: {root}",
            hint="Create it and copy the CIC-IDS2017 CSV files in, "
                 "or point paths.raw_data_dir at your download.",
        )
    if not root.is_dir():
        raise DatasetError(f"paths.raw_data_dir is not a directory: {root}")

    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            if path.stat().st_size == 0:
                continue
            seen.add(resolved)
            found.append(path)
    if not found:
        raise DatasetError(
            f"No CSV files found under {root}",
            hint="Expected the CIC-IDS2017 flow CSVs, e.g. "
                 "'Monday-WorkingHours.pcap_ISCX.csv'.",
        )
    logger.info("Discovered %d CSV file(s) under %s", len(found), root)
    return found


def read_header(path: Path, encoding: str) -> list[str]:
    """Read and canonicalise a file's header row."""
    try:
        head = pd.read_csv(path, nrows=0, encoding=encoding, engine="python")
    except UnicodeDecodeError:
        head = pd.read_csv(path, nrows=0, encoding="latin-1", engine="python")
    except Exception as exc:  # pragma: no cover - malformed file
        raise DatasetError(f"Cannot read header of {path.name}: {exc}") from exc
    normalised = normalise_columns(head.columns)
    return [canonical_name(col) for col in normalised]


def detect_label_column(columns: Sequence[str], configured: str = "Label") -> str:
    """Find the ground-truth column among canonicalised ``columns``."""
    target = canonical_name(configured)
    if target in columns:
        return target
    for candidate in _LABEL_CANDIDATES:
        if candidate in columns:
            return candidate
    for column in reversed(list(columns)):
        if any(candidate in column for candidate in _LABEL_CANDIDATES):
            return column
    raise DatasetError(
        "No label column found in the CSV header.",
        hint=f"Expected a column named '{configured}'. Found: {list(columns)[:12]}",
    )


def _rename_map(path: Path, encoding: str) -> tuple[dict[str, str], list[str]]:
    """Build a raw-header -> canonical-name mapping for one file."""
    try:
        head = pd.read_csv(path, nrows=0, encoding=encoding, engine="python")
    except UnicodeDecodeError:
        head = pd.read_csv(path, nrows=0, encoding="latin-1", engine="python")
    raw = list(head.columns)
    normalised = normalise_columns(raw)
    canonical = [canonical_name(col) for col in normalised]
    return dict(zip(raw, canonical)), canonical


def _count_labels(path: Path, encoding: str, label_col: str) -> dict[str, int]:
    """Pass 1 - read only the label column and count classes."""
    rename, canonical = _rename_map(path, encoding)
    reverse = {v: k for k, v in rename.items()}
    raw_label = reverse.get(label_col)
    if raw_label is None:
        return {}
    counts: dict[str, int] = {}
    reader = pd.read_csv(
        path,
        usecols=[raw_label],
        encoding=encoding,
        chunksize=CHUNK_SIZE,
        engine="c",
        on_bad_lines="skip",
    )
    for chunk in reader:
        series = chunk[raw_label].map(normalise_label)
        for label, count in series.value_counts().items():
            counts[str(label)] = counts.get(str(label), 0) + int(count)
    counts.pop("Label", None)  # repeated header rows inside concatenated CSVs
    counts.pop("nan", None)
    return counts


def _coerce_numeric(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Force every non-label column to float32, mapping junk to NaN."""
    for column in frame.columns:
        if column == label_col:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].astype(np.float32, copy=False)
    return frame


def load_dataset(
    directory: str | Path,
    *,
    patterns: Sequence[str] = ("*.csv", "*.CSV"),
    encoding: str = "latin-1",
    label_column: str = "Label",
    max_rows_per_class: int | None = None,
    max_total_rows: int | None = None,
    max_rows_per_file: int | None = None,
    random_state: int = 42,
) -> LoadResult:
    """Discover, validate, subsample and concatenate the dataset."""
    files = discover_csv_files(directory, patterns)
    rng = np.random.default_rng(random_state)

    # --------------------------------------------------------------- pass 1
    usable: list[Path] = []
    skipped: dict[str, str] = {}
    label_col = ""
    per_file_counts: dict[Path, dict[str, int]] = {}

    for path in files:
        try:
            columns = read_header(path, encoding)
            detected = detect_label_column(columns, label_column)
        except DatasetError as exc:
            skipped[path.name] = str(exc)
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        if not label_col:
            label_col = detected
        elif detected != label_col:
            skipped[path.name] = (
                f"label column '{detected}' differs from '{label_col}'"
            )
            logger.warning("Skipping %s: inconsistent label column", path.name)
            continue
        try:
            counts = _count_labels(path, encoding, label_col)
        except Exception as exc:  # pragma: no cover - malformed file
            skipped[path.name] = f"unreadable: {exc}"
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        if not counts:
            skipped[path.name] = "no usable rows"
            continue
        per_file_counts[path] = counts
        usable.append(path)

    if not usable:
        raise DatasetError(
            "None of the discovered CSV files could be read as CIC-IDS2017 data.",
            hint="Check the encoding setting and that the files are flow CSVs.",
        )

    class_totals: dict[str, int] = {}
    for counts in per_file_counts.values():
        for label, count in counts.items():
            class_totals[label] = class_totals.get(label, 0) + count
    total_available = int(sum(class_totals.values()))
    logger.info(
        "Pass 1 complete: %s rows across %d file(s), %d class(es)",
        f"{total_available:,}", len(usable), len(class_totals),
    )

    # ------------------------------------------------- per-class allocation
    quota: dict[str, int] = {}
    for label, total in class_totals.items():
        target = total if max_rows_per_class is None else min(total, int(max_rows_per_class))
        quota[label] = int(target)

    # Spread each class quota across files in proportion to occurrence.
    alloc: dict[Path, dict[str, int]] = {p: {} for p in usable}
    for label, target in quota.items():
        total = class_totals[label]
        if total <= 0:
            continue
        remaining = target
        holders = [p for p in usable if per_file_counts[p].get(label, 0) > 0]
        for index, path in enumerate(holders):
            available = per_file_counts[path][label]
            if index == len(holders) - 1:
                take = min(remaining, available)
            else:
                take = min(available, int(math.floor(target * available / total)))
            take = max(0, min(take, remaining))
            alloc[path][label] = take
            remaining -= take
        if remaining > 0:  # rounding leftovers
            for path in holders:
                spare = per_file_counts[path][label] - alloc[path].get(label, 0)
                if spare <= 0:
                    continue
                extra = min(spare, remaining)
                alloc[path][label] = alloc[path].get(label, 0) + extra
                remaining -= extra
                if remaining <= 0:
                    break

    # --------------------------------------------------------------- pass 2
    frames: list[pd.DataFrame] = []
    taken: dict[Path, dict[str, int]] = {p: {} for p in usable}
    rows_from_file: dict[Path, int] = {p: 0 for p in usable}

    for path in usable:
        rename, _ = _rename_map(path, encoding)
        wanted = alloc[path]
        if not any(wanted.values()):
            continue
        try:
            reader = pd.read_csv(
                path,
                encoding=encoding,
                chunksize=CHUNK_SIZE,
                engine="c",
                low_memory=False,
                on_bad_lines="skip",
            )
        except Exception as exc:  # pragma: no cover - malformed file
            skipped[path.name] = f"unreadable in pass 2: {exc}"
            continue

        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            chunk.columns = [canonical_name(c) for c in chunk.columns]
            if label_col not in chunk.columns:
                continue
            chunk[label_col] = chunk[label_col].map(normalise_label)
            chunk = chunk[~chunk[label_col].isin(["Label", "nan", ""])]
            if chunk.empty:
                continue

            keep_parts: list[pd.DataFrame] = []
            for label, group in chunk.groupby(label_col, sort=False):
                want = wanted.get(str(label), 0)
                already = taken[path].get(str(label), 0)
                remaining = want - already
                if remaining <= 0:
                    continue
                if len(group) > remaining:
                    idx = rng.choice(len(group), size=remaining, replace=False)
                    group = group.iloc[np.sort(idx)]
                taken[path][str(label)] = already + len(group)
                keep_parts.append(group)

            if not keep_parts:
                if all(
                    taken[path].get(k, 0) >= v for k, v in wanted.items() if v > 0
                ):
                    break
                continue

            kept = pd.concat(keep_parts, ignore_index=True)
            kept = _coerce_numeric(kept, label_col)
            frames.append(kept)
            rows_from_file[path] += len(kept)

            if max_rows_per_file and rows_from_file[path] >= max_rows_per_file:
                break
        logger.info("Loaded %s rows from %s", f"{rows_from_file[path]:,}", path.name)

    if not frames:
        raise DatasetError("Dataset loading produced zero rows.")

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]

    # ------------------------------------------------ global stratified cap
    if max_total_rows and len(frame) > int(max_total_rows):
        frame = _stratified_cap(frame, label_col, int(max_total_rows), random_state)

    frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    loaded_counts = {
        str(k): int(v) for k, v in frame[label_col].value_counts().items()
    }
    logger.info(
        "Dataset ready: %s rows x %d columns, %d class(es)",
        f"{len(frame):,}", frame.shape[1], len(loaded_counts),
    )
    return LoadResult(
        frame=frame,
        label_column=label_col,
        files=usable,
        total_rows_available=total_available,
        rows_loaded=int(len(frame)),
        class_counts_full={str(k): int(v) for k, v in class_totals.items()},
        class_counts_loaded=loaded_counts,
        skipped_files=skipped,
    )


def _stratified_cap(
    frame: pd.DataFrame, label_col: str, cap: int, random_state: int
) -> pd.DataFrame:
    """Reduce ``frame`` to ``cap`` rows while preserving small classes."""
    counts = frame[label_col].value_counts()
    n_classes = len(counts)
    per_class = max(1, cap // max(1, n_classes))
    parts: list[pd.DataFrame] = []
    leftover = cap

    # Give every class its equal share first (small classes take less).
    for label, count in counts.sort_values().items():
        take = min(int(count), per_class)
        group = frame[frame[label_col] == label]
        parts.append(group.sample(n=take, random_state=random_state))
        leftover -= take

    # Distribute what is left over onto the classes that still have spare rows.
    if leftover > 0:
        used = pd.concat(parts).index
        spare = frame.drop(index=used)
        if len(spare) > 0:
            take = min(leftover, len(spare))
            parts.append(spare.sample(n=take, random_state=random_state))
    capped = pd.concat(parts).sample(frac=1.0, random_state=random_state)
    return capped.reset_index(drop=True)


def load_flow_csv(path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a single already-canonical flow CSV (used by demo mode and tests)."""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(
            f"Flow file not found: {file_path}",
            hint="Run `python scripts/make_demo_data.py` to generate it.",
        )
    try:
        frame = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        frame = pd.read_csv(file_path, encoding="latin-1")
    except Exception as exc:
        raise DatasetError(f"Cannot read {file_path.name}: {exc}") from exc
    frame.columns = [canonical_name(c) for c in normalise_columns(frame.columns)]
    return frame


def iter_class_counts(frame: pd.DataFrame, label_col: str) -> Iterable[tuple[str, int]]:
    """Yield ``(label, count)`` pairs sorted by descending frequency."""
    for label, count in frame[label_col].value_counts().items():
        yield str(label), int(count)


__all__ = [
    "LoadResult",
    "discover_csv_files",
    "read_header",
    "detect_label_column",
    "load_dataset",
    "load_flow_csv",
    "iter_class_counts",
]
