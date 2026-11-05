"""Builds a CIC-IDS2017-*shaped* CSV fixture for tests and smoke runs.

The real CIC-IDS2017 archive is ~500 MB and cannot be shipped with this
project, so the test-suite needs a stand-in with the same **shape**: the same
raw header spellings (leading spaces, ``Flow Bytes/s``, ``Init_Win_bytes_forward``,
the duplicated ``Fwd Header Length.1``), the all-zero ``*Bulk*`` columns, the
identifier columns, ``Infinity`` strings in the rate columns, duplicate rows and
the Windows-1252 dash inside the web-attack labels.

The *values* come from :mod:`src.simulation.synthetic`, which simulates packets
in memory and runs them through the project's own flow meter, so the records
are internally consistent.

**Any metric measured on this fixture is a measure of the pipeline, not of
CIC-IDS2017.**  Real numbers require the real dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.features.contract import BY_NAME, LIVE_FEATURES
from src.simulation.synthetic import SyntheticFlowGenerator

# Raw CIC-IDS2017 identifier columns that the cleaner must drop.
IDENTIFIER_COLUMNS = (
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "External IP",
)
# Raw CIC-IDS2017 columns that are constant/degenerate and must be dropped.
DEGENERATE_COLUMNS = (
    "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    " CWE Flag Count", " Fwd Header Length.1",
)

DEFAULT_CLASSES = (
    "BENIGN", "DDoS", "DoS Hulk", "PortScan", "SSH-Patator",
    "Bot", "Web Attack \x96 Brute Force",
)


def _raw_header(canonical: str) -> str:
    """Canonical name -> the raw CIC-IDS2017 spelling (with a leading space)."""
    spec = BY_NAME.get(canonical)
    if spec and spec.cic_aliases:
        return " " + spec.cic_aliases[0]
    return " " + canonical.replace("_", " ").title()


def build_frame(
    rows: int = 900,
    *,
    seed: int = 42,
    classes: Sequence[str] = DEFAULT_CLASSES,
    attack_ratio: float = 0.55,
) -> pd.DataFrame:
    """Generate a CIC-IDS2017-shaped ``DataFrame``."""
    # A non-zero jitter makes the synthetic classes overlap, so the fixture
    # exercises the metric code with imperfect (realistic) separability.
    generator = SyntheticFlowGenerator(seed=seed, jitter=0.85)
    profile_labels = [c.replace("\x96", "-").replace(" - ", " - ") for c in classes]
    flows = generator.generate(
        rows,
        attack_ratio=attack_ratio,
        labels=[c for c in profile_labels if c != "BENIGN"],
    )

    records: list[dict[str, object]] = []
    rnd = random.Random(seed)
    for index, flow in enumerate(flows):
        row: dict[str, object] = {}
        for canonical in LIVE_FEATURES:
            row[_raw_header(canonical)] = flow.features[canonical]
        # Identifier columns (must be dropped by the cleaner).
        row["Flow ID"] = (
            f"{flow.identity['src_ip']}-{flow.identity['dst_ip']}-"
            f"{flow.identity['src_port']}-{flow.identity['dst_port']}-{flow.identity['protocol']}"
        )
        row["Source IP"] = flow.identity["src_ip"]
        row["Source Port"] = flow.identity["src_port"]
        row["Destination IP"] = flow.identity["dst_ip"]
        row["Timestamp"] = f"6/7/2017 {8 + index % 10}:{index % 60:02d}"
        row["External IP"] = ""
        # Degenerate columns (must also be dropped).
        for column in DEGENERATE_COLUMNS:
            if "Header Length.1" in column:
                row[column] = flow.features["fwd_header_length"]
            else:
                row[column] = 0
        label = flow.label
        if label.startswith("Web Attack"):
            label = "Web Attack \x96 Brute Force"
        row[" Label"] = label
        records.append(row)

    frame = pd.DataFrame(records)

    # Inject the artefacts a real CIC-IDS2017 file contains.
    rate_columns = [" Flow Bytes/s", " Flow Packets/s"]
    for column in rate_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(object)
            picks = rnd.sample(range(len(frame)), k=max(1, len(frame) // 60))
            for i in picks:
                frame.loc[i, column] = "Infinity"
    # A few missing values.
    if " Flow IAT Std" in frame.columns:
        for i in rnd.sample(range(len(frame)), k=max(1, len(frame) // 80)):
            frame.loc[i, " Flow IAT Std"] = np.nan
    # Duplicate rows.
    duplicates = frame.sample(n=max(2, len(frame) // 40), random_state=seed)
    frame = pd.concat([frame, duplicates], ignore_index=True)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_dataset(
    directory: str | Path,
    *,
    files: int = 2,
    rows_per_file: int = 900,
    seed: int = 42,
) -> list[Path]:
    """Write ``files`` CIC-IDS2017-shaped CSVs into ``directory``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    names = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ]
    written: list[Path] = []
    for index in range(int(files)):
        frame = build_frame(rows_per_file, seed=seed + index)
        path = target / names[index % len(names)]
        frame.to_csv(path, index=False, encoding="latin-1")
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover - manual helper
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="Write a CIC-IDS2017-shaped fixture.")
    parser.add_argument("directory")
    parser.add_argument("--files", type=int, default=2)
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in write_dataset(args.directory, files=args.files, rows_per_file=args.rows, seed=args.seed):
        print(path)
