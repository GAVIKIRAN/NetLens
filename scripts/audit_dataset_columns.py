"""Measure the facts behind the feature contract on the real CIC-IDS2017 CSVs.

Every exclusion and flow-semantics decision in ``src/features/contract.py``
should be backed by a measurement rather than by what the literature says.  This
script reads every CSV in chunks (memory stays bounded), computes per-column
statistics and the specific equality checks the contract depends on, and writes
them to ``reports/feature_contract_audit.json``.

    python scripts/audit_dataset_columns.py
    python scripts/audit_dataset_columns.py --data-dir D:\\datasets\\CICIDS2017
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap

CHUNK_ROWS = 200_000

#: Column pairs expected to relate, named with the normalised header spelling.
EQUALITY_CHECKS = {
    "subflow_fwd_packets == total_fwd_packets": ("subflow_fwd_packets", "total_fwd_packets"),
    "subflow_bwd_packets == total_backward_packets": (
        "subflow_bwd_packets", "total_backward_packets",
    ),
    "subflow_fwd_bytes == total_length_of_fwd_packets": (
        "subflow_fwd_bytes", "total_length_of_fwd_packets",
    ),
    "subflow_bwd_bytes == total_length_of_bwd_packets": (
        "subflow_bwd_bytes", "total_length_of_bwd_packets",
    ),
    "fwd_header_length == fwd_header_length_1": ("fwd_header_length", "fwd_header_length_1"),
    "avg_fwd_segment_size == fwd_packet_length_mean": (
        "avg_fwd_segment_size", "fwd_packet_length_mean",
    ),
    "average_packet_size == packet_length_mean": ("average_packet_size", "packet_length_mean"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CIC-IDS2017 columns.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    settings = bootstrap(args.config)

    import numpy as np
    import pandas as pd

    from src.features.contract import canonical_name

    data_dir = Path(args.data_dir) if args.data_dir else settings.path("raw_data_dir")
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() == ".csv")
    if not files:
        print(f"[ERROR] No CSV files found under {data_dir}", file=sys.stderr)
        return 2

    encoding = str(settings.get("dataset.encoding", "latin-1"))
    stats: dict[str, dict[str, float]] = {}
    distinct: dict[str, set] = {}
    equality = {name: {"compared": 0, "equal": 0} for name in EQUALITY_CHECKS}
    subflow_gap = {"rows": 0, "gap_flows": 0, "gap_flows_equal_totals": 0}
    active_short = {"under_1s_multi_packet": 0, "with_nonzero_active": 0}
    labels: dict[str, int] = {}
    raw_headers: dict[str, str] = {}
    total_rows = 0

    for path in files:
        print(f"  reading {path.name} ...")
        for chunk in pd.read_csv(
            path, encoding=encoding, chunksize=CHUNK_ROWS, low_memory=False, skipinitialspace=True
        ):
            renamed = {column: canonical_name(column) for column in chunk.columns}
            for raw, name in renamed.items():
                raw_headers.setdefault(name, str(raw).strip())
            chunk = chunk.rename(columns=renamed)
            total_rows += len(chunk)

            if "label" in chunk.columns:
                for label, count in chunk["label"].astype(str).value_counts().items():
                    labels[label] = labels.get(label, 0) + int(count)

            numeric = chunk.drop(columns=["label"], errors="ignore").apply(
                pd.to_numeric, errors="coerce"
            )
            for column in numeric.columns:
                values = numeric[column].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                entry = stats.setdefault(
                    column,
                    {"rows": 0, "nan": 0, "inf": 0, "zero": 0, "negative": 0, "minus_one": 0,
                     "min": float("inf"), "max": float("-inf")},
                )
                entry["rows"] += len(values)
                entry["nan"] += int(np.isnan(values).sum())
                entry["inf"] += int(np.isinf(values).sum())
                if finite.size:
                    entry["zero"] += int((finite == 0).sum())
                    entry["negative"] += int((finite < 0).sum())
                    entry["minus_one"] += int((finite == -1).sum())
                    entry["min"] = min(entry["min"], float(finite.min()))
                    entry["max"] = max(entry["max"], float(finite.max()))
                    seen = distinct.setdefault(column, set())
                    if len(seen) <= 20:
                        seen.update(np.unique(finite)[:25].tolist())

            for name, (left, right) in EQUALITY_CHECKS.items():
                if left in numeric.columns and right in numeric.columns:
                    a, b = numeric[left].to_numpy(), numeric[right].to_numpy()
                    mask = np.isfinite(a) & np.isfinite(b)
                    equality[name]["compared"] += int(mask.sum())
                    equality[name]["equal"] += int((np.isclose(a[mask], b[mask])).sum())

            # A flow whose longest gap exceeds 1 s has at least one subflow break.
            required = {"flow_iat_max", "subflow_fwd_packets", "total_fwd_packets"}
            if required.issubset(numeric.columns):
                gap = numeric["flow_iat_max"].to_numpy() > 1_000_000
                same = np.isclose(
                    numeric["subflow_fwd_packets"].to_numpy(),
                    numeric["total_fwd_packets"].to_numpy(),
                )
                subflow_gap["rows"] += len(numeric)
                subflow_gap["gap_flows"] += int(gap.sum())
                subflow_gap["gap_flows_equal_totals"] += int((gap & same).sum())

            # Our old rule gave every multi-packet flow a non-zero active time;
            # CICFlowMeter's source only updates after a gap of more than 1 s.
            required = {"flow_duration", "active_mean", "total_fwd_packets",
                        "total_backward_packets"}
            if required.issubset(numeric.columns):
                packets = (numeric["total_fwd_packets"] + numeric["total_backward_packets"])
                short = (numeric["flow_duration"].between(1, 1_000_000)) & (packets >= 2)
                active_short["under_1s_multi_packet"] += int(short.sum())
                active_short["with_nonzero_active"] += int(
                    (short & (numeric["active_mean"] != 0)).sum()
                )

    columns = {}
    for name, entry in stats.items():
        values = sorted(distinct.get(name, set()))
        columns[name] = {
            "raw_header": raw_headers.get(name, name),
            **entry,
            "distinct_values_if_few": values if len(values) <= 20 else None,
            "constant": entry["min"] == entry["max"],
        }

    result = {
        "files": [p.name for p in files],
        "total_rows": total_rows,
        "column_count": len(columns),
        "labels": labels,
        "equality_checks": equality,
        "subflow_gap_check": subflow_gap,
        "active_time_short_flows": active_short,
        "columns": columns,
    }
    reports_dir = settings.path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "feature_contract_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n  rows: {total_rows:,}   columns: {len(columns)}   files: {len(files)}")
    print("\n  Constant columns:")
    for name, info in columns.items():
        if info["constant"]:
            print(f"    {name:<34s} value={info['min']}")
    print("\n  Equality checks:")
    for name, counts in equality.items():
        share = counts["equal"] / counts["compared"] if counts["compared"] else 0.0
        print(f"    {name:<52s} {counts['equal']:>9,}/{counts['compared']:<9,} ({share:.4%})")
    print("\n  Subflow gap check:", subflow_gap)
    print("  Active time on flows under 1 s:", active_short)
    print("\n  Flag and sentinel columns:")
    for name in ("fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
                 "init_win_bytes_forward", "init_win_bytes_backward", "act_data_pkt_fwd",
                 "min_seg_size_forward", "cwe_flag_count", "flow_duration",
                 "total_fwd_packets", "total_backward_packets"):
        info = columns.get(name)
        if info:
            print(f"    {name:<26s} min={info['min']:<14} max={info['max']:<16} "
                  f"zero={int(info['zero']):<9,} minus_one={int(info['minus_one']):<9,} "
                  f"distinct={info['distinct_values_if_few']}")
    print(f"\n  Written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
