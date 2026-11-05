"""Load the dataset and print/save its statistics without training anything.

    python scripts/profile_dataset.py
    python scripts/profile_dataset.py --data-dir D:\\datasets\\CICIDS2017
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the CIC-IDS2017 dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    settings = bootstrap(args.config)

    from src.data.cleaner import clean_dataset
    from src.data.loader import load_dataset
    from src.data.profiler import profile_dataset, save_profile
    from src.utils.errors import NIDSError

    dataset_cfg = settings.section("dataset")
    directory = args.data_dir or settings.path("raw_data_dir")

    try:
        result = load_dataset(
            directory,
            patterns=tuple(dataset_cfg.get("file_patterns", ["*.csv"])),
            encoding=str(dataset_cfg.get("encoding", "latin-1")),
            label_column=str(dataset_cfg.get("label_column", "Label")),
            max_rows_per_class=dataset_cfg.get("max_rows_per_class"),
            max_total_rows=args.max_rows or dataset_cfg.get("max_total_rows"),
            random_state=int(dataset_cfg.get("random_state", 42)),
        )
        frame, clean_report = clean_dataset(
            result.frame,
            result.label_column,
            drop_duplicates=bool(dataset_cfg.get("drop_duplicates", True)),
            min_samples_per_class=int(dataset_cfg.get("min_samples_per_class", 50)),
        )
    except NIDSError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 2

    profile = profile_dataset(
        frame,
        result.label_column,
        extra={"loading": result.summary(), "cleaning": clean_report.to_dict()},
    )
    path = save_profile(profile, settings.path("reports_dir"))

    print("\n" + "=" * 72)
    print(f"  Files discovered : {len(result.files)}")
    print(f"  Rows available   : {result.total_rows_available:,}")
    print(f"  Rows loaded      : {result.rows_loaded:,}")
    print(f"  Rows after clean : {profile.n_records:,}")
    print(f"  Features         : {profile.n_features}")
    print(f"  Duplicates       : {clean_report.duplicates_removed:,} removed")
    print(f"  Infinite values  : {clean_report.infinite_values_replaced:,} replaced")
    print(f"  Missing (left)   : {clean_report.missing_values_remaining:,}")
    print(f"  Class imbalance  : {profile.imbalance_ratio}:1 (max/min)")
    print("=" * 72)
    print("\n  Class distribution")
    for label, count in sorted(
        profile.class_distribution.items(), key=lambda kv: kv[1], reverse=True
    ):
        pct = profile.class_distribution_pct.get(label, 0.0)
        print(f"    {label:<32s} {count:>10,}  ({pct:6.2f} %)")
    print(f"\n  Full report written to {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
