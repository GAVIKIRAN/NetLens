"""Build the safe DEMO dataset (Phase 12).

Prefers a class-stratified sample of the real CIC-IDS2017 records; falls back to
coherent synthetic flow records when the dataset has not been downloaded yet.
Neither option transmits anything on the network.

    python scripts/make_demo_data.py
    python scripts/make_demo_data.py --rows 6000 --synthetic
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate data/demo/demo_flows.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rows", type=int, default=None, help="number of demo records")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="force synthetic generation even if the real dataset is present",
    )
    parser.add_argument("--attack-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = bootstrap(args.config)

    from src.simulation.replay import build_demo_dataset
    from src.utils.errors import NIDSError

    try:
        path, report = build_demo_dataset(
            settings,
            rows=args.rows,
            prefer_real=not args.synthetic,
            attack_ratio=args.attack_ratio,
            seed=args.seed,
        )
    except NIDSError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 2

    print("\n" + "=" * 72)
    print(f"  Demo dataset : {path}")
    print(f"  Source       : {report['source']}")
    print(f"  Rows         : {report['rows']:,}")
    print("  Classes      :")
    for label, count in sorted(report["class_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {label:<32s} {count:>8,}")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
