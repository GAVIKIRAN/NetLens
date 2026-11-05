"""Train the intrusion-detection models on CIC-IDS2017.

Windows usage
-------------
    python scripts/train_model.py
    python scripts/train_model.py --data-dir D:\\datasets\\CICIDS2017
    python scripts/train_model.py --quick          (fast smoke run)
    python scripts/train_model.py --max-rows 100000
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the XAI-NIDS models.")
    parser.add_argument("--config", default=None, help="path to an alternative config.yaml")
    parser.add_argument("--data-dir", default=None, help="override paths.raw_data_dir")
    parser.add_argument("--max-rows", type=int, default=None, help="override dataset.max_total_rows")
    parser.add_argument("--quick", action="store_true", help="small, fast smoke run")
    args = parser.parse_args()

    settings = bootstrap(args.config)

    from src.training.train import run_training
    from src.utils.errors import NIDSError

    try:
        artifacts = run_training(
            settings,
            data_dir=args.data_dir,
            max_total_rows=args.max_rows,
            quick=args.quick,
        )
    except NIDSError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        print("See logs/nids.log for the full traceback.", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"\n[ERROR] Unexpected failure: {exc}\n", file=sys.stderr)
        print("See logs/nids.log for the full traceback.", file=sys.stderr)
        raise

    best = artifacts.metrics.get("best_model", {})
    test = (best.get("test") or {})
    print("\n" + "=" * 72)
    print(f"  Selected model : {artifacts.best_model_name}")
    print(f"  Rationale      : {artifacts.metadata['selection']['rationale']}")
    print("  --- TEST split (single, final evaluation) ---")
    print(f"  Accuracy            : {test.get('accuracy', 0):.4f}")
    print(f"  Balanced accuracy   : {test.get('balanced_accuracy', 0):.4f}")
    print(f"  Macro F1            : {test.get('f1_macro', 0):.4f}")
    print(f"  Weighted F1         : {test.get('f1_weighted', 0):.4f}")
    binary = test.get("binary", {}) or {}
    print(f"  Detection rate      : {binary.get('detection_rate', 0):.4f}")
    print(f"  False-positive rate : {binary.get('false_positive_rate', 0):.4f}")
    print("=" * 72)
    print(f"  Artefacts : {artifacts.model_dir}")
    print(f"  Reports   : {settings.path('reports_dir')}")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
