"""Print the measured evaluation results and write them to ``reports/``.

    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --report        (full classification report)

Writes evaluation_summary.csv, classification_report.txt, confusion_matrix.csv
and per_class_metrics.csv.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the stored evaluation metrics.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--report", action="store_true", help="print the classification report")
    parser.add_argument("--confusion", action="store_true", help="print the confusion matrix")
    args = parser.parse_args()

    settings = bootstrap(args.config)

    import pandas as pd

    from src.training.evaluate import write_evaluation_reports
    from src.training.train import load_metadata, load_metrics

    metrics = load_metrics(settings.path("model_dir"))
    if not metrics:
        print(
            "No evaluation metrics found. Run `python scripts/train_model.py` first.",
            file=sys.stderr,
        )
        return 2

    print("\nMetric averaging:", metrics.get("averaging_note", ""), "\n")

    rows = []
    for candidate in metrics.get("candidates", []):
        for split in ("validation", "test"):
            result = candidate.get(split)
            if not result:
                continue
            rows.append(
                {
                    "model": candidate["name"],
                    "split": split,
                    "accuracy": round(result["accuracy"], 4),
                    "balanced_acc": round(result["balanced_accuracy"], 4),
                    "macro_f1": round(result["f1_macro"], 4),
                    "weighted_f1": round(result["f1_weighted"], 4),
                    "detection": round(result["binary"]["detection_rate"], 4),
                    "fpr": round(result["binary"]["false_positive_rate"], 4),
                    "latency_ms": round(result["predict_latency_ms"], 5),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        print(frame.to_string(index=False))

    best = metrics.get("best_model", {})
    print(f"\nSelected model: {best.get('name')}  (metric: {metrics.get('primary_metric')})")

    test = best.get("test") or {}
    if args.report and test:
        print("\nClassification report (TEST split):\n")
        print(test.get("classification_report_text", ""))
    if args.confusion and test:
        print("\nConfusion matrix (TEST split):\n")
        matrix = pd.DataFrame(
            test["confusion_matrix"], index=test["class_names"], columns=test["class_names"]
        )
        print(matrix.to_string())

    metadata = load_metadata(settings.path("model_dir")) or {}
    source = str(metadata.get("dataset", {}).get("source_directory", ""))
    print(f"\nTrained on data from: {source or 'not recorded'}")
    for path in write_evaluation_reports(
        metrics, settings.path("reports_dir"), source_directory=source
    ):
        print(f"  wrote {path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
