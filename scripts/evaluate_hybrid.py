"""Evaluate the hybrid detector on packet-level scenarios, against the model alone.

    python scripts/evaluate_hybrid.py
    python scripts/evaluate_hybrid.py --seeds 10

The model is not retrained here, and the thresholds in config.yaml were fixed before
this script was run.  Each seed builds different traffic from the scenarios.  The
scenarios are synthetic, so these numbers check the logic.  They do not measure accuracy
on a real network: use `scripts/show_detections.py` after a controlled test for that.

Writes reports/hybrid_evaluation.json.  Nothing is transmitted or captured.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from _bootstrap import bootstrap

LABELS = ["BENIGN", "SUSPICIOUS", "ATTACK"]
ORDER = {name: index for index, name in enumerate(LABELS)}
FIRST_SEED = 1000


def _outcome(monitor) -> tuple[str, Counter]:
    """The highest decision for this traffic, across flows and behavioural findings."""
    decisions = [row["final"] for row in monitor.recent]
    decisions += [row["decision"] for row in monitor.recent_findings]
    highest = max(decisions, key=ORDER.__getitem__, default="BENIGN")
    return highest, Counter(row["detection"] for row in monitor.recent_findings)


def _binary(truth: list[str], predicted: list[str]) -> dict[str, float | int]:
    tp = sum(t != "BENIGN" and p != "BENIGN" for t, p in zip(truth, predicted))
    fp = sum(t == "BENIGN" and p != "BENIGN" for t, p in zip(truth, predicted))
    fn = sum(t != "BENIGN" and p == "BENIGN" for t, p in zip(truth, predicted))
    tn = sum(t == "BENIGN" and p == "BENIGN" for t, p in zip(truth, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "scenario_runs": len(truth),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round((tp + tn) / len(truth), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", type=int, default=10, help="traffic variations per scenario")
    args = parser.parse_args()
    settings = bootstrap(args.config)

    from src.database.database import Database
    from src.inference.predictor import Predictor
    from src.network.monitor import MonitorService
    from src.simulation.scenarios import SCENARIOS, build_packets

    predictor = Predictor.load(settings)
    runs = []
    for scenario in SCENARIOS:
        for seed in range(FIRST_SEED, FIRST_SEED + args.seeds):
            database = Database(":memory:")
            monitor = MonitorService(
                settings, database=database, predictor=predictor, buffer_size=100000
            )
            monitor._explainer = None
            monitor.alert_min_band = "CRITICAL"  # no inline SHAP: not needed for decisions
            monitor.correlator.alert_min_band = "CRITICAL"
            monitor.process_packets(build_packets(scenario, seed))
            hybrid, kinds = _outcome(monitor)
            ml_flagged = int(monitor.stats.attacks_detected)
            flows = int(monitor.stats.flows_processed)
            runs.append(
                {
                    "scenario": scenario.name,
                    "note": scenario.note,
                    "seed": seed,
                    "truth": scenario.truth,
                    "hybrid": hybrid,
                    "model_alone": "ATTACK" if ml_flagged else "BENIGN",
                    "flows": flows,
                    "model_flagged_flows": ml_flagged,
                    "packets": int(monitor.stats.packets_seen),
                    "detections": dict(kinds),
                }
            )
            database.close()
        print(f"  finished {scenario.name}", flush=True)

    truth = [r["truth"] for r in runs]
    hybrid = [r["hybrid"] for r in runs]
    alone = [r["model_alone"] for r in runs]
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(truth, hybrid):
        confusion[t][p] += 1

    expected_miss = {s.name for s in SCENARIOS if s.note == "expected miss"}
    kept = [r for r in runs if r["scenario"] not in expected_miss]

    per_scenario = {}
    for scenario in SCENARIOS:
        rows = [r for r in runs if r["scenario"] == scenario.name]
        per_scenario[scenario.name] = {
            "truth": scenario.truth,
            "note": scenario.note,
            "hybrid": dict(Counter(r["hybrid"] for r in rows)),
            "model_alone": dict(Counter(r["model_alone"] for r in rows)),
            "mean_flows": round(sum(r["flows"] for r in rows) / len(rows), 1),
            "mean_model_flagged_flows": round(sum(r["model_flagged_flows"] for r in rows) / len(rows), 1),
            "detections": dict(sum((Counter(r["detections"]) for r in rows), Counter())),
        }

    report = {
        "note": (
            "Synthetic packet scenarios. A check of the logic, not real-world accuracy. "
            "Binary view: any non-BENIGN decision counts as detected."
        ),
        "seeds_per_scenario": args.seeds,
        "hybrid_all_scenarios": _binary(truth, hybrid),
        "hybrid_excluding_expected_misses": _binary(
            [r["truth"] for r in kept], [r["hybrid"] for r in kept]
        ),
        "model_alone_all_scenarios": _binary(truth, alone),
        "three_class_accuracy": round(sum(t == p for t, p in zip(truth, hybrid)) / len(truth), 4),
        "confusion_matrix_truth_by_decision": confusion,
        "per_scenario": per_scenario,
    }
    path = settings.path("reports_dir") / "hybrid_evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
