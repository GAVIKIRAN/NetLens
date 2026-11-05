"""Measure how long one SHAP or LIME explanation takes with the trained model.

    python scripts/measure_xai_latency.py
    python scripts/measure_xai_latency.py --rows 200

Prints median and 95th-percentile times per flow and writes them to
``reports/xai_latency.json``.  The numbers describe the model currently saved in
``models/trained``, so re-run this after training on the real dataset.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

from _bootstrap import bootstrap


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "samples": len(ordered),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[index], 3),
        "max_ms": round(ordered[-1], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure XAI latency per flow.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rows", type=int, default=100, help="flows timed for SHAP")
    parser.add_argument("--lime-rows", type=int, default=20, help="flows timed for LIME")
    args = parser.parse_args()

    settings = bootstrap(args.config)

    from src.inference.predictor import Predictor
    from src.simulation.synthetic import SyntheticFlowGenerator
    from src.training.train import load_metadata
    from src.xai.explanation_engine import ExplanationService

    predictor = Predictor.load(settings)

    started = time.perf_counter()
    service = ExplanationService(predictor, settings)
    build_ms = (time.perf_counter() - started) * 1000.0

    flows = SyntheticFlowGenerator(seed=5, jitter=0.25).generate(
        max(args.rows, args.lime_rows), attack_ratio=0.5
    )
    results = [predictor.predict_one(flow.features) for flow in flows]

    # A failed explanation returns quickly, so it must not count as a fast one.
    def timed(result, key: str, **options) -> float:
        started = time.perf_counter()
        payload = service.explain_prediction(result, **options)
        elapsed = (time.perf_counter() - started) * 1000.0
        if payload.get(key) is None:
            raise RuntimeError(f"{key} explanation failed: {payload.get('errors')}")
        return elapsed

    try:
        shap_times = [
            timed(result, "shap", use_shap=True, use_lime=False)
            for result in results[: args.rows]
        ]
        lime_times = [
            timed(result, "lime", use_shap=False, use_lime=True)
            for result in results[: args.lime_rows]
        ]
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 2

    metadata = load_metadata(settings.path("model_dir")) or {}
    output = {
        "model": predictor.model_name,
        "explainer": service.status(),
        "trained_on": metadata.get("dataset", {}).get("source_directory", "not recorded"),
        "input_flows": "synthetic flows, single row at a time",
        "machine": platform.platform(),
        "explainer_build_ms": round(build_ms, 1),
        "shap_single_row": _summary(shap_times),
        "lime_single_row": _summary(lime_times),
    }
    path = settings.path("reports_dir") / "xai_latency.json"
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print(f"\n  Model              : {output['model']}")
    print(f"  Trained on         : {output['trained_on']}")
    print(f"  Explainer build    : {output['explainer_build_ms']} ms")
    print(f"  SHAP, one row      : {output['shap_single_row']}")
    print(f"  LIME, one row      : {output['lime_single_row']}")
    print(f"\n  Written to {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
