"""One-command demo setup - works before you have downloaded CIC-IDS2017.

Generates a small CIC-IDS2017-*shaped* synthetic dataset, trains the models on
it, builds the demo replay file and leaves the app ready to launch.  Use it to
verify that the whole installation works end to end in about a minute.

    python scripts/quickstart_demo.py

IMPORTANT: the metrics this produces describe the *synthetic fixture*, not
CIC-IDS2017.  For real results, put the dataset in ``data/raw`` and run
``python scripts/train_model.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

BANNER = "=" * 72


def run_pipeline_check(settings, flows: int = 12) -> None:
    """Send simulated flows through every stage, into a throwaway database."""
    from src.database.database import Database
    from src.inference.predictor import Predictor
    from src.simulation.synthetic import SyntheticFlowGenerator
    from src.xai.explanation_engine import ExplanationService

    predictor = Predictor.load(settings)
    explainer = ExplanationService(predictor, settings)
    generator = SyntheticFlowGenerator(seed=11, jitter=0.25)

    with tempfile.TemporaryDirectory() as folder:
        database = Database(Path(folder) / "pipeline_check.db")
        try:
            attack_shown = False
            for flow in generator.generate(flows, attack_ratio=0.5):
                result = predictor.predict_one(flow.features)
                explained = explainer.explain_prediction(result, use_shap=True, use_lime=True)
                flow_id = database.insert_flow(flow.identity, flow.features, mode="QUICKSTART")
                payload = result.to_dict()
                payload["formula"] = result.risk.formula
                payload["top_features"] = explained["top_features"]
                payload["explanation"] = explained["narrative"]["text"]
                prediction_id = database.insert_prediction(flow_id, payload, mode="QUICKSTART")
                if result.is_attack:
                    database.insert_alert(prediction_id, flow_id, flow.identity, payload,
                                          mode="QUICKSTART")
                print(f"    true={flow.label:<26s} predicted={result.predicted_label:<26s} "
                      f"confidence={result.confidence:.3f}  risk={result.risk.score:6.2f} "
                      f"{result.risk.level}")
                if result.is_attack and not attack_shown:
                    attack_shown = True
                    first_attack = (result, explained["narrative"]["text"])
            counts = database.counts()
            print(f"\n    Database rows: {counts['flows']} flows, "
                  f"{counts['predictions']} predictions, {counts['alerts']} alerts")
            if attack_shown:
                print(f"\n    Risk formula: {first_attack[0].risk.formula}")
                print("\n    Explanation of the first attack:\n")
                for line in first_attack[1].splitlines():
                    print(f"      {line}")
        finally:
            database.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up a synthetic end-to-end demo.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rows", type=int, default=900, help="rows per generated file")
    parser.add_argument("--files", type=int, default=3)
    parser.add_argument("--keep", action="store_true",
                        help="keep the generated fixture CSVs in data/fixture")
    args = parser.parse_args()

    settings = bootstrap(args.config)
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.simulation.replay import build_demo_dataset
    from src.training.train import run_training
    from tests.fixture_dataset import write_dataset

    fixture_dir = PROJECT_ROOT / "data" / "fixture"
    print(BANNER)
    print("  STEP 1/4  Generating a CIC-IDS2017-shaped synthetic dataset")
    print(BANNER)
    written = write_dataset(fixture_dir, files=args.files, rows_per_file=args.rows)
    for path in written:
        print(f"    {path}")

    print()
    print(BANNER)
    print("  STEP 2/4  Training the models on the synthetic dataset")
    print(BANNER)
    artifacts = run_training(settings, data_dir=fixture_dir, quick=True)
    test = (artifacts.metrics.get("best_model", {}).get("test") or {})
    print(f"\n    Selected model : {artifacts.best_model_name}")
    print(f"    Accuracy       : {test.get('accuracy', 0):.4f}   (synthetic data!)")
    print(f"    Macro F1       : {test.get('f1_macro', 0):.4f}   (synthetic data!)")

    print()
    print(BANNER)
    print("  STEP 3/4  Building the safe demo replay file")
    print(BANNER)
    path, report = build_demo_dataset(settings, rows=1500, prefer_real=True)
    print(f"    {path}  ({report['rows']:,} rows, source={report['source']})")

    if not args.keep:
        shutil.rmtree(fixture_dir, ignore_errors=True)
        print(f"\n    Removed the temporary fixture directory {fixture_dir}")

    print()
    print(BANNER)
    print("  STEP 4/4  Running the full pipeline on simulated traffic")
    print("            packets -> flows -> features -> prediction -> risk")
    print("            -> SHAP + LIME -> explanation -> database row")
    print(BANNER)
    run_pipeline_check(settings)

    print()
    print(BANNER)
    print("  READY.  Launch the dashboard with:")
    print("      streamlit run app.py")
    print()
    print("  Then open Live Monitor -> mode DEMO -> Start.")
    print()
    print("  WARNING: these metrics come from SYNTHETIC data and say nothing")
    print("  about CIC-IDS2017. Put the real CSVs in data/raw and run")
    print("      python scripts/train_model.py")
    print("  before quoting any number in your report.")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
