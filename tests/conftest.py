"""Shared pytest fixtures.

The heavy fixtures are session-scoped: one synthetic CIC-IDS2017-shaped dataset
is written to a temporary directory, one quick training run produces a real set
of artefacts, and every test then works against those.  Nothing in the suite
touches the user's ``data/``, ``models/`` or ``logs/`` directories, and nothing
touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import DEFAULT_CONFIG_PATH, Settings  # noqa: E402
from tests.fixture_dataset import write_dataset  # noqa: E402


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    """An isolated project workspace for the whole test session."""
    root = tmp_path_factory.mktemp("nids_workspace")
    for name in ("data/raw", "data/processed", "data/demo", "models/trained",
                 "models/preprocessing", "reports", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def raw_dataset(workspace: Path) -> Path:
    """CIC-IDS2017-shaped CSV files written into the temporary workspace."""
    directory = workspace / "data" / "raw"
    write_dataset(directory, files=2, rows_per_file=700, seed=11)
    return directory


@pytest.fixture(scope="session")
def settings(workspace: Path, raw_dataset: Path) -> Settings:
    """Settings pointed at the temporary workspace, with a fast training config."""
    overrides = {
        "paths": {
            "raw_data_dir": str(workspace / "data" / "raw"),
            "processed_data_dir": str(workspace / "data" / "processed"),
            "demo_data_dir": str(workspace / "data" / "demo"),
            "model_dir": str(workspace / "models" / "trained"),
            "preprocessing_dir": str(workspace / "models" / "preprocessing"),
            "reports_dir": str(workspace / "reports"),
            "log_dir": str(workspace / "logs"),
            "database_path": str(workspace / "nids_test.db"),
        },
        "demo": {"source_file": str(workspace / "data" / "demo" / "demo_flows.csv")},
        "dataset": {"max_total_rows": 1400, "max_rows_per_class": 400,
                    "min_samples_per_class": 20},
        "training": {
            "n_jobs": 1,
            "models": {
                "random_forest": {"enabled": True,
                                  "params": {"n_estimators": 40, "max_depth": 12, "n_jobs": 1}},
                "logistic_regression": {"enabled": True,
                                        "params": {"max_iter": 200, "C": 1.0, "solver": "saga"}},
                "svm": {"enabled": False, "max_train_rows": 2000, "params": {}},
                "xgboost": {"enabled": True,
                            "params": {"n_estimators": 60, "max_depth": 5,
                                       "learning_rate": 0.3, "n_jobs": 1}},
            },
        },
        "xai": {"background_samples": 60, "lime": {"num_samples": 200, "num_features": 6}},
        "logging": {"console": False, "file": False},
    }
    created = Settings.load(DEFAULT_CONFIG_PATH, overrides=overrides, use_env=False)
    created.ensure_directories()
    return created


@pytest.fixture(scope="session")
def training_artifacts(settings: Settings):
    """Run the real training pipeline once for the whole session."""
    from src.training.train import run_training

    return run_training(settings)


@pytest.fixture(scope="session")
def predictor(settings: Settings, training_artifacts):
    from src.inference.predictor import Predictor

    return Predictor.load(settings)


@pytest.fixture(scope="session")
def explanation_service(settings: Settings, predictor):
    from src.xai.explanation_engine import ExplanationService

    return ExplanationService(predictor, settings)


@pytest.fixture()
def database(tmp_path: Path):
    """A fresh SQLite store per test."""
    from src.database.database import Database

    store = Database(tmp_path / "test.db")
    yield store
    store.close()


@pytest.fixture(scope="session")
def synthetic_flows():
    """A small set of coherent synthetic flow records (no network involved)."""
    from src.simulation.synthetic import SyntheticFlowGenerator

    return SyntheticFlowGenerator(seed=5, jitter=0.3).generate(12, attack_ratio=0.5)


@pytest.fixture(scope="session")
def demo_file(settings: Settings):
    """Generate the demo CSV once (synthetic source, no dataset needed)."""
    from src.simulation.replay import build_demo_dataset

    path, _report = build_demo_dataset(settings, rows=120, prefer_real=False, seed=3)
    return path
