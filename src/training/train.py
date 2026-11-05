"""End-to-end training pipeline (Phases 1-5).

``python scripts/train_model.py`` calls :func:`run_training`, which performs
the whole sequence and leaves a self-contained set of artefacts on disk:

::

    models/preprocessing/preprocessing_pipeline.joblib   fitted transformations
    models/preprocessing/feature_contract.json           the feature order contract
    models/preprocessing/label_encoder.joblib            class <-> integer codec
    models/trained/<name>_model.joblib                   every trained candidate
    models/trained/best_model.joblib                     the selected model
    models/trained/model_metadata.json                   provenance + config
    models/trained/evaluation_metrics.json               all measured metrics
    models/trained/xai_background.npz                    SHAP/LIME background
    reports/dataset_profile.json                         dataset statistics
    reports/model_comparison.csv                         the comparison table

Nothing in these artefacts is invented: every number is measured on the data
that was actually loaded.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config.settings import Settings, get_settings
from src.data.cleaner import clean_dataset
from src.data.loader import load_dataset
from src.data.profiler import profile_dataset, save_profile
from src.data.splitter import split_dataset
from src.features.contract import resolve_features
from src.preprocessing.encoder import LabelCodec
from src.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from src.training.evaluate import (
    AVERAGING_NOTE,
    comparison_frame,
    evaluate_model,
    write_evaluation_reports,
)
from src.training.model_selection import (
    ModelCandidate,
    XGBOOST_AVAILABLE,
    build_candidate_models,
    compute_sample_weights,
    select_best_model,
)
from src.utils.errors import ModelError
from src.utils.helpers import ensure_dir, utc_now_iso, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

BEST_MODEL_FILENAME = "best_model.joblib"
METADATA_FILENAME = "model_metadata.json"
METRICS_FILENAME = "evaluation_metrics.json"
BACKGROUND_FILENAME = "xai_background.npz"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"


@dataclass
class TrainingArtifacts:
    """Return value of :func:`run_training`."""

    best_model_name: str = ""
    model_dir: Path = Path()
    preprocessing_dir: Path = Path()
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary_line(self) -> str:
        best = self.metrics.get("best_model", {})
        test = best.get("test", {}) or {}
        return (
            f"best={self.best_model_name} "
            f"test_accuracy={test.get('accuracy', 0):.4f} "
            f"test_macro_f1={test.get('f1_macro', 0):.4f}"
        )


def run_training(
    settings: Settings | None = None,
    *,
    data_dir: str | Path | None = None,
    max_total_rows: int | None = None,
    quick: bool = False,
) -> TrainingArtifacts:
    """Run the complete training pipeline and persist every artefact.

    Parameters
    ----------
    quick:
        Shrink the row cap and the forest sizes for a fast smoke run.  Used by
        the test-suite and by ``scripts/train_model.py --quick``.
    """
    settings = settings or get_settings()
    settings.ensure_directories()

    dataset_cfg = settings.section("dataset")
    features_cfg = settings.section("features")
    training_cfg = settings.section("training")
    selection_cfg = settings.section("model_selection")
    random_state = int(training_cfg.get("random_state", 42))
    benign_label = str(settings.get("inference.benign_label", "BENIGN"))

    raw_dir = Path(data_dir) if data_dir else settings.path("raw_data_dir")
    row_cap = max_total_rows or dataset_cfg.get("max_total_rows")
    if quick:
        row_cap = min(int(row_cap or 20000), 20000)

    started = time.perf_counter()
    logger.info("=" * 78)
    logger.info("Training run started - dataset directory: %s", raw_dir)
    logger.info("=" * 78)

    # ------------------------------------------------------------- 1. load
    load_result = load_dataset(
        raw_dir,
        patterns=tuple(dataset_cfg.get("file_patterns", ["*.csv"])),
        encoding=str(dataset_cfg.get("encoding", "latin-1")),
        label_column=str(dataset_cfg.get("label_column", "Label")),
        max_rows_per_class=dataset_cfg.get("max_rows_per_class"),
        max_total_rows=row_cap,
        max_rows_per_file=dataset_cfg.get("max_rows_per_file"),
        random_state=random_state,
    )
    label_column = load_result.label_column

    # ------------------------------------------------------------ 2. clean
    frame, clean_report = clean_dataset(
        load_result.frame,
        label_column,
        drop_duplicates=bool(dataset_cfg.get("drop_duplicates", True)),
        min_samples_per_class=int(dataset_cfg.get("min_samples_per_class", 50)),
        rare_class_label=str(dataset_cfg.get("rare_class_label", "Rare_Attack")),
        benign_label=benign_label,
    )

    # ---------------------------------------------------------- 3. profile
    profile = profile_dataset(
        frame,
        label_column,
        extra={
            "loading": load_result.summary(),
            "cleaning": clean_report.to_dict(),
            "source_directory": str(raw_dir),
        },
    )
    save_profile(profile, settings.path("reports_dir"))

    # ------------------------------------------------- 4. resolve features
    resolved = resolve_features(
        [c for c in frame.columns if c != label_column],
        use_protocol=bool(features_cfg.get("use_protocol", True)),
        use_destination_port=bool(features_cfg.get("use_destination_port", False)),
        use_unverified_columns=bool(features_cfg.get("use_unverified_columns", False)),
    )
    logger.info(
        "Modelling %d feature(s): %d numeric + %d categorical",
        len(resolved.all), len(resolved.numeric), len(resolved.categorical),
    )

    X = frame[resolved.all]
    y_text = frame[label_column].astype(str)

    # ------------------------------------------------------------ 5. split
    splits = split_dataset(
        X,
        y_text,
        test_size=float(settings.get("preprocessing.test_size", 0.2)),
        validation_size=float(settings.get("preprocessing.validation_size", 0.2)),
        stratify=bool(settings.get("preprocessing.stratify", True)),
        random_state=random_state,
    )

    # ------------------------------------------- 6. fit preprocessing (train)
    pipeline = PreprocessingPipeline(resolved, PreprocessingConfig.from_settings(settings))
    X_train = pipeline.fit_transform(splits.X_train, splits.y_train)
    X_val = pipeline.transform(splits.X_val)
    X_test = pipeline.transform(splits.X_test)

    codec = LabelCodec().fit(pd.concat([splits.y_train, splits.y_val, splits.y_test]))
    y_train = codec.transform(splits.y_train)
    y_val = codec.transform(splits.y_val)
    y_test = codec.transform(splits.y_test)
    class_names = codec.classes_
    logger.info("Classes (%d): %s", len(class_names), ", ".join(class_names))

    # ------------------------------------------------------------ 7. train
    use_class_weight = (
        str(settings.get("preprocessing.class_balance.strategy", "class_weight")).lower()
        == "class_weight"
    )
    if quick:
        training_cfg = _shrink_for_quick(training_cfg)
    models = build_candidate_models(
        training_cfg,
        random_state=random_state,
        n_jobs=int(training_cfg.get("n_jobs", -1)),
        n_classes=len(class_names),
        use_class_weight=use_class_weight,
    )

    svm_cap = int(
        (training_cfg.get("models", {}).get("svm", {}) or {}).get("max_train_rows", 50000)
    )
    candidates: list[ModelCandidate] = []

    for name, estimator in models.items():
        candidate = ModelCandidate(name=name, params=_describe_params(estimator))
        fit_X, fit_y = X_train, y_train
        if name == "svm" and len(X_train) > svm_cap:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X_train), size=svm_cap, replace=False)
            fit_X, fit_y = X_train[idx], y_train[idx]
            logger.info("SVM trained on a %s-row sub-sample for tractability.", f"{svm_cap:,}")

        logger.info("Training %s on %s rows x %d features ...",
                    name, f"{len(fit_X):,}", fit_X.shape[1])
        start = time.perf_counter()
        try:
            if name == "xgboost" and use_class_weight:
                _fit_with_retry(
                    estimator, fit_X, fit_y, name=name,
                    sample_weight=compute_sample_weights(fit_y),
                )
            else:
                _fit_with_retry(estimator, fit_X, fit_y, name=name)
            candidate.fit_seconds = time.perf_counter() - start
            candidate.estimator = estimator
            candidate.supports_proba = hasattr(estimator, "predict_proba")
            candidate.validation = evaluate_model(
                estimator, X_val, y_val, class_names,
                model_name=name, split="validation", benign_label=benign_label,
            )
        except Exception as exc:  # keep going; one bad model must not kill the run
            candidate.error = f"{type(exc).__name__}: {exc}"
            candidate.fit_seconds = time.perf_counter() - start
            logger.exception("Training failed for %s", name)
        candidates.append(candidate)

    failed = [c for c in candidates if c.estimator is None]
    if failed:
        logger.warning(
            "INCOMPLETE RUN: %d model(s) failed and are missing from the comparison: %s. "
            "Selection is only among the models that finished; free up memory and re-train "
            "before quoting these results.",
            len(failed), "; ".join(f"{c.name} ({c.error})" for c in failed),
        )

    # --------------------------------------------------------- 8. selection
    selection = select_best_model(
        candidates,
        primary_metric=str(selection_cfg.get("primary_metric", "macro_f1")),
        tie_break_margin=float(selection_cfg.get("tie_break_margin", 0.005)),
    )
    best = next(c for c in candidates if c.name == selection.best_name)

    # ------------------------------ 9. single, final evaluation on TEST split
    best.test = evaluate_model(
        best.estimator, X_test, y_test, class_names,
        model_name=best.name, split="test", benign_label=benign_label,
    )
    for candidate in candidates:
        if candidate.estimator is not None and candidate.name != best.name:
            candidate.test = evaluate_model(
                candidate.estimator, X_test, y_test, class_names,
                model_name=candidate.name, split="test", benign_label=benign_label,
            )

    # ------------------------------------------------------- 10. persistence
    model_dir = ensure_dir(settings.path("model_dir"))
    prep_dir = ensure_dir(settings.path("preprocessing_dir"))
    reports_dir = ensure_dir(settings.path("reports_dir"))

    pipeline.save(prep_dir)
    codec.save(prep_dir / LABEL_ENCODER_FILENAME)

    for candidate in candidates:
        path = model_dir / f"{candidate.name}_model.joblib"
        if candidate.estimator is not None:
            joblib.dump(candidate.estimator, path)
        else:
            # A file from an earlier run would not match this run's preprocessing.
            path.unlink(missing_ok=True)
    joblib.dump(best.estimator, model_dir / BEST_MODEL_FILENAME)

    background = _make_background(
        X_train, y_train, int(settings.get("xai.background_samples", 200)), random_state
    )
    np.savez_compressed(
        model_dir / BACKGROUND_FILENAME,
        X=background["X"],
        y=background["y"],
        feature_names=np.array(pipeline.output_features, dtype=object),
    )

    metadata = {
        "created_at": utc_now_iso(),
        "project_version": settings.get("project.version", "1.0.0"),
        "best_model": best.name,
        "best_model_params": best.params,
        "selection": selection.to_dict(),
        "classes": class_names,
        "n_classes": len(class_names),
        "benign_label": benign_label,
        "input_features": pipeline.input_features,
        "output_features": pipeline.output_features,
        "feature_policy": {
            "use_protocol": bool(features_cfg.get("use_protocol", True)),
            "use_destination_port": bool(features_cfg.get("use_destination_port", False)),
            "packet_length_basis": features_cfg.get("packet_length_basis", "payload"),
            "resolved": resolved.to_dict(),
        },
        "preprocessing": pipeline.metadata().to_dict(),
        "dataset": {
            "source_directory": str(raw_dir),
            "files": [Path(p).name for p in load_result.files],
            "rows_available": load_result.total_rows_available,
            "rows_loaded": load_result.rows_loaded,
            "class_counts_full": load_result.class_counts_full,
            "class_counts_loaded": load_result.class_counts_loaded,
            "cleaning": clean_report.to_dict(),
        },
        "splits": splits.summary(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost_available": XGBOOST_AVAILABLE,
        },
        "training_seconds": round(time.perf_counter() - started, 2),
        "averaging_note": AVERAGING_NOTE,
    }
    write_json(model_dir / METADATA_FILENAME, metadata)

    metrics = {
        "generated_at": utc_now_iso(),
        "primary_metric": selection.primary_metric,
        "averaging_note": AVERAGING_NOTE,
        "best_model": {
            "name": best.name,
            "validation": best.validation.to_dict() if best.validation else None,
            "test": best.test.to_dict() if best.test else None,
        },
        "candidates": [c.to_dict() for c in candidates],
    }
    write_json(model_dir / METRICS_FILENAME, metrics)

    validation_table = comparison_frame(
        [c.validation for c in candidates if c.validation is not None]
    )
    test_table = comparison_frame([c.test for c in candidates if c.test is not None])
    comparison = pd.concat([validation_table, test_table], ignore_index=True)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    selection.table.to_csv(reports_dir / "model_selection.csv", index=False)
    write_evaluation_reports(metrics, reports_dir, source_directory=str(raw_dir))

    elapsed = time.perf_counter() - started
    logger.info("=" * 78)
    logger.info("Training complete in %.1fs - selected '%s'", elapsed, best.name)
    logger.info("Selection rationale: %s", selection.rationale)
    if best.test:
        logger.info(
            "TEST metrics: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f "
            "detection_rate=%.4f false_positive_rate=%.4f",
            best.test.accuracy, best.test.f1_macro, best.test.f1_weighted,
            best.test.binary_detection_rate, best.test.binary_false_positive_rate,
        )
    logger.info("=" * 78)

    return TrainingArtifacts(
        best_model_name=best.name,
        model_dir=model_dir,
        preprocessing_dir=prep_dir,
        metadata=metadata,
        metrics=metrics,
        comparison=comparison,
    )


def _fit_with_retry(estimator, X, y, *, name: str, **fit_kwargs) -> None:
    """Fit a model, retrying a failed parallel fit once on a single core.

    A worker that fails, for example when the machine runs short of memory, makes
    the worker pool shut down, and that shutdown can raise its own unrelated error.
    One core uses less memory and has no pool, so a retry usually succeeds and any
    remaining failure reports its real cause.
    """
    try:
        estimator.fit(X, y, **fit_kwargs)
        return
    except Exception as exc:
        try:
            parallel = estimator.get_params(deep=False).get("n_jobs") not in (None, 1)
        except Exception:  # pragma: no cover - exotic estimators
            parallel = False
        if not parallel:
            raise
        logger.warning(
            "%s failed in parallel (%s: %s); retrying on a single core. "
            "This often means the machine is short of memory.",
            name, type(exc).__name__, exc,
        )
    estimator.set_params(n_jobs=1)
    estimator.fit(X, y, **fit_kwargs)


def _make_background(
    X: np.ndarray, y: np.ndarray, n_samples: int, random_state: int
) -> dict[str, np.ndarray]:
    """Draw a class-stratified background sample for SHAP/LIME."""
    n_samples = max(20, min(int(n_samples), len(X)))
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    per_class = max(1, n_samples // max(1, len(classes)))
    picked: list[int] = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        take = min(per_class, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(picked) < n_samples:
        remaining = np.setdiff1d(np.arange(len(X)), np.array(picked, dtype=int))
        if len(remaining):
            extra = min(n_samples - len(picked), len(remaining))
            picked.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    index = np.array(sorted(set(picked)), dtype=int)
    return {"X": X[index], "y": y[index]}


def _describe_params(estimator) -> dict[str, Any]:
    """Readable parameter snapshot for the metadata file."""
    try:
        params = estimator.get_params(deep=False)
    except Exception:  # pragma: no cover - exotic estimators
        return {}
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = type(value).__name__
    return out


def _shrink_for_quick(training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Reduce model sizes for the fast smoke path."""
    cfg = {k: v for k, v in training_cfg.items()}
    models = {k: dict(v) for k, v in (cfg.get("models", {}) or {}).items()}
    if "random_forest" in models:
        params = dict(models["random_forest"].get("params", {}))
        params.update({"n_estimators": 60, "max_depth": 14})
        models["random_forest"]["params"] = params
    if "xgboost" in models:
        params = dict(models["xgboost"].get("params", {}))
        params.update({"n_estimators": 80, "max_depth": 6})
        models["xgboost"]["params"] = params
    if "decision_tree" in models:
        params = dict(models["decision_tree"].get("params", {}))
        params.update({"max_depth": 14})
        models["decision_tree"]["params"] = params
    if "logistic_regression" in models:
        params = dict(models["logistic_regression"].get("params", {}))
        params.update({"max_iter": 300})
        models["logistic_regression"]["params"] = params
    cfg["models"] = models
    return cfg


def load_metrics(model_dir: str | Path) -> dict[str, Any] | None:
    """Read ``evaluation_metrics.json`` if training has been run."""
    from src.utils.helpers import read_json

    return read_json(Path(model_dir) / METRICS_FILENAME)


def load_metadata(model_dir: str | Path) -> dict[str, Any] | None:
    """Read ``model_metadata.json`` if training has been run."""
    from src.utils.helpers import read_json

    return read_json(Path(model_dir) / METADATA_FILENAME)


__all__ = [
    "TrainingArtifacts",
    "run_training",
    "load_metrics",
    "load_metadata",
    "BEST_MODEL_FILENAME",
    "METADATA_FILENAME",
    "METRICS_FILENAME",
    "BACKGROUND_FILENAME",
    "LABEL_ENCODER_FILENAME",
]
