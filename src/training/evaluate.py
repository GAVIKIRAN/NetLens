"""Model evaluation (Phase 4).

Metric averaging - stated explicitly because it matters
--------------------------------------------------------
This is a **multiclass** problem with an extremely skewed prior (BENIGN is
~80 % of CIC-IDS2017), so a single "accuracy" number is close to meaningless.
Three averaging schemes are therefore reported side by side:

``macro``
    Computes the metric independently for each class and takes the unweighted
    mean.  Every attack class counts the same as BENIGN, so a model that
    ignores Heartbleed is punished.  **This is the model-selection metric.**

``weighted``
    Same per-class metrics, averaged with weights proportional to class
    support.  It tracks overall correctness and is dominated by BENIGN.

``micro`` / accuracy
    Pools all decisions before computing the metric.  In single-label
    multiclass classification micro-precision = micro-recall = accuracy, which
    is why only accuracy is printed.

In addition a **binary security view** collapses every attack class into a
single ``ATTACK`` class, which is what an analyst actually cares about:
detection rate (recall on ATTACK) and false-positive rate (fraction of benign
flows wrongly alerted on).  A per-class one-vs-rest FPR table is also produced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

AVERAGING_NOTE = (
    "macro = unweighted mean over classes (every attack class counts equally, "
    "used for model selection); weighted = mean weighted by class support; "
    "accuracy = micro average. Binary view collapses all attacks into ATTACK."
)


@dataclass
class EvaluationResult:
    """All metrics for one model on one split."""

    model_name: str = ""
    split: str = ""
    n_samples: int = 0
    class_names: list[str] = field(default_factory=list)

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    roc_auc_ovr_macro: float | None = None

    per_class: list[dict[str, Any]] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    classification_report_text: str = ""
    classification_report_dict: dict[str, Any] = field(default_factory=dict)

    binary_detection_rate: float = 0.0
    binary_false_positive_rate: float = 0.0
    binary_precision: float = 0.0
    binary_f1: float = 0.0
    binary_confusion: dict[str, int] = field(default_factory=dict)

    predict_latency_ms: float = 0.0
    averaging_note: str = AVERAGING_NOTE

    def headline(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "macro_f1": self.f1_macro,
            "weighted_f1": self.f1_weighted,
            "detection_rate": self.binary_detection_rate,
            "false_positive_rate": self.binary_false_positive_rate,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "split": self.split,
            "n_samples": self.n_samples,
            "class_names": self.class_names,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "roc_auc_ovr_macro": self.roc_auc_ovr_macro,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "classification_report_text": self.classification_report_text,
            "classification_report_dict": self.classification_report_dict,
            "binary": {
                "detection_rate": self.binary_detection_rate,
                "false_positive_rate": self.binary_false_positive_rate,
                "precision": self.binary_precision,
                "f1": self.binary_f1,
                "confusion": self.binary_confusion,
            },
            "predict_latency_ms": self.predict_latency_ms,
            "averaging_note": self.averaging_note,
        }
        return payload

    def confusion_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.confusion_matrix, index=self.class_names, columns=self.class_names
        )

    def per_class_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_class)


def evaluate_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    *,
    model_name: str = "model",
    split: str = "test",
    benign_label: str = "BENIGN",
    probabilities: np.ndarray | None = None,
) -> EvaluationResult:
    """Score ``model`` on one split and return every metric the report needs."""
    labels = list(range(len(class_names)))
    class_names = [str(c) for c in class_names]

    start = time.perf_counter()
    y_pred = np.asarray(model.predict(X)).astype(int)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    latency = elapsed_ms / max(1, len(X))

    y_true = np.asarray(y_true).astype(int)

    result = EvaluationResult(
        model_name=model_name,
        split=split,
        n_samples=int(len(y_true)),
        class_names=class_names,
        predict_latency_ms=round(latency, 5),
    )

    result.accuracy = float(accuracy_score(y_true, y_pred))
    result.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    result.precision_macro = float(
        precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    result.recall_macro = float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    result.f1_macro = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    result.precision_weighted = float(
        precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    result.recall_weighted = float(
        recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    result.f1_weighted = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    result.confusion_matrix = matrix.astype(int).tolist()

    result.classification_report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0, digits=4
    )
    result.classification_report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    # ------------------------------------------------ per-class + OvR FPR
    total = matrix.sum()
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = int(total - true_positive - false_negative - false_positive)
        support = true_positive + false_negative
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, support)
        f1 = _ratio(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "class": name,
                "support": support,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "false_positive_rate": round(
                    _ratio(false_positive, false_positive + true_negative), 6
                ),
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "true_negatives": true_negative,
            }
        )
    result.per_class = per_class

    # ------------------------------------------------------- binary view
    if benign_label in class_names:
        benign_index = class_names.index(benign_label)
        true_attack = y_true != benign_index
        pred_attack = y_pred != benign_index
        tp = int(np.sum(true_attack & pred_attack))
        fn = int(np.sum(true_attack & ~pred_attack))
        fp = int(np.sum(~true_attack & pred_attack))
        tn = int(np.sum(~true_attack & ~pred_attack))
        result.binary_confusion = {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
        }
        result.binary_detection_rate = round(_ratio(tp, tp + fn), 6)
        result.binary_false_positive_rate = round(_ratio(fp, fp + tn), 6)
        result.binary_precision = round(_ratio(tp, tp + fp), 6)
        result.binary_f1 = round(
            _ratio(
                2 * result.binary_precision * result.binary_detection_rate,
                result.binary_precision + result.binary_detection_rate,
            ),
            6,
        )

    # ---------------------------------------------------------- ROC AUC
    if probabilities is None:
        probabilities = _safe_proba(model, X)
    if probabilities is not None and len(np.unique(y_true)) > 1:
        try:
            result.roc_auc_ovr_macro = float(
                roc_auc_score(
                    y_true, probabilities, multi_class="ovr", average="macro", labels=labels
                )
            )
        except Exception as exc:  # pragma: no cover - degenerate splits
            logger.debug("ROC AUC unavailable for %s: %s", model_name, exc)

    logger.info(
        "[%s/%s] accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f detection=%.4f fpr=%.4f",
        model_name, split, result.accuracy, result.f1_macro, result.f1_weighted,
        result.binary_detection_rate, result.binary_false_positive_rate,
    )
    return result


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    value = float(numerator) / float(denominator)
    return 0.0 if (np.isnan(value) or np.isinf(value)) else value


def _safe_proba(model, X) -> np.ndarray | None:
    """Return class probabilities if the estimator exposes them."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return np.asarray(model.predict_proba(X), dtype=np.float64)
    except Exception:  # pragma: no cover - estimator specific
        return None


def comparison_frame(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    """Build the model-comparison table rendered on the dashboard."""
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.model_name,
                "split": result.split,
                "accuracy": round(result.accuracy, 6),
                "balanced_accuracy": round(result.balanced_accuracy, 6),
                "precision_macro": round(result.precision_macro, 6),
                "recall_macro": round(result.recall_macro, 6),
                "macro_f1": round(result.f1_macro, 6),
                "weighted_f1": round(result.f1_weighted, 6),
                "detection_rate": round(result.binary_detection_rate, 6),
                "false_positive_rate": round(result.binary_false_positive_rate, 6),
                "roc_auc_ovr_macro": (
                    round(result.roc_auc_ovr_macro, 6)
                    if result.roc_auc_ovr_macro is not None
                    else None
                ),
                "predict_latency_ms": round(result.predict_latency_ms, 5),
            }
        )
    return pd.DataFrame(rows)


def write_evaluation_reports(
    metrics: dict[str, Any], reports_dir: str | Path, source_directory: str = ""
) -> list[Path]:
    """Write the measured metrics to ``reports/`` and return the files written.

    The selected model's held-out TEST results are written in full: summary,
    per-class metrics, confusion matrix and classification report.  The dataset
    directory is recorded with them, so numbers from the synthetic quickstart
    fixture can never be mistaken for CIC-IDS2017 results.
    """
    folder = Path(reports_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    synthetic = Path(source_directory).name == "fixture" if source_directory else False
    data_note = (
        "SYNTHETIC FIXTURE: these numbers say nothing about CIC-IDS2017."
        if synthetic
        else f"Dataset directory: {source_directory or 'not recorded'}"
    )

    rows: list[dict[str, Any]] = []
    for candidate in metrics.get("candidates", []):
        for split in ("validation", "test"):
            result = candidate.get(split)
            if not result:
                continue
            binary = result.get("binary", {})
            rows.append(
                {
                    "model": candidate.get("name", ""),
                    "split": split,
                    "n_samples": result.get("n_samples"),
                    "accuracy": result.get("accuracy"),
                    "balanced_accuracy": result.get("balanced_accuracy"),
                    "precision_macro": result.get("precision_macro"),
                    "recall_macro": result.get("recall_macro"),
                    "f1_macro": result.get("f1_macro"),
                    "precision_weighted": result.get("precision_weighted"),
                    "recall_weighted": result.get("recall_weighted"),
                    "f1_weighted": result.get("f1_weighted"),
                    "detection_rate": binary.get("detection_rate"),
                    "false_positive_rate": binary.get("false_positive_rate"),
                    "predict_latency_ms": result.get("predict_latency_ms"),
                    "fit_seconds": candidate.get("fit_seconds"),
                    "selected": candidate.get("name")
                    == (metrics.get("best_model") or {}).get("name"),
                    "synthetic_data": synthetic,
                }
            )
    summary = folder / "evaluation_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    written.append(summary)

    best = metrics.get("best_model") or {}
    test = best.get("test") or {}
    if test:
        report = folder / "classification_report.txt"
        report.write_text(
            f"{data_note}\n"
            f"Selected model: {best.get('name')}\n"
            f"Split: test ({test.get('n_samples')} samples)\n"
            f"Averaging: {metrics.get('averaging_note', '')}\n\n"
            f"{test.get('classification_report_text', '')}\n",
            encoding="utf-8",
        )
        written.append(report)

        names = test.get("class_names", [])
        confusion = folder / "confusion_matrix.csv"
        pd.DataFrame(test.get("confusion_matrix", []), index=names, columns=names).to_csv(
            confusion, index_label="true \\ predicted"
        )
        written.append(confusion)

        per_class = folder / "per_class_metrics.csv"
        pd.DataFrame(test.get("per_class", [])).to_csv(per_class, index=False)
        written.append(per_class)

    logger.info("Evaluation reports written to %s", folder)
    return written


__all__ = [
    "EvaluationResult",
    "evaluate_model",
    "comparison_frame",
    "write_evaluation_reports",
    "AVERAGING_NOTE",
]
