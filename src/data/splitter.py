"""Train / validation / test splitting.

The split happens **before** anything is fitted, so no statistic computed on
validation or test data can reach the training pipeline.  Splits are stratified
on the class label whenever every class has enough members; otherwise the code
falls back to a plain random split and says so in the log rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.errors import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplits:
    """Container for the three disjoint splits."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    stratified: bool = True

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.X_train)),
            "validation": int(len(self.X_val)),
            "test": int(len(self.X_test)),
        }

    def class_distribution(self) -> dict[str, dict[str, int]]:
        return {
            "train": {str(k): int(v) for k, v in self.y_train.value_counts().items()},
            "validation": {str(k): int(v) for k, v in self.y_val.value_counts().items()},
            "test": {str(k): int(v) for k, v in self.y_test.value_counts().items()},
        }

    def summary(self) -> dict[str, Any]:
        return {
            "sizes": self.sizes(),
            "stratified": self.stratified,
            "class_distribution": self.class_distribution(),
        }


def _can_stratify(labels: pd.Series, test_size: float) -> bool:
    counts = labels.value_counts()
    if counts.min() < 2:
        return False
    # Every class needs at least one member on both sides of the split.
    return bool((counts * test_size).min() >= 1.0)


def split_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> DataSplits:
    """Split into train / validation / test.

    ``validation_size`` is a fraction of the *remaining* data after the test
    split is carved out, i.e. the defaults 0.2/0.2 give a 64/16/20 split.
    """
    if len(features) != len(labels):
        raise DatasetError("Feature/label length mismatch during splitting.")
    if len(features) < 10:
        raise DatasetError(
            f"Only {len(features)} usable rows; not enough to train and evaluate."
        )

    use_stratify = bool(stratify) and _can_stratify(labels, test_size)
    if stratify and not use_stratify:
        logger.warning(
            "Stratified splitting disabled: at least one class is too small "
            "for the requested test_size."
        )

    X_rest, X_test, y_rest, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if use_stratify else None,
        shuffle=True,
    )

    use_stratify_val = use_stratify and _can_stratify(y_rest, validation_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_rest if use_stratify_val else None,
        shuffle=True,
    )

    splits = DataSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        stratified=use_stratify and use_stratify_val,
    )
    logger.info(
        "Split sizes -> train=%s, validation=%s, test=%s (stratified=%s)",
        f"{len(splits.X_train):,}", f"{len(splits.X_val):,}",
        f"{len(splits.X_test):,}", splits.stratified,
    )
    return splits


__all__ = ["DataSplits", "split_dataset"]
