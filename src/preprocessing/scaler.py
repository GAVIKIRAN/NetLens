"""Numeric scaling and outlier control.

CIC-IDS2017's rate columns (``Flow Bytes/s``, ``Flow Packets/s``) contain
extreme values produced when CICFlowMeter divides by a sub-microsecond flow
duration.  Left alone these dominate any distance-based or gradient-based
model, so the numeric branch of the pipeline winsorises before scaling.

The clip bounds are learned **on the training split only** and then re-applied
verbatim to validation, test and live traffic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.errors import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise every column at a learned upper/lower quantile.

    Parameters
    ----------
    quantile:
        Upper quantile, e.g. ``0.999``.  The lower bound uses ``1 - quantile``.
        ``None`` disables clipping entirely (the transformer becomes a no-op).
    """

    def __init__(self, quantile: float | None = 0.999) -> None:
        self.quantile = quantile

    def fit(self, X, y=None):  # noqa: D102 - sklearn API
        frame = _as_frame(X)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        if self.quantile is None:
            self.lower_ = None
            self.upper_ = None
            return self
        q = float(self.quantile)
        if not 0.5 < q < 1.0:
            raise PreprocessingError(
                f"preprocessing.clip_quantile must be in (0.5, 1.0); got {q}."
            )
        finite = frame.replace([np.inf, -np.inf], np.nan)
        self.upper_ = finite.quantile(q).to_numpy(dtype=np.float64)
        self.lower_ = finite.quantile(1.0 - q).to_numpy(dtype=np.float64)
        # Guard against all-NaN columns.
        self.upper_ = np.nan_to_num(self.upper_, nan=np.finfo(np.float64).max / 4)
        self.lower_ = np.nan_to_num(self.lower_, nan=-np.finfo(np.float64).max / 4)
        return self

    def transform(self, X):  # noqa: D102 - sklearn API
        frame = _as_frame(X)
        values = np.array(
            frame.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64, copy=True
        )
        values[np.isinf(values)] = np.nan
        if self.lower_ is not None and self.upper_ is not None:
            if values.shape[1] != self.lower_.shape[0]:
                raise PreprocessingError(
                    "QuantileClipper received "
                    f"{values.shape[1]} column(s) but was fitted on "
                    f"{self.lower_.shape[0]}."
                )
            values = np.clip(values, self.lower_, self.upper_)
        return values

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        if input_features is not None:
            return np.asarray([str(f) for f in input_features], dtype=object)
        return np.asarray(getattr(self, "feature_names_in_", []), dtype=object)


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))


def build_scaler(kind: str = "standard"):
    """Return a scaler instance, or ``None`` when scaling is disabled."""
    key = str(kind).strip().lower()
    if key in {"none", "off", "false", ""}:
        return None
    if key == "standard":
        return StandardScaler()
    if key == "minmax":
        return MinMaxScaler()
    if key == "robust":
        return RobustScaler()
    raise PreprocessingError(
        f"Unknown preprocessing.scaler '{kind}'.",
        hint="Use one of: standard, minmax, robust, none.",
    )


__all__ = ["QuantileClipper", "build_scaler"]
