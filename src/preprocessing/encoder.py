"""Categorical encoding and label handling.

Two categorical inputs exist in this project:

``protocol``
    One-hot encoded against a *fixed* category list so that the encoded width
    never changes between training and live inference, even if the live capture
    happens to contain only TCP.

``destination_port``
    Bucketed into ``well_known / registered / dynamic`` **before** one-hot
    encoding.  Feeding the raw port number would let the model memorise the
    CIC-IDS2017 port-to-attack mapping (SSH-Patator on 22, web attacks on 80),
    which does not transfer to another network.  This input is disabled by
    default; see ``features.use_destination_port`` in ``config.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.features.contract import PORT_BUCKETS, PROTOCOL_CATEGORIES, port_bucket
from src.utils.errors import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortBucketTransformer(BaseEstimator, TransformerMixin):
    """Map raw port numbers onto coarse service buckets.

    Stateless (nothing is learned), so it is safe on any split.
    """

    def fit(self, X, y=None):  # noqa: D102 - sklearn API
        self.n_features_in_ = np.asarray(X).reshape(len(X), -1).shape[1]
        return self

    def transform(self, X):  # noqa: D102 - sklearn API
        frame = pd.DataFrame(X).copy()
        for column in frame.columns:
            frame[column] = frame[column].map(port_bucket)
        return frame.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        if input_features is None:
            return np.asarray([f"port_bucket_{i}" for i in range(getattr(self, "n_features_in_", 1))])
        return np.asarray([str(name) for name in input_features], dtype=object)


def build_protocol_encoder() -> OneHotEncoder:
    """One-hot encoder pinned to the known IP protocol numbers."""
    return OneHotEncoder(
        categories=[list(PROTOCOL_CATEGORIES)],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )


def build_port_encoder() -> OneHotEncoder:
    """One-hot encoder pinned to the three port buckets."""
    return OneHotEncoder(
        categories=[list(PORT_BUCKETS)],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )


class LabelCodec:
    """Thin, serialisable wrapper around :class:`~sklearn.preprocessing.LabelEncoder`.

    Keeps the class ordering stable and exposes convenient helpers used by the
    predictor and the dashboard.
    """

    def __init__(self) -> None:
        self._encoder = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, labels) -> "LabelCodec":
        values = pd.Series(labels).astype(str)
        self._encoder.fit(values)
        self.classes_ = [str(c) for c in self._encoder.classes_]
        return self

    def transform(self, labels) -> np.ndarray:
        values = pd.Series(labels).astype(str)
        unknown = set(values.unique()) - set(self.classes_)
        if unknown:
            raise PreprocessingError(
                f"Unseen class label(s) during encoding: {sorted(unknown)}",
                hint="Re-fit the label encoder or re-train the model.",
            )
        return self._encoder.transform(values)

    def fit_transform(self, labels) -> np.ndarray:
        return self.fit(labels).transform(labels)

    def inverse_transform(self, codes) -> list[str]:
        codes = np.asarray(codes).astype(int)
        return [str(c) for c in self._encoder.inverse_transform(codes)]

    def label_of(self, code: int) -> str:
        """Return the class name for a single integer code."""
        index = int(code)
        if 0 <= index < len(self.classes_):
            return self.classes_[index]
        return str(code)

    def code_of(self, label: str) -> int:
        """Return the integer code for a class name, or ``-1`` if unknown."""
        try:
            return int(self.classes_.index(str(label)))
        except ValueError:
            return -1

    @property
    def n_classes(self) -> int:
        return len(self.classes_)

    def save(self, path: str | Path) -> Path:
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @staticmethod
    def load(path: str | Path) -> "LabelCodec":
        import joblib

        target = Path(path)
        if not target.exists():
            raise PreprocessingError(f"Label encoder not found: {target}")
        obj = joblib.load(target)
        if not isinstance(obj, LabelCodec):
            raise PreprocessingError(f"{target} does not contain a LabelCodec.")
        return obj

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<LabelCodec classes={self.classes_}>"


__all__ = [
    "PortBucketTransformer",
    "build_protocol_encoder",
    "build_port_encoder",
    "LabelCodec",
]
