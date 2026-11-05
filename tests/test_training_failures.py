"""A model that fails during training must retry, then fail visibly."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from src.training.train import _fit_with_retry

X = np.zeros((4, 2))
y = np.array([0, 1, 0, 1])


class FlakyModel(BaseEstimator):
    """Fails while parallel, or always when ``always_fail`` is set."""

    def __init__(self, n_jobs=-1, always_fail=False):
        self.n_jobs = n_jobs
        self.always_fail = always_fail

    def fit(self, X, y, **kwargs):
        self.calls_ = getattr(self, "calls_", []) + [(self.n_jobs, kwargs)]
        if self.always_fail or self.n_jobs != 1:
            raise ValueError("concurrent send_bytes() calls are not supported")
        return self


def test_failed_parallel_fit_retries_on_one_core():
    model = FlakyModel()
    _fit_with_retry(model, X, y, name="flaky", sample_weight=[1, 1, 1, 1])
    assert [jobs for jobs, _ in model.calls_] == [-1, 1]
    assert model.calls_[1][1] == {"sample_weight": [1, 1, 1, 1]}


def test_persistent_failure_is_raised_not_hidden():
    with pytest.raises(ValueError):
        _fit_with_retry(FlakyModel(always_fail=True), X, y, name="flaky")


def test_single_core_failure_does_not_retry():
    model = FlakyModel(n_jobs=1, always_fail=True)
    with pytest.raises(ValueError):
        _fit_with_retry(model, X, y, name="flaky")
    assert len(model.calls_) == 1
