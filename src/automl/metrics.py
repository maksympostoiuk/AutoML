"""Metrics with one convention for optimisation: higher is always better.

Reports show every metric in its natural units (RMSE 12.3, log-loss 0.41). The search code
never sees those: it calls ``objective()``, which negates the metrics where lower is better.
Mixing the two conventions is exactly how the original project ended up optimising its
regression models in the wrong direction (``neg_mean_squared_error`` with ``minimize``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    greater_is_better: bool
    # Classification metrics receive class probabilities (n_rows x n_classes);
    # regression metrics receive predicted values.
    func: Callable[[NDArray, NDArray], float]

    def score(self, y_true: NDArray, prediction: NDArray) -> float:
        """The metric in its natural units, for reports."""
        return float(self.func(np.asarray(y_true), np.asarray(prediction)))

    def objective(self, y_true: NDArray, prediction: NDArray) -> float:
        """The value to maximise."""
        value = self.score(y_true, prediction)
        return value if self.greater_is_better else -value


def _labels(proba: NDArray) -> NDArray:
    return proba.argmax(axis=1)


def _log_loss(y: NDArray, proba: NDArray) -> float:
    return skm.log_loss(y, proba, labels=np.arange(proba.shape[1]))


METRICS: dict[str, Metric] = {
    # Binary: the positive class is the second label in sorted order (index 1).
    "roc_auc": Metric("roc_auc", True, lambda y, p: skm.roc_auc_score(y, p[:, 1])),
    "f1": Metric("f1", True, lambda y, p: skm.f1_score(y, _labels(p))),
    # Any classification:
    "log_loss": Metric("log_loss", False, _log_loss),
    "accuracy": Metric("accuracy", True, lambda y, p: skm.accuracy_score(y, _labels(p))),
    "f1_macro": Metric("f1_macro", True, lambda y, p: skm.f1_score(y, _labels(p), average="macro")),
    # Regression:
    "rmse": Metric("rmse", False, lambda y, p: skm.root_mean_squared_error(y, p)),
    "mae": Metric("mae", False, lambda y, p: skm.mean_absolute_error(y, p)),
    "r2": Metric("r2", True, lambda y, p: skm.r2_score(y, p)),
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"Unknown metric {name!r}; known: {sorted(METRICS)}") from None


def model_output(model: object, X: object, is_classification: bool) -> NDArray:
    """What metrics consume: probabilities for classification, values for regression."""
    if is_classification:
        return np.asarray(model.predict_proba(X))  # type: ignore[attr-defined]
    return np.asarray(model.predict(X), dtype=float)  # type: ignore[attr-defined]
