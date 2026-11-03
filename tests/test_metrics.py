import numpy as np
import pytest

from automl.metrics import METRICS, get_metric
from automl.task import METRICS as TASK_METRICS


def test_every_task_metric_is_implemented() -> None:
    for names in TASK_METRICS.values():
        for name in names:
            assert name in METRICS


def test_objective_is_higher_for_the_better_model() -> None:
    # The bug in the original project: optimising a loss in the wrong direction.
    y = np.array([1.0, 2.0, 3.0])
    good, bad = np.array([1.1, 2.0, 2.9]), np.array([3.0, 0.0, 5.0])
    for name in ("rmse", "mae", "r2"):
        metric = get_metric(name)
        assert metric.objective(y, good) > metric.objective(y, bad), name


def test_losses_are_reported_in_natural_units() -> None:
    rmse = get_metric("rmse")
    y, prediction = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert rmse.score(y, prediction) == pytest.approx(np.sqrt(12.5))
    assert rmse.objective(y, prediction) == pytest.approx(-np.sqrt(12.5))


def test_classification_metrics_take_probabilities() -> None:
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    assert get_metric("accuracy").score(y, proba) == pytest.approx(0.75)
    assert get_metric("roc_auc").score(y, proba) == pytest.approx(0.75)  # 3 of 4 pairs ordered
    worse = proba[:, ::-1]
    log_loss = get_metric("log_loss")
    assert log_loss.objective(y, proba) > log_loss.objective(y, worse)


def test_multiclass_log_loss_handles_a_class_missing_from_y() -> None:
    y = np.array([0, 1, 0])  # class 2 does not appear in this sample
    proba = np.full((3, 3), 1 / 3)
    assert get_metric("log_loss").score(y, proba) == pytest.approx(np.log(3))


def test_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("auc")
