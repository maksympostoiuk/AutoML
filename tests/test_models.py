import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_friedman1

from automl.metrics import get_metric
from automl.models import (
    MAX_TREES,
    SearchData,
    _fit_xgboost_with_early_stopping,
    search_dummy,
    search_linear,
    search_xgboost,
    suggest_xgboost_params,
)


def make_data(problem_type: str, n: int = 600, seed: int = 0) -> SearchData:
    if problem_type == "regression":
        X, y = make_friedman1(n_samples=n, noise=1.0, random_state=seed)
        y = y * 1000 + 50_000  # large target scale: the search space must not care
        metric = "rmse"
    else:
        n_classes = 2 if problem_type == "binary" else 3
        X, y = make_classification(
            n_samples=n, n_features=8, n_informative=5, n_classes=n_classes, random_state=seed
        )
        metric = "roc_auc" if problem_type == "binary" else "log_loss"
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    cut = int(n * 0.75)
    return SearchData(
        problem_type=problem_type,
        metric=get_metric(metric),
        X_train_trees=X.iloc[:cut],
        X_valid_trees=X.iloc[cut:],
        X_train_linear=X.iloc[:cut],
        X_valid_linear=X.iloc[cut:],
        y_train=y[:cut],
        y_valid=y[cut:],
        n_jobs=1,
    )


def test_search_space_has_no_invalid_log_ranges() -> None:
    trial = optuna.create_study().ask()
    suggest_xgboost_params(trial, n_train_rows=1000)
    for name, distribution in trial.distributions.items():
        if getattr(distribution, "log", False):
            assert distribution.low > 0, name


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_every_corner_of_the_search_space_trains(problem_type: str) -> None:
    """Random points of the space, including the extremes, all train without errors."""
    data = make_data(problem_type, n=300)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(
        lambda t: _fit_xgboost_with_early_stopping(data, suggest_xgboost_params(t, 225))[0],
        n_trials=15,
        catch=(),  # any exception fails the test
    )
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_regression_search_minimises_rmse() -> None:
    data = make_data("regression")
    candidate = search_xgboost(data, n_trials=8, timeout=None)
    rmses = [-value for value in candidate.details["history"]]
    assert candidate.valid_score == pytest.approx(min(rmses))
    assert candidate.valid_score < np.std(data.y_valid)  # better than predicting the mean


def test_early_stopping_stops(capsys: pytest.CaptureFixture[str]) -> None:
    data = make_data("binary")
    _, n_trees = _fit_xgboost_with_early_stopping(data, {"learning_rate": 0.3})
    assert n_trees < MAX_TREES


def test_final_regression_model_predicts_in_original_units() -> None:
    data = make_data("regression")
    candidate = search_xgboost(data, n_trials=3, timeout=None)
    model = candidate.estimator.fit(data.X_train_trees, data.y_train)
    prediction = model.predict(data.X_valid_trees)
    assert abs(prediction.mean() - data.y_valid.mean()) < np.std(data.y_valid)


def test_baselines() -> None:
    data = make_data("binary")
    assert search_dummy(data).valid_score == pytest.approx(0.5)
    linear = search_linear(data)
    assert linear.name == "logistic_regression"
    assert linear.valid_score > 0.8
    assert linear.details["regularisation"] in (0.01, 0.1, 1.0, 10.0, 100.0)
