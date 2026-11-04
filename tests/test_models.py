import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.model_selection import KFold

from automl.metrics import get_metric
from automl.models import (
    MAX_TREES,
    Fold,
    SearchData,
    _fit_xgboost_with_early_stopping,
    evaluate_xgboost,
    make_xgboost,
    search_dummy,
    search_linear,
    search_xgboost,
    suggest_xgboost_params,
    xgboost_params,
)


def fold(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, valid: np.ndarray) -> Fold:
    return Fold(X.iloc[train], X.iloc[valid], X.iloc[train], X.iloc[valid], y[train], y[valid])


def make_data(problem_type: str, n: int = 600, n_folds: int = 1, seed: int = 0) -> SearchData:
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
    if n_folds == 1:
        cut = int(n * 0.75)
        folds = [fold(X, y, np.arange(cut), np.arange(cut, n))]
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = [fold(X, y, train, valid) for train, valid in splitter.split(X)]
    return SearchData(problem_type, get_metric(metric), folds, n_development_rows=n, n_jobs=1)


def test_search_space_has_no_invalid_log_ranges() -> None:
    trial = optuna.create_study().ask()
    suggest_xgboost_params(trial)
    for name, distribution in trial.distributions.items():
        if getattr(distribution, "log", False):
            assert distribution.low > 0, name


def test_gamma_is_scaled_by_rows() -> None:
    params = xgboost_params({"gamma_share": 0.001, "max_depth": 3}, n_rows=5_000)
    assert params == {"gamma": pytest.approx(5.0), "max_depth": 3}


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_every_corner_of_the_search_space_trains(problem_type: str) -> None:
    """Random points of the space, including the extremes, all train without errors."""
    data = make_data(problem_type, n=300)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(
        lambda t: evaluate_xgboost(data, suggest_xgboost_params(t))[0][0],
        n_trials=15,
        catch=(),  # any exception fails the test
    )
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_regression_search_minimises_rmse() -> None:
    data = make_data("regression")
    candidate = search_xgboost(data, n_trials=8, timeout=None)
    rmses = [-value for value in candidate.details["history"]]
    assert candidate.valid_score == pytest.approx(min(rmses))
    assert candidate.valid_score < np.std(data.folds[0].y_valid)  # better than the mean


def test_early_stopping_stops() -> None:
    data = make_data("binary")
    _, n_trees = _fit_xgboost_with_early_stopping(data, data.folds[0], {"learning_rate": 0.3})
    assert n_trees < MAX_TREES


def test_final_tree_count_grows_with_the_refit_data() -> None:
    data = make_data("binary", n=600, n_folds=5)  # each fold trains on 480 rows
    model = make_xgboost(data, {}, n_trees_early_stopping=100)
    assert model.get_params()["n_estimators"] == 125  # 100 * 600 / 480


def test_final_regression_model_predicts_in_original_units() -> None:
    data = make_data("regression")
    candidate = search_xgboost(data, n_trials=3, timeout=None)
    f = data.folds[0]
    prediction = candidate.estimator.fit(f.X_train_trees, f.y_train).predict(f.X_valid_trees)
    assert abs(prediction.mean() - f.y_valid.mean()) < np.std(f.y_valid)


def test_baselines() -> None:
    data = make_data("binary")
    assert search_dummy(data).valid_score == pytest.approx(0.5)
    linear = search_linear(data)
    assert linear.name == "logistic_regression"
    assert linear.valid_score > 0.8
    assert linear.details["regularisation"] in (0.01, 0.1, 1, 10, 100)


def test_cross_validation_averages_folds() -> None:
    data = make_data("binary", n_folds=5)
    candidate = search_linear(data)
    assert "valid_std" in candidate.details  # spread over the 5 folds
    per_fold = [
        get_metric("roc_auc").score(
            f.y_valid,
            candidate.estimator.fit(f.X_train_linear, f.y_train).predict_proba(f.X_valid_linear),
        )
        for f in data.folds
    ]
    assert candidate.valid_score == pytest.approx(np.mean(per_fold))
