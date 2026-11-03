"""Candidate models and how each one is tuned on the validation set.

Every ``search_*`` function trains on the train split, scores on the validation split and
returns a ``Candidate``: an *unfitted* estimator with the chosen settings, plus its validation
score. The engine later refits the chosen settings on train + validation and scores the test
split once. The test split never reaches this module.

Candidates, from simplest to strongest:

* dummy: predicts the class balance / the mean. Anything worse than this is broken.
* linear: logistic or ridge regression, regularisation strength picked on validation.
* random_forest: a strong default that needs no tuning.
* xgboost_default: XGBoost with library defaults and early stopping, no tuning.
* xgboost: XGBoost tuned by Optuna.

The two XGBoost lines together show how much tuning actually adds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from automl.metrics import Metric, model_output

MAX_TREES = 2_000  # upper bound; early stopping decides the real number
EARLY_STOPPING_ROUNDS = 50
EVAL_METRIC = {"binary": "logloss", "multiclass": "mlogloss", "regression": "rmse"}


@dataclass
class SearchData:
    """Everything a search needs: preprocessed train/valid matrices and the task."""

    problem_type: str
    metric: Metric
    X_train_trees: pd.DataFrame
    X_valid_trees: pd.DataFrame
    X_train_linear: pd.DataFrame
    X_valid_linear: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    random_state: int = 0
    n_jobs: int = -1

    @property
    def is_classification(self) -> bool:
        return self.problem_type != "regression"

    def valid_score(self, model: BaseEstimator, linear: bool = False) -> float:
        X = self.X_valid_linear if linear else self.X_valid_trees
        return self.metric.score(self.y_valid, model_output(model, X, self.is_classification))


@dataclass
class Candidate:
    name: str
    estimator: BaseEstimator  # unfitted, with the settings chosen on validation
    linear_features: bool  # which preprocessing flavour it needs
    valid_score: float  # natural units of the metric
    fit_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


# ----- simple candidates ----------------------------------------------------------------------


def search_dummy(data: SearchData) -> Candidate:
    start = time.perf_counter()
    model = DummyClassifier(strategy="prior") if data.is_classification else DummyRegressor()
    model.fit(data.X_train_trees, data.y_train)
    return Candidate("dummy", model, False, data.valid_score(model), time.perf_counter() - start)


def search_linear(data: SearchData) -> Candidate:
    """Try a few regularisation strengths, keep the best on validation."""
    start = time.perf_counter()
    best: tuple[float, BaseEstimator, float] | None = None
    for strength in (0.01, 0.1, 1.0, 10.0, 100.0):
        if data.is_classification:
            model = LogisticRegression(C=1 / strength, max_iter=2_000)
        else:
            model = Ridge(alpha=strength)
        model.fit(data.X_train_linear, data.y_train)
        objective = data.metric.objective(
            data.y_valid, model_output(model, data.X_valid_linear, data.is_classification)
        )
        if best is None or objective > best[0]:
            best = (objective, model, strength)
    assert best is not None
    _, model, strength = best
    name = "logistic_regression" if data.is_classification else "ridge"
    return Candidate(
        name,
        clone(model),
        True,
        data.valid_score(model, linear=True),
        time.perf_counter() - start,
        {"regularisation": strength},
    )


def search_random_forest(data: SearchData) -> Candidate:
    start = time.perf_counter()
    params = {"n_estimators": 300, "random_state": data.random_state, "n_jobs": data.n_jobs}
    model = (
        RandomForestClassifier(**params)
        if data.is_classification
        else RandomForestRegressor(**params)
    )
    model.fit(data.X_train_trees, data.y_train)
    return Candidate(
        "random_forest", clone(model), False, data.valid_score(model), time.perf_counter() - start
    )


# ----- XGBoost --------------------------------------------------------------------------------


def suggest_xgboost_params(trial: optuna.Trial, n_train_rows: int) -> dict[str, Any]:
    """The Optuna search space.

    Every log-scale range starts above zero: ``log=True`` with ``low=0`` is invalid, and in
    the original project that silently disabled XGBoost altogether.

    ``gamma`` is the minimum gain of a split, and gain is a sum over the rows in a node
    (stage 1c), so a fixed range cannot fit every dataset. The search works on
    ``gamma_share`` instead: gamma as a fraction of the number of training rows. With
    standardised regression targets this puts classification and regression on the same scale.
    """
    gamma_share = trial.suggest_float("gamma_share", 1e-6, 1e-2, log=True)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "gamma": gamma_share * n_train_rows,
    }


@dataclass
class _ScaledTarget:
    """Regression targets are standardised for XGBoost, so one search space fits all scales."""

    mean: float
    std: float

    @classmethod
    def fit(cls, y: np.ndarray) -> _ScaledTarget:
        std = float(np.std(y))
        return cls(float(np.mean(y)), std if std > 0 else 1.0)

    def forward(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


def _fit_xgboost_with_early_stopping(data: SearchData, params: dict[str, Any]) -> tuple[float, int]:
    """Fit on train, stop on validation. Returns (validation objective, number of trees)."""
    common = {
        "n_estimators": MAX_TREES,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "eval_metric": EVAL_METRIC[data.problem_type],
        "tree_method": "hist",
        "random_state": data.random_state,
        "n_jobs": data.n_jobs,
        **params,
    }
    if data.is_classification:
        model = xgb.XGBClassifier(**common)
        model.fit(
            data.X_train_trees,
            data.y_train,
            eval_set=[(data.X_valid_trees, data.y_valid)],
            verbose=False,
        )
        prediction = model.predict_proba(data.X_valid_trees)
    else:
        scaler = _ScaledTarget.fit(data.y_train)
        model = xgb.XGBRegressor(**common)
        model.fit(
            data.X_train_trees,
            scaler.forward(data.y_train),
            eval_set=[(data.X_valid_trees, scaler.forward(data.y_valid))],
            verbose=False,
        )
        prediction = scaler.inverse(model.predict(data.X_valid_trees))
    return data.metric.objective(data.y_valid, prediction), int(model.best_iteration) + 1


def make_xgboost(
    problem_type: str, params: dict[str, Any], n_estimators: int, data: SearchData
) -> BaseEstimator:
    """The final, unfitted estimator: a fixed number of trees, no early stopping."""
    common = {
        "n_estimators": n_estimators,
        "tree_method": "hist",
        "random_state": data.random_state,
        "n_jobs": data.n_jobs,
        **params,
    }
    if problem_type == "regression":
        return TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(**common), transformer=StandardScaler()
        )
    return xgb.XGBClassifier(**common)


def _final_tree_count(n_trees: int, data: SearchData) -> int:
    # The refit sees train + validation rows; scale the number of trees with the data size.
    n_train, n_valid = len(data.y_train), len(data.y_valid)
    return max(1, math.ceil(n_trees * (n_train + n_valid) / n_train))


def search_xgboost_default(data: SearchData) -> Candidate:
    start = time.perf_counter()
    objective, n_trees = _fit_xgboost_with_early_stopping(data, {})
    return Candidate(
        "xgboost_default",
        make_xgboost(data.problem_type, {}, _final_tree_count(n_trees, data), data),
        False,
        _natural(objective, data.metric),
        time.perf_counter() - start,
        {"n_trees_early_stopping": n_trees},
    )


def search_xgboost(data: SearchData, n_trials: int, timeout: float | None) -> Candidate:
    """Tune XGBoost with Optuna (TPE). Each trial uses early stopping on validation."""
    start = time.perf_counter()
    n_train = len(data.y_train)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgboost_params(trial, n_train)
        value, n_trees = _fit_xgboost_with_early_stopping(data, params)
        trial.set_user_attr("n_trees", n_trees)
        trial.set_user_attr("params", params)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",  # always: Metric.objective already flips losses
        sampler=optuna.samplers.TPESampler(seed=data.random_state),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    params = best.user_attrs["params"]
    n_trees = best.user_attrs["n_trees"]
    return Candidate(
        "xgboost",
        make_xgboost(data.problem_type, params, _final_tree_count(n_trees, data), data),
        False,
        _natural(best.value, data.metric),
        time.perf_counter() - start,
        {
            "n_trials": len(study.trials),
            "best_trial": best.number,
            "n_trees_early_stopping": n_trees,
            "params": params,
            "history": [t.value for t in study.trials if t.value is not None],
        },
    )


def _natural(objective: float, metric: Metric) -> float:
    return objective if metric.greater_is_better else -objective
