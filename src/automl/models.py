"""Candidate models and how each one is tuned on validation data.

Validation comes as a list of folds (``Fold``): one fold is a plain train/validation holdout,
several folds are k-fold cross-validation over the development rows. Every candidate is scored
as the mean over the folds, so the code is the same either way. Cross-validation matters on
small data: a validation set of a few hundred rows is noisy, and a hyperparameter search
tuned on one such set learns its noise (stage 3 showed valid 0.87 vs. test 0.63).

Every ``search_*`` function returns a ``Candidate``: an *unfitted* estimator with the chosen
settings, plus its validation score. The engine later refits it on all development rows
(train + validation) and scores the test split once. The test split never reaches this module.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
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
class Fold:
    """One train/validation pair, preprocessed both ways (for trees and for linear models)."""

    X_train_trees: pd.DataFrame
    X_valid_trees: pd.DataFrame
    X_train_linear: pd.DataFrame
    X_valid_linear: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray


@dataclass
class SearchData:
    """Everything a search needs: the task, the metric and the validation folds."""

    problem_type: str
    metric: Metric
    folds: list[Fold]
    n_development_rows: int  # rows the final model is refitted on (train + validation)
    random_state: int = 0
    n_jobs: int = -1

    @property
    def is_classification(self) -> bool:
        return self.problem_type != "regression"

    @property
    def mean_fold_train_rows(self) -> float:
        return float(np.mean([len(f.y_train) for f in self.folds]))

    def cross_validate(
        self, fit_predict: Callable[[Fold], np.ndarray]
    ) -> tuple[float, list[float]]:
        """Mean objective over folds (higher is better) and the per-fold objectives."""
        scores = [self.metric.objective(f.y_valid, fit_predict(f)) for f in self.folds]
        return float(np.mean(scores)), scores

    def evaluate(
        self, make_model: Callable[[], BaseEstimator], linear: bool
    ) -> tuple[float, list[float]]:
        """Fit a fresh model on every fold's train part and score its validation part."""

        def fit_predict(fold: Fold) -> np.ndarray:
            X_train = fold.X_train_linear if linear else fold.X_train_trees
            X_valid = fold.X_valid_linear if linear else fold.X_valid_trees
            model = make_model().fit(X_train, fold.y_train)
            return model_output(model, X_valid, self.is_classification)

        return self.cross_validate(fit_predict)


@dataclass
class Candidate:
    name: str
    estimator: BaseEstimator  # unfitted, with the settings chosen on validation
    linear_features: bool  # which preprocessing flavour it needs
    valid_score: float  # mean over folds, natural units of the metric
    fit_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


def _natural(objective: float, metric: Metric) -> float:
    return objective if metric.greater_is_better else -objective


def _candidate(
    name: str,
    estimator: BaseEstimator,
    linear: bool,
    objective: tuple[float, list[float]],
    data: SearchData,
    started: float,
    **details: Any,
) -> Candidate:
    mean, per_fold = objective
    if len(per_fold) > 1:
        details["valid_std"] = float(np.std(per_fold))
    return Candidate(
        name,
        estimator,
        linear,
        _natural(mean, data.metric),
        time.perf_counter() - started,
        details,
    )


# ----- simple candidates ----------------------------------------------------------------------


def search_dummy(data: SearchData) -> Candidate:
    start = time.perf_counter()

    def make() -> BaseEstimator:
        return DummyClassifier(strategy="prior") if data.is_classification else DummyRegressor()

    return _candidate("dummy", make(), False, data.evaluate(make, linear=False), data, start)


def search_linear(data: SearchData) -> Candidate:
    """Try a few regularisation strengths, keep the best on validation."""
    start = time.perf_counter()

    def make(strength: float) -> BaseEstimator:
        if data.is_classification:
            return LogisticRegression(C=1 / strength, max_iter=2_000)
        return Ridge(alpha=strength)

    results = {s: data.evaluate(lambda s=s: make(s), linear=True) for s in (0.01, 0.1, 1, 10, 100)}
    strength = max(results, key=lambda s: results[s][0])
    name = "logistic_regression" if data.is_classification else "ridge"
    return _candidate(
        name, make(strength), True, results[strength], data, start, regularisation=strength
    )


def search_random_forest(data: SearchData) -> Candidate:
    start = time.perf_counter()
    params = {"n_estimators": 300, "random_state": data.random_state, "n_jobs": data.n_jobs}

    def make() -> BaseEstimator:
        if data.is_classification:
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    return _candidate(
        "random_forest", make(), False, data.evaluate(make, linear=False), data, start
    )


# ----- XGBoost --------------------------------------------------------------------------------


def suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    """The Optuna search space.

    Every log-scale range starts above zero: ``log=True`` with ``low=0`` is invalid, and in
    the original project that silently disabled XGBoost altogether.

    ``gamma`` is the minimum gain of a split, and gain is a sum over the rows in a node
    (stage 1c), so a fixed range cannot fit every dataset. The search works on
    ``gamma_share`` instead: gamma as a fraction of the number of training rows (see
    ``xgboost_params``). With standardised regression targets this puts classification and
    regression on the same scale.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "gamma_share": trial.suggest_float("gamma_share", 1e-6, 1e-2, log=True),
    }


def xgboost_params(search_params: dict[str, Any], n_rows: float) -> dict[str, Any]:
    """Search parameters -> XGBoost parameters for a model trained on ``n_rows`` rows."""
    params = dict(search_params)
    if "gamma_share" in params:
        params["gamma"] = params.pop("gamma_share") * n_rows
    return params


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


def _fit_xgboost_with_early_stopping(
    data: SearchData, fold: Fold, search_params: dict[str, Any]
) -> tuple[np.ndarray, int]:
    """Fit on the fold's train part, stop on its validation part.

    Returns (validation prediction, number of trees kept).
    """
    common = {
        "n_estimators": MAX_TREES,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "eval_metric": EVAL_METRIC[data.problem_type],
        "tree_method": "hist",
        "random_state": data.random_state,
        "n_jobs": data.n_jobs,
        **xgboost_params(search_params, len(fold.y_train)),
    }
    if data.is_classification:
        model = xgb.XGBClassifier(**common)
        model.fit(
            fold.X_train_trees,
            fold.y_train,
            eval_set=[(fold.X_valid_trees, fold.y_valid)],
            verbose=False,
        )
        prediction = model.predict_proba(fold.X_valid_trees)
    else:
        scaler = _ScaledTarget.fit(fold.y_train)
        model = xgb.XGBRegressor(**common)
        model.fit(
            fold.X_train_trees,
            scaler.forward(fold.y_train),
            eval_set=[(fold.X_valid_trees, scaler.forward(fold.y_valid))],
            verbose=False,
        )
        prediction = scaler.inverse(model.predict(fold.X_valid_trees))
    return prediction, int(model.best_iteration) + 1


def evaluate_xgboost(
    data: SearchData, search_params: dict[str, Any]
) -> tuple[tuple[float, list[float]], int]:
    """Cross-validated objective of one parameter set, and the mean number of trees."""
    n_trees: list[int] = []

    def fit_predict(fold: Fold) -> np.ndarray:
        prediction, trees = _fit_xgboost_with_early_stopping(data, fold, search_params)
        n_trees.append(trees)
        return prediction

    objective = data.cross_validate(fit_predict)
    return objective, round(float(np.mean(n_trees)))


def make_xgboost(
    data: SearchData, search_params: dict[str, Any], n_trees_early_stopping: int
) -> BaseEstimator:
    """The final, unfitted estimator for all development rows: a fixed number of trees.

    The number of trees found by early stopping is scaled up with the data size, because the
    refit sees more rows than each fold's train part.
    """
    n_rows = data.n_development_rows
    n_estimators = max(1, math.ceil(n_trees_early_stopping * n_rows / data.mean_fold_train_rows))
    common = {
        "n_estimators": n_estimators,
        "tree_method": "hist",
        "random_state": data.random_state,
        "n_jobs": data.n_jobs,
        **xgboost_params(search_params, n_rows),
    }
    if data.problem_type == "regression":
        return TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(**common), transformer=StandardScaler()
        )
    return xgb.XGBClassifier(**common)


def search_xgboost_default(data: SearchData) -> Candidate:
    start = time.perf_counter()
    objective, n_trees = evaluate_xgboost(data, {})
    return _candidate(
        "xgboost_default",
        make_xgboost(data, {}, n_trees),
        False,
        objective,
        data,
        start,
        n_trees_early_stopping=n_trees,
    )


def search_xgboost(data: SearchData, n_trials: int, timeout: float | None) -> Candidate:
    """Tune XGBoost with Optuna (TPE). Every trial is cross-validated with early stopping."""
    start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgboost_params(trial)
        (mean, per_fold), n_trees = evaluate_xgboost(data, params)
        trial.set_user_attr("n_trees", n_trees)
        trial.set_user_attr("per_fold", per_fold)
        return mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",  # always: Metric.objective already flips losses
        sampler=optuna.samplers.TPESampler(seed=data.random_state),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    n_trees = best.user_attrs["n_trees"]
    return _candidate(
        "xgboost",
        make_xgboost(data, best.params, n_trees),
        False,
        (best.value, best.user_attrs["per_fold"]),
        data,
        start,
        n_trials=len(study.trials),
        best_trial=best.number,
        n_trees_early_stopping=n_trees,
        params=xgboost_params(best.params, data.n_development_rows),
        history=[t.value for t in study.trials if t.value is not None],
    )
