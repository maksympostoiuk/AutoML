"""Compare the from-scratch gradient boosting with scikit-learn and XGBoost.

Run: uv run python scripts/compare_boosting.py

Same hyperparameters for all three, test-set quality and fit time. Uses datasets bundled with
scikit-learn plus one larger synthetic set, so it needs no network.
"""

import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from automl.boosting import ScratchGradientBoostingClassifier, ScratchGradientBoostingRegressor

N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH = 200, 0.05, 3
COMMON = {"n_estimators": N_ESTIMATORS, "learning_rate": LEARNING_RATE, "max_depth": MAX_DEPTH}


def timed_fit(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def regression(name: str, X: np.ndarray, y: np.ndarray) -> None:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    models = {
        "scratch": ScratchGradientBoostingRegressor(**COMMON, random_state=0),
        "sklearn": GradientBoostingRegressor(**COMMON, random_state=0),
        # XGBoost's own regularisation is switched off to make the comparison fair.
        "xgboost": xgb.XGBRegressor(**COMMON, reg_lambda=0.0, random_state=0),
    }
    print(f"\n{name}: regression, {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  {'model':<8} {'test RMSE':>10} {'fit, s':>8}")
    for label, model in models.items():
        model, seconds = timed_fit(model, X_tr, y_tr)
        rmse = mean_squared_error(y_te, model.predict(X_te)) ** 0.5
        print(f"  {label:<8} {rmse:>10.3f} {seconds:>8.2f}")
    print(f"  baseline (predict the mean): {np.std(y_te):.3f}")


def classification(name: str, X: np.ndarray, y: np.ndarray) -> None:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    models = {
        "scratch": ScratchGradientBoostingClassifier(**COMMON, random_state=0),
        "sklearn": GradientBoostingClassifier(**COMMON, random_state=0),
        "xgboost": xgb.XGBClassifier(**COMMON, reg_lambda=0.0, random_state=0),
    }
    print(f"\n{name}: binary classification, {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  {'model':<8} {'log-loss':>10} {'ROC AUC':>8} {'fit, s':>8}")
    for label, model in models.items():
        model, seconds = timed_fit(model, X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        print(
            f"  {label:<8} {log_loss(y_te, proba):>10.4f} "
            f"{roc_auc_score(y_te, proba):>8.4f} {seconds:>8.2f}"
        )


def learning_curve() -> None:
    """How the test error changes as trees are added, for two learning rates."""
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    checkpoints = [1, 10, 50, 100, 200, 500]
    print("\nDiabetes: test RMSE after N trees (scratch implementation)")
    print(f"  {'learning_rate':<14}" + "".join(f"{n:>8}" for n in checkpoints))
    for lr in (0.5, 0.05):
        model = ScratchGradientBoostingRegressor(n_estimators=max(checkpoints), learning_rate=lr)
        model.fit(X_tr, y_tr)
        rmse = [mean_squared_error(y_te, pred) ** 0.5 for pred in model.staged_predict(X_te)]
        print(f"  {lr:<14}" + "".join(f"{rmse[n - 1]:>8.1f}" for n in checkpoints))


def main() -> None:
    regression("Diabetes", *load_diabetes(return_X_y=True))
    classification("Breast cancer", *load_breast_cancer(return_X_y=True))
    classification(
        "Synthetic",
        *make_classification(n_samples=20_000, n_features=20, n_informative=10, random_state=0),
    )
    learning_curve()


if __name__ == "__main__":
    main()
