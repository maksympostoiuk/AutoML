from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_friedman1

import automl.engine as engine
from automl.engine import AutoML, load_model
from automl.task import TaskSpec, TaskSpecError

FAST = {"time_budget": 5, "n_trials": 5, "n_jobs": 1, "validation": "holdout"}


@pytest.fixture
def fitted(churn_df: pd.DataFrame) -> AutoML:
    return AutoML(**FAST).fit(churn_df, target="churned")


def test_leaderboard(fitted: AutoML) -> None:
    board = fitted.leaderboard_
    assert set(board["model"]) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "xgboost_default",
        "xgboost",
    }
    assert board["selected"].sum() == 1
    assert list(board.columns) == [
        "model",
        "valid_roc_auc",
        "test_roc_auc",
        "fit_seconds",
        "selected",
    ]


def test_selection_uses_validation_not_test(fitted: AutoML) -> None:
    board = fitted.leaderboard_
    chosen = board.loc[board["selected"], "model"].item()
    assert chosen == board.loc[board["valid_roc_auc"].idxmax(), "model"]
    assert fitted.model_.model_name == chosen


def test_test_rows_never_reach_the_search(
    churn_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, set] = {}
    original = engine.search_dummy

    def spy(data):
        seen["rows"] = set().union(
            *(set(f.X_train_trees.index) | set(f.X_valid_trees.index) for f in data.folds)
        )
        return original(data)

    monkeypatch.setattr(engine, "search_dummy", spy)
    automl = AutoML(**FAST).fit(churn_df, target="churned")
    sizes = automl.split_sizes_
    assert len(seen["rows"]) == sizes["train"] + sizes["valid"]
    assert len(seen["rows"]) + sizes["test"] == len(churn_df)


def test_save_load_predict(fitted: AutoML, churn_df: pd.DataFrame, tmp_path: Path) -> None:
    path = fitted.save(tmp_path / "model.joblib")
    model = load_model(path)
    new = churn_df.drop(columns="churned").head(20)  # no target at prediction time
    np.testing.assert_array_equal(model.predict(new), fitted.model_.predict(new))
    assert set(model.predict(new)) <= {"yes", "no"}  # original labels, not 0/1
    proba = model.predict_proba(new)
    assert list(proba.columns) == ["no", "yes"]
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert model.metadata["leaderboard"]


def test_missing_columns_are_reported(fitted: AutoML, churn_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="tenure_months"):
        fitted.model_.predict(churn_df.drop(columns="tenure_months"))


def test_multiclass_with_string_labels() -> None:
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=4, n_classes=3, random_state=0
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["species"] = np.array(["setosa", "versicolor", "virginica"])[y]
    automl = AutoML(**FAST).fit(df, target="species")
    assert automl.spec_.problem_type == "multiclass"
    board = automl.leaderboard_.set_index("model")
    assert board.loc["xgboost", "test_log_loss"] < board.loc["dummy", "test_log_loss"]
    assert set(automl.model_.predict(df.head(30))) <= {"setosa", "versicolor", "virginica"}


def test_regression() -> None:
    X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["price"] = y * 1000
    automl = AutoML(**FAST).fit(df, target="price")
    board = automl.leaderboard_.set_index("model")
    assert board.loc["xgboost", "test_rmse"] < 0.6 * board.loc["dummy", "test_rmse"]
    with pytest.raises(ValueError, match="classification"):
        automl.model_.predict_proba(df)


def test_a_given_spec_is_used_and_checked(churn_df: pd.DataFrame) -> None:
    spec = TaskSpec(
        target="churned",
        problem_type="binary",
        metric="log_loss",
        drop_columns={"customer_id": "id", "cancellation_date": "leak"},
    )
    automl = AutoML(spec=spec, **FAST).fit(churn_df)
    assert "valid_log_loss" in automl.leaderboard_.columns
    assert "contract" in automl.model_.feature_columns

    wrong = TaskSpec(target="churned", problem_type="regression", metric="rmse")
    with pytest.raises(TaskSpecError):
        AutoML(spec=wrong, **FAST).fit(churn_df)


def test_small_data_uses_cross_validation_and_the_one_se_rule(churn_df: pd.DataFrame) -> None:
    settings = {**FAST, "validation": "auto"}
    automl = AutoML(**settings).fit(churn_df, target="churned")  # 320 development rows
    assert automl.validation_ == "5-fold CV"
    assert automl.selection_ == "one-standard-error rule"
    assert automl.model_.metadata["validation"] == "5-fold CV"


def test_one_se_rule_prefers_the_simpler_model() -> None:
    from automl.models import Candidate

    automl = AutoML(selection="one_se")

    def candidate(name: str, std: float) -> Candidate:
        return Candidate(name, None, False, 0.0, 0.0, {"valid_std": std})

    candidates = [candidate("dummy", 0), candidate("linear", 0.02), candidate("xgboost", 0.05)]
    # xgboost is best by 0.01, but one standard error is 0.05 / sqrt(5) = 0.022.
    assert automl._choose(candidates, [0.5, 0.80, 0.81], use_cv=True) == 1
    # A clear win (0.10 > 0.022) keeps xgboost.
    assert automl._choose(candidates, [0.5, 0.71, 0.81], use_cv=True) == 2
    # "best" ignores the noise.
    assert AutoML(selection="best")._choose(candidates, [0.5, 0.80, 0.81], use_cv=True) == 2


def test_holdout_can_be_forced(churn_df: pd.DataFrame) -> None:
    automl = AutoML(**FAST).fit(churn_df, target="churned")
    assert automl.validation_ == "holdout"
    assert automl.selection_ == "best validation score"


def test_invalid_validation_settings() -> None:
    with pytest.raises(ValueError):
        AutoML(validation="bootstrap")
    with pytest.raises(ValueError, match="cross-validation"):
        AutoML(validation="holdout", selection="one_se")
