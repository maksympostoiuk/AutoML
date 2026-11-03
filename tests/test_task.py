import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from automl.data import find_leakage_suspects, find_missingness_warnings, profile_dataset
from automl.task import ProblemType, TaskSpec, TaskSpecError, infer_problem_type, infer_task_spec


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["yes", "no", "yes"], ProblemType.BINARY),
        ([True, False, True], ProblemType.BINARY),
        ([0, 1, 1, 0], ProblemType.BINARY),
        (["cat", "dog", "bird", "cat"], ProblemType.MULTICLASS),
        ([1, 2, 3, 4, 5, 3, 2], ProblemType.MULTICLASS),
        (np.linspace(0, 1, 100), ProblemType.REGRESSION),
        # The bug in the original project: a float target with few values is still regression.
        ([1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 4.5], ProblemType.REGRESSION),
        (np.arange(100), ProblemType.REGRESSION),
    ],
)
def test_infer_problem_type(values: list, expected: ProblemType) -> None:
    problem_type, reason = infer_problem_type(pd.Series(values, name="y"))
    assert problem_type == expected
    assert reason


def test_constant_target_is_rejected() -> None:
    with pytest.raises(TaskSpecError, match="fewer than 2"):
        infer_problem_type(pd.Series([1, 1, 1], name="y"))


def test_infer_task_spec_on_churn_table(churn_df: pd.DataFrame) -> None:
    spec = infer_task_spec(churn_df, target="churned")
    assert spec.problem_type == ProblemType.BINARY
    assert spec.metric == "roc_auc"
    assert set(spec.drop_columns) == {
        "customer_id",
        "support_notes",
        "country",
        "cancellation_date",
    }
    assert "leakage" in spec.drop_columns["cancellation_date"]
    assert any("cancellation_date" in note for note in spec.notes)
    spec.check_against(churn_df)


def test_target_is_guessed_when_not_given(churn_df: pd.DataFrame) -> None:
    spec = infer_task_spec(churn_df.rename(columns={"churned": "target"}).iloc[:, ::-1])
    assert spec.target == "target"  # found by name even though it is no longer last
    assert any("guessed" in note for note in spec.notes)


def test_filled_in_flag_is_caught_as_leakage(churn_df: pd.DataFrame) -> None:
    df = churn_df.drop(columns="cancellation_date")
    df["cancelled"] = np.where(churn_df["churned"] == "yes", "Y", None)
    spec = infer_task_spec(df, target="churned")
    assert "leakage" in spec.drop_columns["cancelled"]


def test_real_features_are_not_flagged(churn_df: pd.DataFrame) -> None:
    df = churn_df.drop(columns="cancellation_date")
    suspects = find_leakage_suspects(df, "churned", "binary", profile_dataset(df))
    assert suspects == []


def test_regression_leakage() -> None:
    rng = np.random.default_rng(0)
    price = rng.normal(100, 20, 500)
    df = pd.DataFrame(
        {
            "rooms": rng.integers(1, 6, 500),
            "price_with_vat": price * 1.21,  # the target in disguise
            "price": price,
        }
    )
    spec = infer_task_spec(df, target="price")
    assert spec.problem_type == ProblemType.REGRESSION
    assert list(spec.drop_columns) == ["price_with_vat"]


def test_spec_rejects_a_metric_of_another_problem_type() -> None:
    with pytest.raises(ValidationError, match="does not fit"):
        TaskSpec(target="y", problem_type="regression", metric="roc_auc")


def test_spec_rejects_dropping_the_target() -> None:
    with pytest.raises(ValidationError, match="target"):
        TaskSpec(target="y", problem_type="binary", metric="f1", drop_columns={"y": "oops"})


def test_check_against_lists_every_problem(churn_df: pd.DataFrame) -> None:
    spec = TaskSpec(
        target="contract",  # three classes, but declared binary
        problem_type="binary",
        metric="roc_auc",
        drop_columns={"no_such_column": "typo"},
    )
    with pytest.raises(TaskSpecError) as error:
        spec.check_against(churn_df)
    assert len(error.value.problems) == 2


def test_check_against_regression_needs_numbers(churn_df: pd.DataFrame) -> None:
    spec = TaskSpec(target="churned", problem_type="regression", metric="rmse")
    with pytest.raises(TaskSpecError, match="numeric"):
        spec.check_against(churn_df)


def test_spec_round_trips_through_json(churn_df: pd.DataFrame) -> None:
    spec = infer_task_spec(churn_df, target="churned")
    assert TaskSpec.model_validate_json(spec.model_dump_json()) == spec


def test_filled_only_for_one_class_gives_a_warning_not_a_drop() -> None:
    # Like the Titanic "body" column: an ID that exists only for passengers who died.
    rng = np.random.default_rng(0)
    n = 1000
    survived = rng.random(n) < 0.4
    body = np.where(~survived & (rng.random(n) < 0.15), rng.integers(1, 300, n), np.nan)
    df = pd.DataFrame(
        {"age": rng.normal(30, 10, n), "body": body, "survived": survived.astype(int)}
    )
    spec = infer_task_spec(df, target="survived")
    assert "body" not in spec.drop_columns
    warning = next(note for note in spec.notes if "'body'" in note)
    assert "filled in" in warning and "'0' in 100%" in warning


def test_random_gaps_give_no_warning(churn_df: pd.DataFrame) -> None:
    df = churn_df.drop(columns="cancellation_date")  # the real leak in this table
    assert find_missingness_warnings(df, "churned", "binary", profile_dataset(df)) == []
