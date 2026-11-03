import numpy as np
import pandas as pd
import pytest

from automl.split import split_dataset
from automl.task import TaskSpec

FEATURES = ["tenure_months", "monthly_charges", "contract"]


def test_three_way_split_sizes_and_no_overlap(churn_df: pd.DataFrame) -> None:
    spec = TaskSpec(target="churned", problem_type="binary", metric="roc_auc")
    splits = split_dataset(churn_df, spec, FEATURES, test_size=0.2, valid_size=0.2)
    assert splits.sizes() == {"train": 240, "valid": 80, "test": 80}
    indexes = [set(splits.X_train.index), set(splits.X_valid.index), set(splits.X_test.index)]
    assert not (indexes[0] & indexes[1] or indexes[0] & indexes[2] or indexes[1] & indexes[2])


def test_labels_are_encoded_and_stratified(churn_df: pd.DataFrame) -> None:
    spec = TaskSpec(target="churned", problem_type="binary", metric="roc_auc")
    splits = split_dataset(churn_df, spec, FEATURES)
    assert list(splits.classes) == ["no", "yes"]
    assert set(np.unique(splits.y_train)) == {0, 1}
    overall = (churn_df["churned"] == "yes").mean()
    for y in (splits.y_train, splits.y_valid, splits.y_test):
        assert y.mean() == pytest.approx(overall, abs=0.03)


def test_rows_without_target_are_skipped(churn_df: pd.DataFrame) -> None:
    df = churn_df.copy()
    df.loc[:9, "monthly_charges"] = np.nan
    df["spend"] = df["monthly_charges"]
    spec = TaskSpec(target="spend", problem_type="regression", metric="rmse")
    splits = split_dataset(df, spec, ["tenure_months", "contract"])
    assert splits.label_encoder is None
    assert splits.n_rows_without_target == df["spend"].isna().sum()
    assert sum(splits.sizes().values()) == df["spend"].notna().sum()


def test_split_is_reproducible(churn_df: pd.DataFrame) -> None:
    spec = TaskSpec(target="churned", problem_type="binary", metric="roc_auc")
    a = split_dataset(churn_df, spec, FEATURES, random_state=7)
    b = split_dataset(churn_df, spec, FEATURES, random_state=7)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)


def test_tiny_classes_are_reported(churn_df: pd.DataFrame) -> None:
    df = churn_df.copy()
    df.loc[0, "contract"] = "lifetime"  # a class with a single row
    spec = TaskSpec(target="contract", problem_type="multiclass", metric="log_loss")
    with pytest.raises(ValueError, match="lifetime"):
        split_dataset(df, spec, ["tenure_months"])
