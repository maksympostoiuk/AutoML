import pickle

import numpy as np
import pandas as pd
import pytest

from automl.data import profile_dataset
from automl.preprocess import DatetimeFeatures, make_preprocessor, plan_features
from automl.task import infer_task_spec


@pytest.fixture
def plan_and_data(churn_df: pd.DataFrame):
    profile = profile_dataset(churn_df)
    spec = infer_task_spec(churn_df, target="churned", profile=profile)
    plan = plan_features(profile, spec)
    return plan, churn_df[plan.columns]


def test_plan_routes_every_column(plan_and_data) -> None:
    plan, _ = plan_and_data
    assert plan.numeric == ["tenure_months", "monthly_charges", "has_partner"]
    assert plan.categorical == ["contract", "payment_method"]
    assert plan.datetime == ["signup_date"]
    # Nothing disappears silently: every other column has a reason.
    assert set(plan.ignored) == {"customer_id", "support_notes", "country", "cancellation_date"}


def test_tree_preprocessing_keeps_missing_numbers(plan_and_data) -> None:
    plan, X = plan_and_data
    out = make_preprocessor(plan).fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out["monthly_charges"].isna().sum() == 30  # left for XGBoost to handle
    assert "payment_method___missing__" in out.columns  # missing category is its own level
    assert {"signup_date_year", "signup_date_month", "signup_date_day_of_week"} <= set(out.columns)
    np.testing.assert_array_equal(out["tenure_months"], X["tenure_months"])  # not scaled


def test_linear_preprocessing_imputes_and_scales(plan_and_data) -> None:
    plan, X = plan_and_data
    out = make_preprocessor(plan, for_linear=True).fit_transform(X)
    assert out.isna().sum().sum() == 0
    assert "missingindicator_monthly_charges" in out.columns
    assert out["tenure_months"].mean() == pytest.approx(0, abs=1e-9)


def test_unseen_and_rare_categories_do_not_break_predict(plan_and_data) -> None:
    plan, X = plan_and_data
    preprocessor = make_preprocessor(plan).fit(X)
    new = X.head(3).copy()
    new["contract"] = ["three years", "one year", None]
    out = preprocessor.transform(new)
    assert out.shape[0] == 3
    assert out.filter(like="contract_").iloc[0].sum() == 0  # unknown level: all zeros


def test_many_categories_are_capped() -> None:
    from automl.preprocess import FeaturePlan

    X = pd.DataFrame({"city": [f"city_{i % 60}" for i in range(600)]})
    out = make_preprocessor(FeaturePlan(categorical=["city"])).fit_transform(X)
    assert out.shape[1] <= 20


def test_datetime_features_add_hour_only_when_times_exist() -> None:
    dates = pd.DataFrame({"d": ["2024-03-01", "2024-03-02"]})
    times = pd.DataFrame({"t": ["2024-03-01 10:30", "2024-03-02 18:00"]})
    assert list(DatetimeFeatures().fit(dates).get_feature_names_out()) == [
        "d_year",
        "d_month",
        "d_day",
        "d_day_of_week",
    ]
    out = DatetimeFeatures().fit_transform(times)
    assert list(out["t_hour"]) == [10.0, 18.0]


def test_preprocessor_can_be_pickled(plan_and_data) -> None:
    plan, X = plan_and_data
    preprocessor = make_preprocessor(plan).fit(X)
    restored = pickle.loads(pickle.dumps(preprocessor))
    pd.testing.assert_frame_equal(restored.transform(X), preprocessor.transform(X))
