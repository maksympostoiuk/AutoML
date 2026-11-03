import json

import numpy as np
import pandas as pd
import pytest

from automl.data import ColumnKind, profile_column, profile_dataset


def kind_of(values: list | pd.Series, name: str = "col") -> ColumnKind:
    return profile_column(pd.Series(values, name=name)).kind


def test_kinds_in_a_realistic_table(churn_df: pd.DataFrame) -> None:
    profile = profile_dataset(churn_df)
    kinds = {c.name: c.kind for c in profile.columns}
    assert kinds == {
        "customer_id": ColumnKind.ID_LIKE,
        "tenure_months": ColumnKind.NUMERIC,
        "monthly_charges": ColumnKind.NUMERIC,
        "contract": ColumnKind.CATEGORICAL,
        "payment_method": ColumnKind.CATEGORICAL,
        "has_partner": ColumnKind.BOOLEAN,
        "signup_date": ColumnKind.DATETIME,
        "support_notes": ColumnKind.TEXT,
        "country": ColumnKind.CONSTANT,
        "cancellation_date": ColumnKind.DATETIME,
        "churned": ColumnKind.CATEGORICAL,
    }


def test_missing_values_and_stats(churn_df: pd.DataFrame) -> None:
    column = profile_dataset(churn_df)["monthly_charges"]
    assert column.n_missing == 30
    assert column.missing_share == pytest.approx(30 / 400)
    assert set(column.stats) == {"min", "max", "mean", "std"}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.5, 2.25, np.nan, 4.0], ColumnKind.NUMERIC),
        ([0, 1, 1, 0], ColumnKind.BOOLEAN),
        ([True, False, True, True], ColumnKind.BOOLEAN),
        (["a", "b", "a", None], ColumnKind.CATEGORICAL),
        (pd.Categorical(["x", "y", "x"]), ColumnKind.CATEGORICAL),
        (pd.to_datetime(["2024-01-01", "2024-02-01", None]), ColumnKind.DATETIME),
        (["2024-01-31", "2024-02-29", "2024-03-15"], ColumnKind.DATETIME),
        (["15/01/2024 10:30", "16/01/2024 11:45", "17/01/2024 09:00"], ColumnKind.DATETIME),
        (["7", "7", "7"], ColumnKind.CONSTANT),
        ([None, None, None], ColumnKind.EMPTY),
        # One value plus gaps is a "filled in or not" flag, not a constant.
        (["yes", None, None, "yes"], ColumnKind.CATEGORICAL),
    ],
)
def test_single_column_kinds(values: list, expected: ColumnKind) -> None:
    assert kind_of(values) == expected


def test_codes_that_look_like_numbers_are_not_dates() -> None:
    assert kind_of(["A12", "B7", "C3", "A12"]) == ColumnKind.CATEGORICAL


def test_unique_integers_are_ids_only_with_a_hint() -> None:
    shuffled = np.random.default_rng(0).permutation(100)
    assert kind_of(shuffled, name="user_id") == ColumnKind.ID_LIKE
    assert kind_of(np.arange(100), name="row") == ColumnKind.ID_LIKE  # increasing row numbers
    assert kind_of(shuffled + 18, name="age") == ColumnKind.NUMERIC  # just happens to be unique


def test_unique_strings_are_id_like() -> None:
    names = [f"Person {i}" for i in range(50)]
    assert kind_of(names, name="name") == ColumnKind.ID_LIKE


def test_profile_is_json_serialisable_and_has_no_rows(churn_df: pd.DataFrame) -> None:
    payload = profile_dataset(churn_df).to_dict()
    text = json.dumps(payload)
    assert payload["n_rows"] == 400
    # At most a handful of example values per column, never whole columns.
    assert all(len(c["examples"]) <= 5 for c in payload["columns"])
    assert len(text) < 10_000


def test_duplicate_column_names_are_rejected() -> None:
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="Duplicate"):
        profile_dataset(df)
