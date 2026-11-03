"""Turn a table into a numeric feature matrix, column kind by column kind.

Two flavours of the same plan:

* ``for_trees`` (XGBoost, random forest): numbers are passed through untouched, missing values
  included, because XGBoost learns where to send them (stage 1b) and scaling does not change
  a tree. Categories are one-hot encoded with a "missing" level of their own.
* ``for_linear`` (linear and logistic regression baselines): numbers are imputed with the
  median, get a "was missing" flag and are standardised; categories as above.

Dates become calendar features (year, month, day, day of week, hour). The output is a pandas
DataFrame with readable column names, which SHAP explanations rely on later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from automl.data.profile import ColumnKind, DatasetProfile
from automl.task import TaskSpec

MAX_CATEGORIES = 20  # per column; rarer levels are grouped into "infrequent"
MISSING_CATEGORY = "__missing__"


@dataclass(frozen=True)
class FeaturePlan:
    """Which columns go through which transformation, and which are ignored and why."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    ignored: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical + self.datetime


def plan_features(profile: DatasetProfile, spec: TaskSpec) -> FeaturePlan:
    numeric, categorical, datetime = [], [], []
    ignored = dict(spec.drop_columns)
    for column in profile.columns:
        if column.name == spec.target or column.name in spec.drop_columns:
            continue
        if column.kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN):
            numeric.append(column.name)
        elif column.kind == ColumnKind.CATEGORICAL:
            categorical.append(column.name)
        elif column.kind == ColumnKind.DATETIME:
            datetime.append(column.name)
        else:
            ignored[column.name] = column.kind.value
    return FeaturePlan(numeric, categorical, datetime, ignored)


def make_preprocessor(plan: FeaturePlan, for_linear: bool = False) -> ColumnTransformer:
    to_float = FunctionTransformer(_to_float, feature_names_out="one-to-one")
    if for_linear:
        numeric = make_pipeline(
            to_float, SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
        )
        dates = make_pipeline(
            DatetimeFeatures(),
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
        )
    else:
        numeric, dates = to_float, DatetimeFeatures()

    categorical = Pipeline(
        [
            ("as_text", FunctionTransformer(_to_text, feature_names_out="one-to-one")),
            ("missing", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=MAX_CATEGORIES,
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []
    if plan.numeric:
        transformers.append(("numeric", numeric, plan.numeric))
    if plan.categorical:
        transformers.append(("categorical", categorical, plan.categorical))
    if plan.datetime:
        transformers.append(("datetime", dates, plan.datetime))
    if not transformers:
        raise ValueError("No usable feature columns")

    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")


class DatetimeFeatures(TransformerMixin, BaseEstimator):
    """Calendar features from date columns (real datetimes or date strings).

    The hour is added only if the training data has times other than midnight.
    """

    PARTS = ("year", "month", "day", "day_of_week")

    def fit(self, X: pd.DataFrame, y: object = None) -> DatetimeFeatures:
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.with_hour_ = {
            column: bool((_as_datetime(X[column]).dt.hour.fillna(0) != 0).any())
            for column in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        out = {}
        for column in self.feature_names_in_:
            values = _as_datetime(X[column])
            for part in self._parts(column):
                out[f"{column}_{part}"] = getattr(values.dt, part).astype(float)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(
            [
                f"{column}_{part}"
                for column in self.feature_names_in_
                for part in self._parts(column)
            ],
            dtype=object,
        )

    def _parts(self, column: str) -> tuple[str, ...]:
        return (*self.PARTS, "hour") if self.with_hour_[column] else self.PARTS


def _as_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _to_float(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype(float)


def _to_text(X: pd.DataFrame) -> pd.DataFrame:
    # One representation for every category, whatever the original dtype; NaN stays NaN.
    return pd.DataFrame(X).astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
