"""Detect columns that leak the answer.

A leaking column holds information that would not be available at prediction time, usually
because it was recorded after the outcome: "cancellation_date" when predicting churn, the
lifeboat number when predicting Titanic survival. Models trained with such a column look
excellent and are useless in production.

Two checks:

* ``find_leakage_suspects``: for every column, fit a small decision tree on that column alone
  and measure it with cross-validation. Real features almost never predict the target nearly
  perfectly on their own; a column that does is a suspect and is dropped by default. Missing
  values are kept as their own signal, because "the field is filled in" is a classic leak.
* ``find_missingness_warnings``: a column filled in for only part of the rows, where those rows
  (almost) all share one class. On average such a column predicts little, so the first check
  misses it, but it may still be written after the outcome: the body identification number of
  a Titanic passenger exists only for those who died. Legitimate columns can look like this
  too, so these are warnings for a human (or the LLM layer) to judge, not automatic drops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl.data.profile import ColumnKind, DatasetProfile

LEAKAGE_THRESHOLD = 0.97  # single-column score (ROC AUC, balanced accuracy or R^2)
MAX_ROWS = 5_000  # the check runs on a sample: it only needs to detect near-perfect columns


@dataclass(frozen=True)
class LeakageSuspect:
    column: str
    score: float
    metric: str

    @property
    def reason(self) -> str:
        return f"possible leakage: alone predicts the target with {self.metric} {self.score:.3f}"


def find_leakage_suspects(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    profile: DatasetProfile,
    threshold: float = LEAKAGE_THRESHOLD,
    random_state: int = 0,
) -> list[LeakageSuspect]:
    data = df[df[target].notna()]
    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=random_state)
    y = data[target]

    if problem_type == "regression":
        model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=random_state)
        metric, y = "r2", y.astype(float)
    else:
        model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=random_state)
        metric = "roc_auc" if problem_type == "binary" else "balanced_accuracy"
        y = y.astype(str)
        if y.value_counts().min() < 3:
            return []  # too few rows in some class for a 3-fold check

    suspects = []
    for column in profile.columns:
        if column.name == target or not column.usable:
            continue
        x = _encode_single_column(data[column.name], column.kind)
        score = float(np.mean(cross_val_score(model, x, y, cv=3, scoring=metric)))
        if score >= threshold:
            suspects.append(LeakageSuspect(column.name, round(score, 4), metric))
    return sorted(suspects, key=lambda s: -s.score)


def _encode_single_column(series: pd.Series, kind: ColumnKind) -> np.ndarray:
    """One column as a numeric matrix: values (or category codes) plus a missing flag."""
    missing = series.isna().to_numpy()
    if kind == ColumnKind.DATETIME:
        values = pd.to_datetime(series, errors="coerce", format="mixed")
        numbers = (values - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    elif kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN):
        numbers = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    else:
        numbers = pd.factorize(series.astype(str))[0].astype(float)
    numbers = np.where(np.isnan(numbers), -1e30, numbers)
    return np.column_stack([numbers, missing.astype(float)])


PURE_SHARE = 0.99  # share of one class among the rows where the column is filled (or empty)
MIN_SEGMENT_SHARE = 0.05  # the segment must cover at least this share of rows
MAX_BASE_RATE = 0.9  # ...and the class must not simply dominate the whole dataset


@dataclass(frozen=True)
class MissingnessWarning:
    column: str
    filled: bool  # True: "when filled in"; False: "when empty"
    label: str
    n_rows: int
    share: float

    @property
    def message(self) -> str:
        state = "filled in" if self.filled else "empty"
        return (
            f"When {self.column!r} is {state} ({self.n_rows} rows), the target is "
            f"{self.label!r} in {self.share:.0%} of them. If this column is recorded after the "
            "outcome, it leaks the answer and should be dropped."
        )


def find_missingness_warnings(
    df: pd.DataFrame, target: str, problem_type: str, profile: DatasetProfile
) -> list[MissingnessWarning]:
    if problem_type == "regression":
        return []
    data = df[df[target].notna()]
    y = data[target].astype(str)
    base_rates = y.value_counts(normalize=True)

    warnings = []
    for column in profile.columns:
        if column.name == target or column.n_missing == 0 or not column.usable:
            continue
        filled = data[column.name].notna()
        for state, mask in ((True, filled), (False, ~filled)):
            if mask.mean() < MIN_SEGMENT_SHARE:
                continue
            shares = y[mask].value_counts(normalize=True)
            label, share = str(shares.index[0]), float(shares.iloc[0])
            if share >= PURE_SHARE and base_rates[label] <= MAX_BASE_RATE:
                warnings.append(
                    MissingnessWarning(column.name, state, label, int(mask.sum()), share)
                )
    return warnings
