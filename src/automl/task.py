"""TaskSpec: what to predict and how to judge it.

A ``TaskSpec`` is the contract between "understanding the task" and "training models". It can
be written by hand, inferred by the heuristics below, or (stage 6) proposed by an LLM. Whoever
writes it, ``check_against`` validates it against the actual data before anything is trained.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from automl.data.leakage import find_leakage_suspects, find_missingness_warnings
from automl.data.profile import ColumnKind, DatasetProfile, profile_dataset


class ProblemType(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


# Allowed metrics per problem type; the first one is the default.
METRICS: dict[ProblemType, tuple[str, ...]] = {
    ProblemType.BINARY: ("roc_auc", "log_loss", "f1", "accuracy"),
    ProblemType.MULTICLASS: ("log_loss", "f1_macro", "accuracy"),
    ProblemType.REGRESSION: ("rmse", "mae", "r2"),
}

MAX_CLASSES = 50  # more distinct labels than this is almost certainly not a classification task
MAX_INTEGER_CLASSES = 10  # integer targets with at most this many values are treated as classes
TARGET_NAME_HINTS = ("target", "label", "class", "y", "outcome", "churn", "survived", "default")


class TaskSpecError(ValueError):
    """The spec does not fit the data. ``problems`` lists every issue found, not just the first."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


class TaskSpec(BaseModel):
    target: str = Field(description="Name of the column to predict.")
    problem_type: ProblemType
    metric: str = Field(description="Metric used to compare models; see METRICS.")
    drop_columns: dict[str, str] = Field(
        default_factory=dict, description="Columns not used as features, with the reason."
    )
    notes: list[str] = Field(
        default_factory=list, description="Assumptions and warnings to show the user."
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        allowed = METRICS[self.problem_type]
        if self.metric not in allowed:
            raise ValueError(
                f"metric {self.metric!r} does not fit {self.problem_type.value}; use one of "
                f"{list(allowed)}"
            )
        if self.target in self.drop_columns:
            raise ValueError("the target column cannot also be in drop_columns")
        return self

    def check_against(self, df: pd.DataFrame, profile: DatasetProfile | None = None) -> None:
        """Raise ``TaskSpecError`` listing everything that does not match the data."""
        profile = profile or profile_dataset(df)
        problems: list[str] = []
        if self.target not in df.columns:
            raise TaskSpecError([f"target column {self.target!r} is not in the data"])

        missing_drops = [c for c in self.drop_columns if c not in df.columns]
        if missing_drops:
            problems.append(f"drop_columns not in the data: {missing_drops}")

        y = df[self.target].dropna()
        n_classes = y.nunique()
        if len(y) == 0:
            problems.append(f"target {self.target!r} has no values")
        elif self.problem_type == ProblemType.REGRESSION:
            if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
                problems.append(f"regression needs a numeric target, {self.target!r} is not")
        elif self.problem_type == ProblemType.BINARY and n_classes != 2:
            problems.append(f"binary classification needs 2 classes, the target has {n_classes}")
        elif self.problem_type == ProblemType.MULTICLASS and not 3 <= n_classes <= MAX_CLASSES:
            problems.append(
                f"multiclass classification needs 3 to {MAX_CLASSES} classes, "
                f"the target has {n_classes}"
            )

        features = self.feature_columns(profile)
        if not features:
            problems.append("no usable feature columns are left")
        if problems:
            raise TaskSpecError(problems)

    def feature_columns(self, profile: DatasetProfile) -> list[str]:
        """Columns the models will use: usable kinds, minus the target and dropped columns."""
        return [
            c.name
            for c in profile.columns
            if c.usable and c.name != self.target and c.name not in self.drop_columns
        ]

    @property
    def is_classification(self) -> bool:
        return self.problem_type != ProblemType.REGRESSION


def infer_problem_type(target: pd.Series) -> tuple[ProblemType, str]:
    """Problem type from the target column alone, with the reason in words."""
    y = target.dropna()
    n_unique = y.nunique()
    if n_unique < 2:
        raise TaskSpecError([f"target {target.name!r} has fewer than 2 distinct values"])

    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        if n_unique > MAX_CLASSES:
            raise TaskSpecError(
                [f"target {target.name!r} has {n_unique} distinct labels: too many classes"]
            )
        kind = ProblemType.BINARY if n_unique == 2 else ProblemType.MULTICLASS
        return kind, f"the target holds labels ({n_unique} distinct values)"

    if n_unique == 2:
        return ProblemType.BINARY, "the target has exactly two values"
    is_integer = bool(np.all(np.mod(y.astype(float), 1) == 0))
    if is_integer and n_unique <= MAX_INTEGER_CLASSES:
        return (
            ProblemType.MULTICLASS,
            f"the target is an integer with only {n_unique} values; if they are amounts "
            "rather than categories, switch to regression",
        )
    return ProblemType.REGRESSION, "the target is a number with many distinct values"


def guess_target(profile: DatasetProfile) -> tuple[str, str]:
    """Best guess of the target column when nobody named it: by name, else the last column."""
    for hint in TARGET_NAME_HINTS:
        for column in profile.columns:
            if column.name.lower() == hint and column.kind != ColumnKind.ID_LIKE:
                return column.name, f"the column is called {column.name!r}"
    return profile.columns[-1].name, "no obvious name, so the last column (a common convention)"


def infer_task_spec(
    df: pd.DataFrame,
    target: str | None = None,
    profile: DatasetProfile | None = None,
    check_leakage: bool = True,
) -> TaskSpec:
    """Build a TaskSpec without an LLM. Every decision is recorded in ``notes``."""
    profile = profile or profile_dataset(df)
    notes: list[str] = []

    if target is None:
        target, why = guess_target(profile)
        notes.append(f"Target guessed: {target!r} ({why}).")
    elif target not in df.columns:
        raise TaskSpecError([f"target column {target!r} is not in the data"])

    problem_type, why = infer_problem_type(df[target])
    notes.append(f"Problem type {problem_type.value}: {why}.")
    metric = METRICS[problem_type][0]

    drop_columns = {
        c.name: f"{c.kind.value}: {c.reason}" if c.reason else c.kind.value
        for c in profile.columns
        if c.name != target and not c.usable
    }

    if check_leakage:
        for suspect in find_leakage_suspects(df, target, problem_type.value, profile):
            drop_columns[suspect.column] = suspect.reason
            notes.append(
                f"Column {suspect.column!r} was dropped as {suspect.reason}. "
                "Keep it only if its value is known at prediction time."
            )
        for warning in find_missingness_warnings(df, target, problem_type.value, profile):
            if warning.column not in drop_columns:
                notes.append(warning.message)

    n_missing_target = int(df[target].isna().sum())
    if n_missing_target:
        notes.append(f"{n_missing_target} rows without a target value will be skipped.")

    return TaskSpec(
        target=target,
        problem_type=problem_type,
        metric=metric,
        drop_columns=drop_columns,
        notes=notes,
    )
