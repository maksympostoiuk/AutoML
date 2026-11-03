"""Dataset profile: what kind of data each column holds.

The profile is the only view of the data that leaves this package: the LLM layer receives the
profile (types, missing shares, a few example values), never the rows themselves. It is also
what the preprocessing step uses to decide how to treat every column, so a column is never
dropped silently: anything the models cannot use gets a kind that says why.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


class ColumnKind(StrEnum):
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    # Kinds the models do not use as features:
    ID_LIKE = "id_like"  # a different value in (almost) every row: row identifiers, names
    TEXT = "text"  # free text: needs NLP, out of scope
    CONSTANT = "constant"  # one value (or none): carries no information
    EMPTY = "empty"  # all values missing


USABLE_KINDS = frozenset(
    {ColumnKind.NUMERIC, ColumnKind.BOOLEAN, ColumnKind.CATEGORICAL, ColumnKind.DATETIME}
)

# Thresholds for the heuristics below. They are deliberately simple and documented, because
# the LLM layer and the user can always override the result.
ID_UNIQUE_SHARE = 0.95  # share of distinct values above which a column looks like an ID
TEXT_MIN_MEAN_LENGTH = 30  # average string length above which a column looks like free text
TEXT_UNIQUE_SHARE = 0.5
DATETIME_PARSE_SHARE = 0.9  # share of sampled strings that must parse as dates
N_EXAMPLES = 5
_DATE_LIKE = re.compile(r"\d.*[-/.:].*\d|\d{8}")
_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key)$|^id(_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    kind: ColumnKind
    n_missing: int
    missing_share: float
    n_unique: int
    unique_share: float
    examples: list[Any]
    stats: dict[str, float] = field(default_factory=dict)  # min/max/mean/std for numbers
    top_values: dict[str, int] = field(default_factory=dict)  # most frequent categories
    reason: str = ""  # why the kind was chosen, when it is not obvious from the dtype

    @property
    def usable(self) -> bool:
        return self.kind in USABLE_KINDS


@dataclass(frozen=True)
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]

    def __getitem__(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return any(column.name == name for column in self.columns)

    @property
    def names(self) -> list[str]:
        return [column.name for column in self.columns]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form: what the LLM layer sends. No full rows, only summaries."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": [
                {key: value for key, value in asdict(column).items() if value not in ({}, "")}
                for column in self.columns
            ],
        }

    def summary(self) -> str:
        """A plain-text table for the terminal."""
        header = f"{'column':<24} {'kind':<12} {'dtype':<10} {'missing':>8} {'unique':>8}  note"
        lines = [f"{self.n_rows:,} rows x {self.n_columns} columns", header, "-" * len(header)]
        for c in self.columns:
            lines.append(
                f"{c.name[:24]:<24} {c.kind:<12} {c.dtype[:10]:<10} "
                f"{c.missing_share:>7.0%} {c.n_unique:>8,}  {c.reason}"
            )
        return "\n".join(lines)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    if df.columns.duplicated().any():
        duplicated = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(f"Duplicate column names: {duplicated}")
    columns = [profile_column(df[name]) for name in df.columns]
    return DatasetProfile(n_rows=len(df), n_columns=df.shape[1], columns=columns)


def profile_column(series: pd.Series) -> ColumnProfile:
    n_rows = len(series)
    present = series.dropna()
    n_unique = int(present.nunique())
    kind, reason = _detect_kind(series, present, n_unique)

    stats: dict[str, float] = {}
    top_values: dict[str, int] = {}
    if kind == ColumnKind.NUMERIC and len(present):
        values = present.astype(float)
        stats = {
            "min": _round(values.min()),
            "max": _round(values.max()),
            "mean": _round(values.mean()),
            "std": _round(values.std()) if len(values) > 1 else 0.0,
        }
    if kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
        top_values = {str(k): int(v) for k, v in present.value_counts().head(10).items()}

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        kind=kind,
        n_missing=int(n_rows - len(present)),
        missing_share=_round(1 - len(present) / n_rows) if n_rows else 0.0,
        n_unique=n_unique,
        unique_share=_round(n_unique / len(present)) if len(present) else 0.0,
        examples=[_plain(v) for v in present.drop_duplicates().head(N_EXAMPLES)],
        stats=stats,
        top_values=top_values,
        reason=reason,
    )


def _detect_kind(series: pd.Series, present: pd.Series, n_unique: int) -> tuple[ColumnKind, str]:
    if len(present) == 0:
        return ColumnKind.EMPTY, "all values are missing"
    if n_unique == 1 and len(present) == len(series):
        return ColumnKind.CONSTANT, f"only one value: {_plain(present.iloc[0])!r}"
    if n_unique == 1:
        # One value plus gaps is not constant: "filled in or not" can be a strong signal.
        return ColumnKind.CATEGORICAL, "one value or missing: works as a filled-in flag"

    # Order matters: in pandas, bool is also "numeric".
    if ptypes.is_bool_dtype(series):
        return ColumnKind.BOOLEAN, ""
    if ptypes.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME, ""
    if ptypes.is_numeric_dtype(series):
        return _numeric_kind(series, present, n_unique)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return ColumnKind.CATEGORICAL, ""
    return _string_kind(present.astype(str), n_unique)


def _numeric_kind(series: pd.Series, present: pd.Series, n_unique: int) -> tuple[ColumnKind, str]:
    is_integer = ptypes.is_integer_dtype(series) or bool(np.all(np.mod(present, 1) == 0))
    if n_unique == 2 and set(present.unique()) <= {0, 1}:
        return ColumnKind.BOOLEAN, "only 0 and 1"
    if is_integer and n_unique / len(present) >= ID_UNIQUE_SHARE:
        if _ID_NAME.search(str(series.name)):
            return ColumnKind.ID_LIKE, "unique integers with an ID-like name"
        if present.is_monotonic_increasing:
            return ColumnKind.ID_LIKE, "unique, increasing integers (a row number)"
    return ColumnKind.NUMERIC, ""


def _string_kind(values: pd.Series, n_unique: int) -> tuple[ColumnKind, str]:
    unique_share = n_unique / len(values)
    if _looks_like_dates(values):
        return ColumnKind.DATETIME, "text values that parse as dates"
    mean_length = values.str.len().mean()
    if mean_length >= TEXT_MIN_MEAN_LENGTH and unique_share >= TEXT_UNIQUE_SHARE:
        return ColumnKind.TEXT, f"free text, {mean_length:.0f} characters on average"
    if unique_share >= ID_UNIQUE_SHARE and n_unique > 20:
        return ColumnKind.ID_LIKE, f"{unique_share:.0%} of values are distinct"
    return ColumnKind.CATEGORICAL, ""


def _looks_like_dates(values: pd.Series) -> bool:
    sample = values.drop_duplicates().head(200)
    if not sample.str.contains(_DATE_LIKE).mean() >= DATETIME_PARSE_SHARE:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= DATETIME_PARSE_SHARE)


def _round(value: float) -> float:
    return float(np.round(float(value), 6))


def _plain(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python values for JSON."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value
