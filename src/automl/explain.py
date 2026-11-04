"""SHAP explanations: how much each column pushed each prediction up or down.

SHAP values split one prediction into a sum:

    prediction = base value + contribution(column 1) + ... + contribution(column n)

where the base value is the average prediction and each contribution is that column's fair
share of the difference (Shapley values from game theory). Two views come from the same
numbers:

* local: why did *this* row get *this* prediction? (``Explanation.row``)
* global: which columns matter most overall? (mean absolute contribution,
  ``Explanation.importance``)

How the contributions are computed depends on the chosen model, always exactly:

* XGBoost: its built-in TreeSHAP (``pred_contribs=True``), in log-odds for classification;
* random forest: ``shap.TreeExplainer``, in probability for classification;
* linear and logistic regression: coefficient x (value - average value), in log-odds;
* dummy: every contribution is zero.

The model sees preprocessed features (one-hot columns, date parts). Because SHAP values add
up, the contributions of all features made from one original column are summed back into
that column: the user sees "contract", not "contract_one year" and "contract_two year".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from automl.preprocess import safe_name

if TYPE_CHECKING:
    from automl.engine import FittedModel

MAX_ROWS = 2_000  # SHAP on more rows adds little to global importance and costs time
MISSING_INDICATOR_PREFIX = "missingindicator_"


@dataclass
class Explanation:
    columns: list[str]  # original feature columns
    values: np.ndarray  # contributions, shape (rows, columns, outputs)
    base_values: np.ndarray  # shape (outputs,)
    outputs: list[str]  # one per explained output: a class label, or the target name
    units: str  # "log-odds", "probability" or "target units"
    data: pd.DataFrame  # the explained rows, original columns and values
    predicted_output: np.ndarray  # per row: index of the output to show (predicted class)

    def importance(self) -> pd.Series:
        """Global importance: mean |contribution| per column, summed over outputs."""
        scores = np.abs(self.values).mean(axis=0).sum(axis=1)
        return pd.Series(scores, index=self.columns, name="mean_abs_shap").sort_values(
            ascending=False
        )

    def row(self, i: int, output: int | None = None) -> pd.DataFrame:
        """Local explanation of the i-th explained row, largest contributions first."""
        k = int(self.predicted_output[i]) if output is None else output
        table = pd.DataFrame(
            {
                "column": self.columns,
                "value": [self.data.iloc[i][c] for c in self.columns],
                "contribution": self.values[i, :, k],
            }
        )
        order = table["contribution"].abs().sort_values(ascending=False).index
        return table.loc[order].reset_index(drop=True)

    def row_total(self, i: int, output: int | None = None) -> float:
        """Base value + all contributions: the model's raw output for this row."""
        k = int(self.predicted_output[i]) if output is None else output
        return float(self.base_values[k] + self.values[i, :, k].sum())

    def describe_row(self, i: int) -> str:
        """The row's prediction in words: average prediction -> this row."""
        k = int(self.predicted_output[i])
        output, base, total = self.outputs[k], float(self.base_values[k]), self.row_total(i)
        if self.units == "log-odds":
            return f"P({output}) {expit(base):.0%} on average -> {expit(total):.0%} here"
        if self.units == "probability":
            return f"P({output}) {base:.0%} on average -> {total:.0%} here"
        return f"{output} {base:,.4g} on average -> {total:,.4g} here"

    def to_dict(self, top: int = 10) -> dict[str, Any]:
        """Compact, JSON-friendly summary for reports and the LLM layer."""
        importance = self.importance()
        return {
            "units": self.units,
            "outputs": self.outputs,
            "n_rows_explained": len(self.data),
            "importance": {k: round(float(v), 6) for k, v in importance.head(top).items()},
        }


def explain(model: FittedModel, df: pd.DataFrame, max_rows: int = MAX_ROWS) -> Explanation:
    data = df[model.feature_columns]
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=0)

    preprocess: ColumnTransformer = model.pipeline.named_steps["preprocess"]
    estimator = model.pipeline.named_steps["model"]
    X = preprocess.transform(data)
    feature_values, base, units = _contributions(estimator, X)

    columns, groups = _group_features(preprocess, list(X.columns), model.feature_columns)
    values = np.stack([feature_values[:, idx, :].sum(axis=1) for idx in groups], axis=1)

    if model.classes is None:
        outputs, predicted = [model.spec.target], np.zeros(len(data), dtype=int)
    else:
        outputs = [str(c) for c in model.classes]
        proba = np.asarray(model.pipeline.predict_proba(data))
        predicted = proba.argmax(axis=1)
        if len(outputs) == 2:
            # Binary models have one output: the score of the second (positive) class.
            outputs, predicted = [outputs[1]], np.zeros(len(data), dtype=int)

    return Explanation(
        columns=columns,
        values=values,
        base_values=base,
        outputs=outputs,
        units=units,
        data=data.reset_index(drop=True),
        predicted_output=predicted,
    )


# ----- contributions per model type -----------------------------------------------------------


def _contributions(estimator: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (values (rows, features, outputs), base values (outputs,), units)."""
    if isinstance(estimator, TransformedTargetRegressor):
        # XGBoost regression is trained on a standardised target: undo the scaling.
        values, base, _ = _contributions(estimator.regressor_, X)
        scale = float(estimator.transformer_.scale_[0])
        mean = float(estimator.transformer_.mean_[0])
        return values * scale, base * scale + mean, "target units"

    if isinstance(estimator, xgb.XGBModel):
        raw = estimator.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        if raw.ndim == 2:  # binary or regression: (rows, features + 1)
            raw = raw[:, None, :]
        values = np.transpose(raw[:, :, :-1], (0, 2, 1))  # (rows, features, outputs)
        base = raw[0, :, -1]  # the bias column is the same for every row
        units = "log-odds" if isinstance(estimator, xgb.XGBClassifier) else "target units"
        return values.astype(float), base.astype(float), units

    if isinstance(estimator, RandomForestClassifier | RandomForestRegressor):
        import shap  # imported lazily: only needed when a forest wins

        explainer = shap.TreeExplainer(estimator)
        values = np.asarray(explainer.shap_values(X.to_numpy(), check_additivity=False))
        base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
        if values.ndim == 2:
            values = values[:, :, None]
        if isinstance(estimator, RandomForestClassifier):
            if values.shape[2] == 2:  # keep only the positive class, like the other models
                values, base = values[:, :, 1:], base[1:]
            return values, base, "probability"
        return values, base, "target units"

    if isinstance(estimator, LogisticRegression | Ridge):
        coef = np.atleast_2d(estimator.coef_)  # (outputs, features)
        centred = X.to_numpy() - X.to_numpy().mean(axis=0)
        values = centred[:, :, None] * coef.T[None, :, :]
        base = np.atleast_1d(estimator.intercept_) + coef @ X.to_numpy().mean(axis=0)
        units = "log-odds" if isinstance(estimator, LogisticRegression) else "target units"
        return values, np.asarray(base, dtype=float), units

    if isinstance(estimator, DummyClassifier | DummyRegressor):
        n_outputs = 1
        if isinstance(estimator, DummyClassifier) and len(estimator.classes_) > 2:
            n_outputs = len(estimator.classes_)
        values = np.zeros((len(X), X.shape[1], n_outputs))
        if isinstance(estimator, DummyRegressor):
            return values, np.atleast_1d(estimator.constant_.ravel()).astype(float), "target units"
        prior = np.asarray(estimator.class_prior_, dtype=float)
        base = prior[1:] if n_outputs == 1 else prior
        return values, base, "probability"

    raise TypeError(f"No explainer for {type(estimator).__name__}")


# ----- mapping model features back to original columns ----------------------------------------


def _group_features(
    preprocess: ColumnTransformer, feature_names: list[str], original_columns: list[str]
) -> tuple[list[str], list[list[int]]]:
    """For every original column, the positions of the model features made from it."""
    position = {name: i for i, name in enumerate(feature_names)}
    groups: dict[str, list[int]] = {}
    for _, transformer, columns in preprocess.transformers_:
        if transformer == "drop" or not len(columns):
            continue
        names = transformer.get_feature_names_out(columns)
        for name in names:
            groups.setdefault(_source_column(str(name), list(columns)), []).append(position[name])
    kept = [c for c in original_columns if c in groups]
    return kept, [groups[c] for c in kept]


def _source_column(feature: str, columns: list[str]) -> str:
    """The original column a feature came from: exact name, or the longest "column_" prefix.

    Feature names are built from ``safe_name(column)``, so the comparison uses it too.
    """
    if feature.startswith(MISSING_INDICATOR_PREFIX):
        feature = feature[len(MISSING_INDICATOR_PREFIX) :]
    safe = {safe_name(c): c for c in columns}
    if feature in safe:
        return safe[feature]
    matches = [name for name in safe if feature.startswith(f"{name}_")]
    if not matches:
        raise ValueError(f"Cannot tell which column feature {feature!r} came from")
    return safe[max(matches, key=len)]
