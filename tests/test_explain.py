import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from matplotlib.figure import Figure
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_classification

from automl.cli import main
from automl.engine import AutoML, FittedModel
from automl.explain import _source_column, explain
from automl.plots import plot_importance, plot_row
from conftest import make_churn_df

FAST = {"time_budget": 5, "n_trials": 4, "n_jobs": 1}


def raw_output(model: FittedModel, df: pd.DataFrame) -> np.ndarray:
    """What SHAP values must add up to: margin, probability or value, per model type."""
    X = model.pipeline.named_steps["preprocess"].transform(df[model.feature_columns])
    estimator = model.pipeline.named_steps["model"]
    if isinstance(estimator, TransformedTargetRegressor | xgb.XGBRegressor):
        return estimator.predict(X)
    if isinstance(estimator, xgb.XGBClassifier):
        margin = estimator.predict(X, output_margin=True)
    elif hasattr(estimator, "decision_function"):
        margin = estimator.decision_function(X)
    elif hasattr(estimator, "predict_proba"):
        margin = estimator.predict_proba(X)
    else:
        return estimator.predict(X)
    margin = np.asarray(margin)
    if margin.ndim == 2 and margin.shape[1] == 2:
        margin = margin[:, 1]  # binary: the positive class
    return margin


@pytest.fixture(scope="module")
def binary() -> tuple[AutoML, pd.DataFrame]:
    df = make_churn_df()
    return AutoML(**FAST).fit(df, target="churned"), df


@pytest.fixture(scope="module")
def multiclass() -> tuple[AutoML, pd.DataFrame]:
    X, y = make_classification(
        n_samples=500, n_features=5, n_informative=4, n_redundant=0, n_classes=3, random_state=1
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["colour"] = np.where(X[:, 0] > 0, "warm", "cool")  # a categorical feature too
    df["label"] = np.array(["a", "b", "c"])[y]
    return AutoML(**FAST).fit(df, target="label"), df


@pytest.fixture(scope="module")
def regression() -> tuple[AutoML, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "size_m2": rng.uniform(20, 150, n),
            "district": rng.choice(["centre", "suburb", "village"], n),
            "noise": rng.normal(size=n),
        }
    )
    premium = df["district"].map({"centre": 2000, "suburb": 500, "village": 0})
    df["price"] = 3000 * df["size_m2"] + premium * df["size_m2"] + rng.normal(0, 5000, n)
    return AutoML(**FAST).fit(df, target="price"), df


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_contributions_add_up_for_every_model(task: str, request: pytest.FixtureRequest) -> None:
    """Local accuracy: base value + contributions == the model's raw output, for each row."""
    automl, df = request.getfixturevalue(task)
    rows = df.head(40)
    for name, model in automl.models_.items():
        explanation = explain(model, rows)
        expected = raw_output(model, rows)
        for i in range(len(rows)):
            k = int(explanation.predicted_output[i])
            target = expected[i] if expected.ndim == 1 else expected[i, k]
            assert explanation.row_total(i) == pytest.approx(target, rel=1e-4, abs=1e-4), name


def test_contributions_are_grouped_by_original_column(binary: tuple[AutoML, pd.DataFrame]) -> None:
    automl, df = binary
    explanation = explain(automl.model_, df)
    assert explanation.columns == automl.model_.feature_columns
    assert "contract" in explanation.columns  # not contract_one year, contract_two year
    assert "signup_date" in explanation.columns  # not signup_date_year, ..._month
    assert explanation.values.shape == (len(df), len(explanation.columns), 1)


def test_importance_finds_the_real_drivers(regression: tuple[AutoML, pd.DataFrame]) -> None:
    automl, df = regression
    importance = explain(automl.model_, df).importance()
    assert list(importance.index[:2]) == ["size_m2", "district"]
    assert importance.index[-1] == "noise"


def test_regression_contributions_are_in_target_units(
    regression: tuple[AutoML, pd.DataFrame],
) -> None:
    automl, df = regression
    explanation = explain(automl.model_, df)
    assert explanation.units == "target units"
    assert explanation.base_values[0] == pytest.approx(df["price"].mean(), rel=0.1)


def test_multiclass_explains_the_predicted_class(multiclass: tuple[AutoML, pd.DataFrame]) -> None:
    automl, df = multiclass
    explanation = explain(automl.model_, df.head(50))
    assert explanation.outputs == ["a", "b", "c"]
    predicted = automl.model_.predict(df.head(50))
    assert [explanation.outputs[k] for k in explanation.predicted_output] == list(predicted)


def test_row_table_is_sorted_by_impact(binary: tuple[AutoML, pd.DataFrame]) -> None:
    automl, df = binary
    table = explain(automl.model_, df.head(10)).row(0)
    assert list(table.columns) == ["column", "value", "contribution"]
    impact = table["contribution"].abs().to_numpy()
    assert np.all(impact[:-1] >= impact[1:])


def test_fit_stores_a_json_summary(binary: tuple[AutoML, pd.DataFrame]) -> None:
    automl, _ = binary
    summary = automl.model_.metadata["explanation"]
    json.dumps(summary)
    assert summary["units"] in ("log-odds", "probability")
    assert set(summary["importance"]) <= set(automl.model_.feature_columns)


def test_source_column_mapping() -> None:
    columns = ["plan", "plan_type", "age"]
    assert _source_column("age", columns) == "age"
    assert _source_column("missingindicator_age", columns) == "age"
    assert _source_column("plan_type_premium", columns) == "plan_type"  # longest prefix wins
    assert _source_column("plan_basic", columns) == "plan"
    with pytest.raises(ValueError):
        _source_column("unknown", columns)


def test_plots(binary: tuple[AutoML, pd.DataFrame], tmp_path: Path) -> None:
    automl, df = binary
    explanation = explain(automl.model_, df.head(100))
    for figure in (plot_importance(explanation), plot_row(explanation, 0, top=3)):
        assert isinstance(figure, Figure)
        figure.savefig(tmp_path / "chart.png")
        assert (tmp_path / "chart.png").stat().st_size > 1000


def test_explain_command(
    binary: tuple[AutoML, pd.DataFrame], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    automl, df = binary
    model_path = automl.save(tmp_path / "model.joblib")
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    assert (
        main(["explain", str(model_path), str(csv), "--row", "5", "--out-dir", str(tmp_path)]) == 0
    )
    out = capsys.readouterr().out
    assert "Most important columns" in out and "Row 5: P(yes)" in out
    assert (tmp_path / "row_5.png").exists()
    assert main(["explain", str(model_path), str(csv), "--row", "9999"]) == 1
