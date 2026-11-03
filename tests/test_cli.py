import json
from pathlib import Path

import pandas as pd
import pytest

from automl.cli import main


@pytest.fixture
def churn_csv(churn_df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "churn.csv"
    churn_df.to_csv(path, index=False)
    return path


def test_profile_report(churn_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", str(churn_csv), "--target", "churned"]) == 0
    out = capsys.readouterr().out
    assert "predict 'churned' (binary), metric roc_auc" in out
    assert "cancellation_date: possible leakage" in out


def test_profile_json(churn_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", str(churn_csv), "--target", "churned", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"]["target"] == "churned"
    assert len(payload["profile"]["columns"]) == 11


def test_profile_unknown_target(churn_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", str(churn_csv), "--target", "nope"]) == 1
    assert "not in the data" in capsys.readouterr().err


def test_fit_then_predict(
    churn_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model.joblib"
    args = ["fit", str(churn_csv), "--target", "churned", "--time-budget", "3", "--trials", "3"]
    assert main([*args, "--out", str(model_path)]) == 0
    out = capsys.readouterr().out
    assert "Leaderboard" in out and "xgboost" in out
    assert model_path.exists()

    predictions = tmp_path / "predictions.csv"
    assert main(["predict", str(model_path), str(churn_csv), "--out", str(predictions)]) == 0
    result = pd.read_csv(predictions)
    assert list(result.columns) == ["prediction", "proba_no", "proba_yes"]
    assert len(result) == 400
