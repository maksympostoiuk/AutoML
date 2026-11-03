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
