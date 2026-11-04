import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPORT_PY = Path(__file__).resolve().parents[1] / "benchmarks" / "report.py"


@pytest.fixture
def report():
    spec = importlib.util.spec_from_file_location("benchmark_report", REPORT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_results() -> pd.DataFrame:
    rows = []
    scores = {
        "dummy": 0.5,
        "logistic_regression": 0.80,
        "random_forest": 0.78,
        "xgboost_default": 0.74,
        "xgboost": 0.76,
    }
    for seed in (0, 1):
        for validation in ("holdout", "5-fold CV"):
            for model, test in scores.items():
                rows.append(
                    {
                        "dataset": "toy",
                        "rows": 500,
                        "problem_type": "binary",
                        "metric": "roc_auc",
                        "greater_is_better": True,
                        "seed": seed,
                        "validation": validation,
                        "model": model,
                        "valid_score": 0.9 if model == "xgboost" else test,
                        "valid_std": 0.02,
                        "test_score": test,
                        # holdout and "best" fall for the overfitted xgboost validation score;
                        # the 1-SE rule picks the linear model.
                        "chosen_by_best": model == "xgboost",
                        "chosen_by_one_se": model
                        == ("xgboost" if validation == "holdout" else "logistic_regression"),
                        "xgboost_trials": 10,
                        "run_seconds": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_selection_table_measures_the_captured_gain(report) -> None:
    df = fake_results()
    df["model"] = df["model"].replace(report.LINEAR)
    _, picks = report.selection_table(df)
    captured = picks.groupby("strategy")["captured"].mean()
    # oracle 0.80, dummy 0.5: xgboost (0.76) captures 26/30 of the gain, linear all of it.
    assert captured["holdout, best"] == pytest.approx(0.26 / 0.30)
    assert captured["CV, 1-SE"] == pytest.approx(1.0)


def test_report_keeps_the_hand_written_part(
    report, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, output = tmp_path / "results.csv", tmp_path / "results.md"
    fake_results().to_csv(results, index=False)
    output.write_text(f"# Benchmark\n\nMy notes.\n\n{report.GENERATED}\nold tables\n")
    monkeypatch.setattr(report, "RESULTS", results)
    monkeypatch.setattr(report, "REPORT", output)
    report.main()
    text = output.read_text(encoding="utf-8")
    assert "My notes." in text
    assert "old tables" not in text
    assert "| toy | 500 | binary |" in text
