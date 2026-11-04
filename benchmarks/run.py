"""Benchmark the AutoML engine on public OpenML datasets.

    uv run python benchmarks/run.py                 # everything (about an hour)
    uv run python benchmarks/run.py --quick         # 2 datasets, 1 seed: a smoke test
    uv run python benchmarks/report.py              # results.csv -> results.md

For every dataset and seed the engine runs end to end, exactly as a user would run it, and
every candidate's validation and test score is recorded. Small datasets (cross-validation
territory) are run twice, with holdout and with 5-fold CV validation, to measure whether CV
and the one-standard-error rule pick better models. Results are appended to
``benchmarks/results/results.csv``; finished runs are skipped, so the script can be resumed.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from automl.engine import CV_MAX_ROWS, AutoML
from automl.metrics import get_metric

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "results.csv"
CACHE = ROOT.parent / "data" / "openml"


@dataclass(frozen=True)
class Dataset:
    name: str  # OpenML name
    version: int
    target: str
    numeric_target: bool = False  # stored as a category in OpenML, but really a number


DATASETS = [
    Dataset("blood-transfusion-service-center", 1, "Class"),
    Dataset("diabetes", 1, "class"),
    Dataset("vehicle", 1, "Class"),
    Dataset("credit-g", 1, "class"),
    Dataset("house_prices", 1, "SalePrice"),
    Dataset("car", 3, "class"),
    Dataset("kc1", 1, "defects"),
    Dataset("abalone", 1, "Class_number_of_rings", numeric_target=True),
    Dataset("phoneme", 1, "Class"),
    Dataset("bank-marketing", 1, "Class"),
    Dataset("adult", 2, "class"),
]
QUICK = ["diabetes", "car"]


def load(dataset: Dataset) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = fetch_openml(
            dataset.name,
            version=dataset.version,
            as_frame=True,
            data_home=str(CACHE),
            parser="liac-arff",  # the default parser fails with pandas 3
        ).frame
    if dataset.numeric_target:
        frame[dataset.target] = pd.to_numeric(frame[dataset.target].astype(str))
    return frame


def run_one(
    dataset: Dataset, df: pd.DataFrame, seed: int, validation: str, budget: float
) -> list[dict]:
    started = time.perf_counter()
    automl = AutoML(time_budget=budget, n_trials=60, validation=validation, random_state=seed)
    automl.fit(df, target=dataset.target)
    seconds = time.perf_counter() - started

    metric = get_metric(automl.spec_.metric)
    board = automl.leaderboard_
    valid, test = board[f"valid_{metric.name}"], board[f"test_{metric.name}"]
    chosen_best = board.loc[(valid if metric.greater_is_better else -valid).idxmax(), "model"]
    chosen_default = board.loc[board["selected"], "model"].item()
    xgb_details = next(c for c in automl.candidates_ if c.name == "xgboost").details

    return [
        {
            "dataset": dataset.name,
            "rows": len(df),
            "problem_type": automl.spec_.problem_type.value,
            "metric": metric.name,
            "greater_is_better": metric.greater_is_better,
            "seed": seed,
            "validation": automl.validation_,
            "model": row["model"],
            "valid_score": row[valid.name],
            "valid_std": next(
                c.details.get("valid_std", np.nan)
                for c in automl.candidates_
                if c.name == row["model"]
            ),
            "test_score": row[test.name],
            "chosen_by_best": row["model"] == chosen_best,
            "chosen_by_default_rule": row["model"] == chosen_default,
            "default_rule": automl.selection_,
            "xgboost_trials": xgb_details["n_trials"],
            "run_seconds": round(seconds, 1),
        }
        for _, row in board.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="2 datasets, 1 seed")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=float, default=45.0, help="seconds per run (small data)")
    args = parser.parse_args()

    datasets = [d for d in DATASETS if not args.quick or d.name in QUICK]
    seeds = range(1 if args.quick else args.seeds)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        previous = pd.read_csv(RESULTS)
        done = set(zip(previous["dataset"], previous["seed"], previous["validation"], strict=True))

    for dataset in datasets:
        df = load(dataset)
        small = len(df) * 0.8 < CV_MAX_ROWS  # development rows after the test split
        plans = [("holdout", args.budget), ("cv", args.budget)] if small else [("holdout", 120.0)]
        for seed in seeds:
            for validation, budget in plans:
                label = "holdout" if validation == "holdout" else "5-fold CV"
                if (dataset.name, seed, label) in done:
                    continue
                rows = run_one(dataset, df, seed, validation, budget)
                pd.DataFrame(rows).to_csv(
                    RESULTS, mode="a", header=not RESULTS.exists(), index=False
                )
                chosen = next(r for r in rows if r["chosen_by_default_rule"])
                print(
                    f"{dataset.name:34} seed {seed} {label:9} -> {chosen['model']:20} "
                    f"test {chosen['metric']} {chosen['test_score']:.4f} "
                    f"({rows[0]['run_seconds']:.0f} s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
