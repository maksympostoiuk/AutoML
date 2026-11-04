"""Turn ``results/results.csv`` into ``results.md``.

    uv run python benchmarks/report.py

Everything above the ``GENERATED`` marker in results.md is hand-written commentary and is kept;
everything below it is regenerated from the CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "results.csv"
REPORT = ROOT / "results.md"
GENERATED = "<!-- GENERATED: everything below is rebuilt by benchmarks/report.py -->"
MODELS = ["dummy", "linear", "random_forest", "xgboost_default", "xgboost"]
LINEAR = {"logistic_regression": "linear", "ridge": "linear"}


def fmt(value: float) -> str:
    if np.isnan(value):
        return "–"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3f}"


def default_runs(df: pd.DataFrame) -> pd.DataFrame:
    """The runs a user gets by default: CV for small data, holdout for large."""
    has_cv = df.groupby("dataset")["validation"].transform(lambda v: (v != "holdout").any())
    return df[(df["validation"] != "holdout") | ~has_cv]


def model_table(df: pd.DataFrame) -> str:
    runs = default_runs(df)
    lines = [
        "| dataset | rows | task | metric | " + " | ".join(MODELS) + " |",
        "|---|---:|---|---|" + "---:|" * len(MODELS),
    ]
    for dataset, group in runs.groupby("dataset", sort=False):
        first = group.iloc[0]
        scores = group.groupby("model")["test_score"].agg(["mean", "std"])
        means = scores["mean"]
        best = means.idxmax() if first["greater_is_better"] else means.idxmin()
        cells = []
        for model in MODELS:
            if model not in scores.index:
                cells.append("–")
                continue
            mean, std = scores.loc[model]
            text = f"{fmt(mean)} ± {fmt(std)}"
            cells.append(f"**{text}**" if model == best else text)
        arrow = "↑" if first["greater_is_better"] else "↓"
        lines.append(
            f"| {dataset} | {first['rows']:,} | {first['problem_type']} | "
            f"{first['metric']} {arrow} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def selection_table(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Small datasets: test score of the model each selection strategy picks."""
    rows = []
    for (dataset, seed), group in df.groupby(["dataset", "seed"], sort=False):
        holdout = group[group["validation"] == "holdout"]
        cv = group[group["validation"] != "holdout"]
        if cv.empty or holdout.empty:
            continue
        sign = 1 if group["greater_is_better"].iloc[0] else -1
        dummy = cv.loc[cv["model"] == "dummy", "test_score"].item()
        oracle = sign * (sign * group["test_score"]).max()  # best test score of any fit model
        picks = {
            "holdout, best": holdout.loc[holdout["chosen_by_best"]].iloc[0],
            "CV, best": cv.loc[cv["chosen_by_best"]].iloc[0],
            "CV, 1-SE (default)": cv.loc[cv["chosen_by_default_rule"]].iloc[0],
        }
        for strategy, pick in picks.items():
            captured = (pick["test_score"] - dummy) / (oracle - dummy) if oracle != dummy else 1
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "strategy": strategy,
                    "model": LINEAR.get(pick["model"], pick["model"]),
                    "test_score": pick["test_score"],
                    "captured": captured,
                }
            )
    picks = pd.DataFrame(rows)
    if picks.empty:
        return "_No datasets with both holdout and CV runs yet._", picks

    strategies = list(dict.fromkeys(picks["strategy"]))
    lines = [
        "| dataset | " + " | ".join(strategies) + " |",
        "|---|" + "---:|" * len(strategies),
    ]
    for dataset, group in picks.groupby("dataset", sort=False):
        means = group.groupby("strategy")["test_score"].mean()
        lines.append(f"| {dataset} | " + " | ".join(fmt(means[s]) for s in strategies) + " |")
    captured = picks.groupby("strategy")["captured"].mean()
    lines.append(
        "| **share of the achievable gain** | "
        + " | ".join(f"**{captured[s]:.1%}**" for s in strategies)
        + " |"
    )
    return "\n".join(lines), picks


def picks_table(picks: pd.DataFrame) -> str:
    counts = picks.groupby(["strategy", "model"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[m for m in MODELS if m in counts.columns])
    lines = [
        "| strategy | " + " | ".join(counts.columns) + " |",
        "|---|" + "---:|" * len(counts.columns),
    ]
    for strategy, row in counts.reindex(list(dict.fromkeys(picks["strategy"]))).iterrows():
        lines.append(f"| {strategy} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def tuning_table(df: pd.DataFrame) -> str:
    runs = default_runs(df)
    lines = [
        "| dataset | metric | xgboost_default | xgboost (tuned) | tuned better in | trials |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, group in runs.groupby("dataset", sort=False):
        sign = 1 if group["greater_is_better"].iloc[0] else -1
        pivot = group.pivot_table(index="seed", columns="model", values="test_score")
        better = int((sign * (pivot["xgboost"] - pivot["xgboost_default"]) > 0).sum())
        trials = group.loc[group["model"] == "xgboost", "xgboost_trials"].mean()
        lines.append(
            f"| {dataset} | {group['metric'].iloc[0]} | {fmt(pivot['xgboost_default'].mean())} | "
            f"{fmt(pivot['xgboost'].mean())} | {better} of {len(pivot)} seeds | {trials:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(RESULTS)
    df["model"] = df["model"].replace(LINEAR)
    selection, picks = selection_table(df)
    generated = "\n\n".join(
        [
            GENERATED,
            "## Test scores by model",
            "Mean ± standard deviation over seeds, on the test split (never used for any "
            "choice). Small datasets use 5-fold CV for tuning, large ones a holdout split. "
            "↑ higher is better, ↓ lower is better; the best mean is in bold.",
            model_table(df),
            "## Does cross-validation pick better models?",
            "Small datasets, same seeds. Each cell is the mean test score of the model a "
            "strategy picks. *Share of the achievable gain*: (picked − dummy) / (best − dummy), "
            "where *best* is the model with the best test score in hindsight; 100% means the "
            "strategy always picked it.",
            selection,
            "Which model each strategy picked (runs):",
            picks_table(picks) if not picks.empty else "",
            "## How much does tuning add?",
            tuning_table(df),
            f"_{df[['dataset', 'seed', 'validation']].drop_duplicates().shape[0]} runs, "
            f"{df['dataset'].nunique()} datasets, "
            "generated from `benchmarks/results/results.csv`._",
        ]
    )
    head = ""
    if REPORT.exists():
        text = REPORT.read_text(encoding="utf-8")
        head = text.split(GENERATED)[0]
    if not head.strip():
        head = "# Benchmark\n\n"
    REPORT.write_text(head + generated + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
