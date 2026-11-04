"""Command-line interface. Commands are added stage by stage (see docs/PLAN.md).

automl profile data.csv [--target churn] [--json]     profile the table, infer the task
automl fit data.csv --target churn [--time-budget 120] [--out artifacts/model.joblib]
automl fit data.csv --spec artifacts/spec.json        use a checked task spec

Task description in plain words, via a chat window (free, no API key):
automl prompt data.csv --describe "who will churn" --out prompt.md   paste into the chat
automl check-spec data.csv answer.json                 save the reply, check it -> spec.json
Or directly through the API (needs ANTHROPIC_API_KEY):
automl fit data.csv --describe "who will churn" --llm anthropic
automl predict artifacts/model.joblib new.csv [--out predictions.csv]
automl explain artifacts/model.joblib data.csv [--row 0] [--out-dir artifacts/explain]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from automl import __version__
from automl.data import profile_dataset
from automl.preprocess import FeaturePlan, plan_features
from automl.task import TaskSpec, TaskSpecError, infer_task_spec

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_SPEC_PATH = Path("artifacts/spec.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="automl", description="AutoML for tabular data.")
    parser.add_argument("--version", action="version", version=f"automl {__version__}")
    commands = parser.add_subparsers(dest="command")

    profile = commands.add_parser("profile", help="profile a CSV file and infer the task")
    profile.add_argument("csv", type=Path)
    profile.add_argument("--target", help="column to predict (guessed if omitted)")
    profile.add_argument("--json", action="store_true", help="print JSON instead of a report")
    profile.add_argument("--no-leakage-check", action="store_true", help="skip the leakage check")

    fit = commands.add_parser("fit", help="train models on a CSV file and save the best one")
    fit.add_argument("csv", type=Path)
    fit.add_argument("--target", help="column to predict (guessed if omitted)")
    fit.add_argument("--time-budget", type=float, default=120.0, help="seconds (default 120)")
    fit.add_argument("--trials", type=int, default=100, help="max Optuna trials (default 100)")
    fit.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH, help="where to save the model")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--spec", type=Path, help="task spec JSON (from check-spec or by hand)")
    fit.add_argument("--describe", help="the task in plain words (with --llm)")
    fit.add_argument("--llm", choices=["anthropic"], help="ask an LLM for the task spec")
    fit.add_argument("--llm-model", help="model id for --llm (default: the provider's default)")

    prompt = commands.add_parser("prompt", help="write a prompt to paste into an LLM chat")
    prompt.add_argument("csv", type=Path)
    prompt.add_argument("--describe", required=True, help="the task in plain words")
    prompt.add_argument("--target", help="if you know the target: include its leakage checks")
    prompt.add_argument("--out", type=Path, help="file to write (default: print)")

    check = commands.add_parser("check-spec", help="check an LLM's answer against the data")
    check.add_argument("csv", type=Path)
    check.add_argument("answer", type=Path, help="the LLM's reply (JSON, fences allowed)")
    check.add_argument("--out", type=Path, default=DEFAULT_SPEC_PATH, help="where to save the spec")

    predict = commands.add_parser("predict", help="predict with a saved model")
    predict.add_argument("model", type=Path)
    predict.add_argument("csv", type=Path)
    predict.add_argument("--out", type=Path, help="CSV to write (default: print to stdout)")

    explain = commands.add_parser("explain", help="explain a saved model with SHAP")
    explain.add_argument("model", type=Path)
    explain.add_argument("csv", type=Path)
    explain.add_argument("--row", type=int, default=0, help="row to explain (0-based, default 0)")
    explain.add_argument("--top", type=int, default=10, help="columns to show (default 10)")
    explain.add_argument("--out-dir", type=Path, help="also save the charts as PNG here")

    args = parser.parse_args(argv)
    if args.command == "profile":
        return _profile(args)
    if args.command == "fit":
        return _fit(args)
    if args.command == "predict":
        return _predict(args)
    if args.command == "explain":
        return _explain(args)
    if args.command == "prompt":
        return _prompt(args)
    if args.command == "check-spec":
        return _check_spec(args)
    parser.print_help()
    return 0


def _profile(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    profile = profile_dataset(df)
    try:
        spec = infer_task_spec(
            df, target=args.target, profile=profile, check_leakage=not args.no_leakage_check
        )
        spec.check_against(df, profile)
    except TaskSpecError as error:
        print(profile.summary())
        print("\nCould not infer the task:", *error.problems, sep="\n  - ", file=sys.stderr)
        return 1

    if args.json:
        payload = {"profile": profile.to_dict(), "task": spec.model_dump(mode="json")}
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    print(profile.summary())
    _print_task(spec, plan_features(profile, spec))
    return 0


def _fit(args: argparse.Namespace) -> int:
    from automl.engine import AutoML  # heavy imports only when training

    df = pd.read_csv(args.csv)
    try:
        spec = _spec_for_fit(args, df)
        automl = AutoML(
            spec=spec, time_budget=args.time_budget, n_trials=args.trials, random_state=args.seed
        )
        automl.fit(df, target=args.target)
    except TaskSpecError as error:
        print("Could not set up the task:", *error.problems, sep="\n  - ", file=sys.stderr)
        return 1

    _print_task(automl.spec_, automl.plan_)
    sizes = automl.split_sizes_
    print(f"\nRows: {sizes['train']} train, {sizes['valid']} validation, {sizes['test']} test")
    print(f"Model selection: {automl.validation_} on train + validation, {automl.selection_}")
    print("\nLeaderboard (chosen by validation score; test score is the honest estimate):")
    print(automl.leaderboard_.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    path = automl.save(args.out)
    print(f"\nSaved {automl.model_.model_name} to {path}")
    return 0


def _spec_for_fit(args: argparse.Namespace, df: pd.DataFrame) -> TaskSpec | None:
    if args.spec:
        return TaskSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    if args.llm:
        from automl.llm import AnthropicLLM, propose_task_spec

        if not args.describe:
            raise TaskSpecError(["--llm needs --describe with the task in plain words"])
        llm = AnthropicLLM(**({"model": args.llm_model} if args.llm_model else {}))
        return propose_task_spec(df, args.describe, llm)
    return None


def _prompt(args: argparse.Namespace) -> int:
    from automl.llm import build_chat_prompt
    from automl.llm.propose import automatic_flags
    from automl.task import infer_problem_type

    df = pd.read_csv(args.csv)
    profile = profile_dataset(df)
    flags = None
    if args.target:
        if args.target not in df.columns:
            print(f"Column {args.target!r} is not in the data", file=sys.stderr)
            return 1
        problem_type, _ = infer_problem_type(df[args.target])
        flags = automatic_flags(df, args.target, problem_type.value, profile)
    text = build_chat_prompt(profile, args.describe, flags)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote the prompt to {args.out} ({len(text):,} characters).")
        print("Paste it into a Claude chat, save the reply as a file, then run:")
        print(f"  automl check-spec {args.csv} <reply file>")
    else:
        print(text)
    return 0


def _check_spec(args: argparse.Namespace) -> int:
    from automl.llm import check_answer

    df = pd.read_csv(args.csv)
    result = check_answer(args.answer.read_text(encoding="utf-8"), df)
    if not result.ok:
        print("The answer does not fit the data:", *result.problems, sep="\n  - ")
        print("\nSend this follow-up to the same chat, save the new reply and check it again:\n")
        print(result.repair_prompt())
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.spec.model_dump_json(indent=2), encoding="utf-8")
    profile = profile_dataset(df)
    _print_task(result.spec, plan_features(profile, result.spec))
    print(f"\nThe answer fits the data. Saved the spec to {args.out}; train with:")
    print(f"  automl fit {args.csv} --spec {args.out}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from automl.engine import load_model

    model = load_model(args.model)
    df = pd.read_csv(args.csv)
    try:
        result = pd.DataFrame({"prediction": model.predict(df)}, index=df.index)
        if model.classes is not None:
            proba = model.predict_proba(df).add_prefix("proba_")
            result = pd.concat([result, proba], axis=1)
    except ValueError as error:
        print(f"Cannot predict: {error}", file=sys.stderr)
        return 1

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Wrote {len(result)} predictions to {args.out}")
    else:
        print(result.to_csv(index=False), end="")
    return 0


def _explain(args: argparse.Namespace) -> int:
    from automl.engine import load_model
    from automl.explain import explain

    model = load_model(args.model)
    df = pd.read_csv(args.csv)
    if not 0 <= args.row < len(df):
        print(f"Row {args.row} is out of range: the file has {len(df)} rows", file=sys.stderr)
        return 1
    try:
        # The chosen row first, then the rest of the file as background for global importance.
        ordered = pd.concat([df.iloc[[args.row]], df.drop(index=df.index[args.row])])
        explanation = explain(model, ordered, max_rows=len(ordered))
    except ValueError as error:
        print(f"Cannot explain: {error}", file=sys.stderr)
        return 1

    print(f"Model: {model.model_name}; contributions in {explanation.units}")
    print(f"\nMost important columns ({len(explanation.data)} rows):")
    for column, score in explanation.importance().head(args.top).items():
        print(f"  {column:<24} {score:.4f}")

    print(f"\nRow {args.row}: {explanation.describe_row(0)}")
    for _, item in explanation.row(0).head(args.top).iterrows():
        print(f"  {item['column']:<24} {str(item['value'])[:24]:<24} {item['contribution']:+.4f}")

    if args.out_dir:
        from automl.plots import plot_importance, plot_row

        args.out_dir.mkdir(parents=True, exist_ok=True)
        plot_importance(explanation, top=args.top).savefig(args.out_dir / "importance.png")
        plot_row(explanation, 0, top=args.top).savefig(args.out_dir / f"row_{args.row}.png")
        print(f"\nCharts saved to {args.out_dir}")
    return 0


def _print_task(spec: TaskSpec, plan: FeaturePlan) -> None:
    print(f"\nTask: predict {spec.target!r} ({spec.problem_type.value}), metric {spec.metric}")
    print(
        f"Features: {len(plan.numeric)} numeric, {len(plan.categorical)} categorical, "
        f"{len(plan.datetime)} date"
    )
    if plan.ignored:
        print("Not used:")
        for column, reason in plan.ignored.items():
            print(f"  - {column}: {reason}")
    if spec.notes:
        print("Notes:")
        for note in spec.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    sys.exit(main())
