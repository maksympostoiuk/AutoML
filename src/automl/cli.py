"""Command-line interface. Commands are added stage by stage (see docs/PLAN.md).

automl profile data.csv                 profile the table and infer the task
automl profile data.csv --target churn  same, with the target named explicitly
automl profile data.csv --json          machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from automl import __version__
from automl.data import profile_dataset
from automl.preprocess import plan_features
from automl.task import TaskSpecError, infer_task_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="automl", description="AutoML for tabular data.")
    parser.add_argument("--version", action="version", version=f"automl {__version__}")
    commands = parser.add_subparsers(dest="command")

    profile = commands.add_parser("profile", help="profile a CSV file and infer the task")
    profile.add_argument("csv", type=Path)
    profile.add_argument("--target", help="column to predict (guessed if omitted)")
    profile.add_argument("--json", action="store_true", help="print JSON instead of a report")
    profile.add_argument("--no-leakage-check", action="store_true", help="skip the leakage check")

    args = parser.parse_args(argv)
    if args.command == "profile":
        return _profile(args)
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

    plan = plan_features(profile, spec)
    print(profile.summary())
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
