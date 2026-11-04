"""Prompts for proposing a TaskSpec.

The model sees the dataset *profile* (column types, missing shares, up to five example values
per column), never the rows. The same text is used by every provider: sent to the API, or
printed for the user to paste into a chat window.
"""

from __future__ import annotations

import json

from automl.data import DatasetProfile
from automl.llm.schema import proposal_json_schema

SYSTEM = """\
You are a senior data scientist setting up a supervised machine learning task on a table.
You receive a profile of the table (column names, types, missing values, example values) and
the user's description of what they want to predict. Decide:

1. target: the column to predict. It must be one of the profile's columns.
2. problem_type: binary (two classes), multiclass (3-50 classes) or regression (a quantity).
3. metric: the metric to compare models by. Prefer roc_auc for binary, log_loss for
   multiclass and rmse for regression, unless the description asks for something else.
4. drop_columns: columns that must not be used as features. Above all, drop LEAKAGE:
   columns whose value is only known after the outcome happened, or is a consequence of it
   (a cancellation date when predicting churn, a lifeboat or body identification number when
   predicting survival, a final grade when predicting a pass). Also drop columns that would
   not be available when a prediction is made. Judge by what each column means, not only by
   its statistics. Identifier, free-text and constant columns are removed automatically;
   list them only if you have a further reason.
5. keep_flagged_columns: for every column listed under "Flagged by automatic checks" that you
   decide NOT to drop, say why it is known at prediction time. Every flagged column must
   appear in exactly one of drop_columns and keep_flagged_columns.

Use exact column names. Do not invent columns. Keep reasons to one short sentence."""


def build_task_prompt(
    profile: DatasetProfile, description: str, flags: dict[str, str] | None = None
) -> str:
    """The user message: description, profile and (optionally) automatic-check flags."""
    parts = [
        f"What the user wants to predict:\n{description.strip()}",
        "Table profile (JSON):\n" + json.dumps(profile.to_dict(), ensure_ascii=False, default=str),
    ]
    if flags:
        lines = "\n".join(f"- {column}: {reason}" for column, reason in flags.items())
        parts.append(f"Flagged by automatic checks:\n{lines}")
    return "\n\n".join(parts)


def build_chat_prompt(
    profile: DatasetProfile, description: str, flags: dict[str, str] | None = None
) -> str:
    """One self-contained message for a chat window (no API, no system prompt field)."""
    schema = json.dumps(proposal_json_schema(), indent=2)
    return (
        f"{SYSTEM}\n\n{build_task_prompt(profile, description, flags)}\n\n"
        "Reply with a single JSON object that matches this JSON schema, and nothing else "
        f"(no explanations outside the JSON):\n{schema}\n"
    )


def build_repair_prompt(problems: list[str]) -> str:
    """Follow-up message when the proposal does not fit the data."""
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Your proposal was checked against the data and has these problems:\n"
        f"{listed}\n\n"
        "Reply with a corrected JSON object in the same format, and nothing else."
    )
