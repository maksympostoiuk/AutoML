"""From an LLM answer to a TaskSpec the code trusts.

An answer is accepted only after the same checks as a hand-written spec, plus one more: after
the target is known, the automatic leakage checks (stage 2) run for that target, and every
column they flag must be judged explicitly, dropped or kept with a reason. The LLM supplies
the judgement the statistics cannot (what a column *means*); the code makes sure no flagged
column is silently ignored.

``propose_task_spec`` runs the whole loop with an API provider. The chat workflow uses the same
pieces by hand: ``build_chat_prompt`` -> answer pasted into a file -> ``check_answer``, which
either returns a TaskSpec or a follow-up prompt with the problems.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd
from pydantic import ValidationError

from automl.data import (
    DatasetProfile,
    find_leakage_suspects,
    find_missingness_warnings,
    profile_dataset,
)
from automl.llm.prompts import SYSTEM, build_repair_prompt, build_task_prompt
from automl.llm.providers import LLM
from automl.llm.schema import TaskProposal, proposal_json_schema
from automl.task import TaskSpec, TaskSpecError

MAX_ATTEMPTS = 3


@dataclass
class CheckResult:
    spec: TaskSpec | None  # set when the answer is accepted
    problems: list[str]  # otherwise: everything that is wrong with it

    @property
    def ok(self) -> bool:
        return self.spec is not None

    def repair_prompt(self) -> str:
        return build_repair_prompt(self.problems)


def extract_json(text: str) -> dict:
    """The JSON object in an answer, also when wrapped in ```json fences or surrounded by text."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the answer")
    return json.loads(candidate[start : end + 1])


def check_answer(
    answer: str, df: pd.DataFrame, profile: DatasetProfile | None = None
) -> CheckResult:
    """Validate an LLM answer against the schema, the data and the leakage checks."""
    profile = profile or profile_dataset(df)
    try:
        proposal = TaskProposal.model_validate(extract_json(answer))
        spec = proposal.to_task_spec()
    except (ValueError, ValidationError) as error:
        return CheckResult(None, [f"the answer is not valid: {_short(error)}"])

    problems: list[str] = []
    unknown = [
        d.column
        for d in proposal.drop_columns + proposal.keep_flagged_columns
        if d.column not in profile
    ]
    if unknown:
        problems.append(f"these columns do not exist: {unknown}")
    try:
        spec.check_against(df, profile)
    except TaskSpecError as error:
        problems += error.problems
    if problems:
        return CheckResult(None, problems)

    decided = {d.column for d in proposal.drop_columns + proposal.keep_flagged_columns}
    undecided = {
        column: reason
        for column, reason in automatic_flags(df, spec.target, spec.problem_type, profile).items()
        if column not in decided
    }
    if undecided:
        return CheckResult(
            None,
            [
                f"column {column!r} was flagged by an automatic check ({reason}); put it in "
                "drop_columns if its value is only known after the outcome, otherwise in "
                "keep_flagged_columns with the reason"
                for column, reason in undecided.items()
            ],
        )
    return CheckResult(spec, [])


def automatic_flags(
    df: pd.DataFrame, target: str, problem_type: str, profile: DatasetProfile
) -> dict[str, str]:
    """Columns the stage-2 checks find suspicious for this target, with the reason."""
    flags = {s.column: s.reason for s in find_leakage_suspects(df, target, problem_type, profile)}
    for warning in find_missingness_warnings(df, target, problem_type, profile):
        flags.setdefault(warning.column, warning.message)
    return flags


def propose_task_spec(
    df: pd.DataFrame,
    description: str,
    llm: LLM,
    profile: DatasetProfile | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> TaskSpec:
    """Ask the LLM for a TaskSpec; on problems, send them back and ask again."""
    profile = profile or profile_dataset(df)
    messages = [{"role": "user", "content": build_task_prompt(profile, description)}]
    problems: list[str] = []
    for _ in range(max_attempts):
        answer = llm.complete(SYSTEM, messages, proposal_json_schema())
        result = check_answer(answer, df, profile)
        if result.ok:
            return result.spec  # type: ignore[return-value]
        problems = result.problems
        messages += [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": result.repair_prompt()},
        ]
    raise TaskSpecError([f"no valid proposal after {max_attempts} attempts", *problems])


def _short(error: Exception, limit: int = 300) -> str:
    text = " ".join(str(error).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
