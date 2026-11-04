"""What the LLM is asked to return: a task proposal, as a strict JSON schema.

The proposal is deliberately a separate model from ``TaskSpec``: it asks the LLM for explicit
decisions (which columns to drop *and* which flagged columns to keep, each with a reason),
and it uses only JSON shapes that structured-output APIs accept (lists of objects instead of a
free-form dict, no optional fields, no extra keys).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from automl.task import TaskSpec

MetricName = Literal["roc_auc", "log_loss", "f1", "accuracy", "f1_macro", "rmse", "mae", "r2"]


class ColumnDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Exact column name from the profile.")
    reason: str = Field(description="One short sentence.")


class TaskProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(description="The column to predict, exactly as named in the profile.")
    problem_type: Literal["binary", "multiclass", "regression"]
    metric: MetricName = Field(
        description="binary: roc_auc, log_loss, f1 or accuracy; multiclass: log_loss, "
        "f1_macro or accuracy; regression: rmse, mae or r2."
    )
    drop_columns: list[ColumnDecision] = Field(
        description="Columns that must not be used as features, above all columns whose value "
        "is only known after the outcome (leakage)."
    )
    keep_flagged_columns: list[ColumnDecision] = Field(
        description="Columns flagged as suspicious that are nevertheless safe to use, "
        "with the reason they are known at prediction time."
    )
    reasoning: str = Field(description="Two or three sentences explaining the choices.")

    def to_task_spec(self) -> TaskSpec:
        notes = [f"Proposed by an LLM: {self.reasoning}"]
        notes += [
            f"Kept flagged column {d.column!r}: {d.reason}" for d in self.keep_flagged_columns
        ]
        return TaskSpec(
            target=self.target,
            problem_type=self.problem_type,
            metric=self.metric,
            drop_columns={d.column: f"LLM: {d.reason}" for d in self.drop_columns},
            notes=notes,
        )


def proposal_json_schema() -> dict:
    return TaskProposal.model_json_schema()
