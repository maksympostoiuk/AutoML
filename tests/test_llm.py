import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from automl.cli import main
from automl.data import profile_dataset
from automl.llm import (
    AnthropicLLM,
    RecordedLLM,
    TaskProposal,
    build_chat_prompt,
    check_answer,
    extract_json,
    propose_task_spec,
)
from automl.llm.schema import proposal_json_schema
from automl.task import TaskSpecError

DESCRIPTION = "Predict which customers will cancel their contract."


def answer(**overrides: object) -> str:
    """A good answer for the churn table; override fields to make it wrong."""
    proposal = {
        "target": "churned",
        "problem_type": "binary",
        "metric": "roc_auc",
        "drop_columns": [
            {"column": "cancellation_date", "reason": "Only known after the customer left."}
        ],
        "keep_flagged_columns": [],
        "reasoning": "Churn is a yes/no outcome; the cancellation date is recorded after it.",
    }
    proposal.update(overrides)
    return json.dumps(proposal)


def test_schema_is_strict_enough_for_structured_outputs() -> None:
    schema = proposal_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    decision = schema["$defs"]["ColumnDecision"]
    assert decision["additionalProperties"] is False


def test_prompt_contains_the_profile_not_the_rows(churn_df: pd.DataFrame) -> None:
    prompt = build_chat_prompt(profile_dataset(churn_df), DESCRIPTION)
    assert DESCRIPTION in prompt
    assert '"name": "cancellation_date"' in prompt
    assert '"keep_flagged_columns"' in prompt  # the JSON schema is included
    # Free-text notes are summarised by up to five examples, not all 400 rows.
    assert prompt.count("Customer called about billing issue") <= 5


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here is the proposal:\n```\n{"a": 1}\n```\nHope this helps.',
        'Sure! {"a": 1}',
    ],
)
def test_extract_json(text: str) -> None:
    assert extract_json(text) == {"a": 1}


def test_a_good_answer_becomes_a_task_spec(churn_df: pd.DataFrame) -> None:
    result = check_answer(answer(), churn_df)
    assert result.ok
    assert result.spec.target == "churned"
    assert "cancellation_date" in result.spec.drop_columns
    assert any("LLM" in note for note in result.spec.notes)


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"target": "churn"}, "not in the data"),
        ({"problem_type": "regression", "metric": "rmse"}, "numeric"),
        ({"metric": "rmse"}, "does not fit binary"),
        ({"drop_columns": [{"column": "nope", "reason": "x"}]}, "do not exist"),
        ({"target": "churned", "drop_columns": [{"column": "churned", "reason": "x"}]}, "target"),
    ],
)
def test_wrong_answers_are_rejected_with_reasons(
    churn_df: pd.DataFrame, overrides: dict, problem: str
) -> None:
    result = check_answer(answer(**overrides), churn_df)
    assert not result.ok
    assert problem in " ".join(result.problems)
    assert problem in result.repair_prompt()


def test_flagged_columns_must_be_judged(churn_df: pd.DataFrame) -> None:
    """The leak the statistics found cannot be ignored: drop it or keep it with a reason."""
    result = check_answer(answer(drop_columns=[]), churn_df)
    assert not result.ok
    assert "cancellation_date" in result.repair_prompt()

    kept = [{"column": "cancellation_date", "reason": "Known in advance in this business."}]
    result = check_answer(answer(drop_columns=[], keep_flagged_columns=kept), churn_df)
    assert result.ok
    assert "cancellation_date" not in result.spec.drop_columns
    assert any("Kept flagged column 'cancellation_date'" in n for n in result.spec.notes)


def test_not_json_at_all(churn_df: pd.DataFrame) -> None:
    result = check_answer("I think the target is churned.", churn_df)
    assert not result.ok
    assert "not valid" in result.problems[0]


def test_propose_retries_with_the_problems(churn_df: pd.DataFrame) -> None:
    llm = RecordedLLM([answer(target="churn"), answer()])
    spec = propose_task_spec(churn_df, DESCRIPTION, llm)
    assert spec.target == "churned"
    second_call = llm.calls[1]
    assert [m["role"] for m in second_call] == ["user", "assistant", "user"]
    assert "not in the data" in second_call[-1]["content"]


def test_propose_gives_up_after_max_attempts(churn_df: pd.DataFrame) -> None:
    llm = RecordedLLM([answer(target="nope")] * 3)
    with pytest.raises(TaskSpecError, match="no valid proposal after 3 attempts"):
        propose_task_spec(churn_df, DESCRIPTION, llm)


def test_anthropic_provider_requests_the_schema() -> None:
    """The API call itself, with a stand-in client: no network, no key."""
    sent: dict = {}

    def create(**kwargs: object) -> SimpleNamespace:
        sent.update(kwargs)
        block = SimpleNamespace(type="text", text=answer())
        return SimpleNamespace(stop_reason="end_turn", content=[block])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    llm = AnthropicLLM(client=client)
    text = llm.complete("system", [{"role": "user", "content": "hi"}], proposal_json_schema())
    assert TaskProposal.model_validate_json(text).target == "churned"
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["fallbacks"] == "default"


def test_anthropic_provider_reports_a_refusal() -> None:
    response = SimpleNamespace(stop_reason="refusal", content=[])
    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))
    )
    with pytest.raises(RuntimeError, match="declined"):
        AnthropicLLM(client=client).complete("s", [], {})


# ----- the chat workflow through the CLI ------------------------------------------------------


@pytest.fixture
def churn_csv(churn_df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "churn.csv"
    churn_df.to_csv(path, index=False)
    return path


def test_chat_workflow(churn_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompt = tmp_path / "prompt.md"
    assert main(["prompt", str(churn_csv), "--describe", DESCRIPTION, "--out", str(prompt)]) == 0
    assert "keep_flagged_columns" in prompt.read_text(encoding="utf-8")

    # A first reply that forgets the leak: rejected, with a follow-up prompt to send back.
    reply = tmp_path / "reply.json"
    reply.write_text(answer(drop_columns=[]), encoding="utf-8")
    assert main(["check-spec", str(churn_csv), str(reply)]) == 1
    assert "cancellation_date" in capsys.readouterr().out

    # The corrected reply is accepted and saved as a spec the fit command uses.
    reply.write_text(f"```json\n{answer()}\n```", encoding="utf-8")
    spec = tmp_path / "spec.json"
    assert main(["check-spec", str(churn_csv), str(reply), "--out", str(spec)]) == 0
    args = ["fit", str(churn_csv), "--spec", str(spec), "--time-budget", "3", "--trials", "2"]
    assert main([*args, "--out", str(tmp_path / "model.joblib")]) == 0
    out = capsys.readouterr().out
    assert "cancellation_date: LLM: Only known after the customer left." in out


def test_prompt_with_target_includes_the_flags(churn_csv: Path, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.md"
    args = ["prompt", str(churn_csv), "--describe", DESCRIPTION, "--target", "churned"]
    assert main([*args, "--out", str(prompt)]) == 0
    text = prompt.read_text(encoding="utf-8")
    assert "Flagged by automatic checks:\n- cancellation_date" in text
