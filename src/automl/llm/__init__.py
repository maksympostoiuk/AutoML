"""LLM layer: turn a plain-language task description into a checked TaskSpec."""

from automl.llm.prompts import build_chat_prompt, build_repair_prompt, build_task_prompt
from automl.llm.propose import CheckResult, check_answer, extract_json, propose_task_spec
from automl.llm.providers import DEFAULT_MODEL, LLM, AnthropicLLM, RecordedLLM
from automl.llm.schema import ColumnDecision, TaskProposal

__all__ = [
    "DEFAULT_MODEL",
    "LLM",
    "AnthropicLLM",
    "CheckResult",
    "ColumnDecision",
    "RecordedLLM",
    "TaskProposal",
    "build_chat_prompt",
    "build_repair_prompt",
    "build_task_prompt",
    "check_answer",
    "extract_json",
    "propose_task_spec",
]
