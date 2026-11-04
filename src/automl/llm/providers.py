"""LLM providers behind one small interface.

* ``AnthropicLLM``: the Claude API with structured outputs (needs ``ANTHROPIC_API_KEY`` in the
  environment and ``pip install "automl[llm]"``). The key is never stored in the project.
* ``RecordedLLM``: replays prepared answers. Tests and CI use it: no network, no key.

The chat workflow needs no provider at all: the prompt is printed, the user pastes it into a
chat window and saves the answer to a file (see ``automl prompt`` / ``automl check-spec``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict[str, str]], schema: dict) -> str:
        """Return the model's answer text: a JSON object matching ``schema``."""
        ...


class RecordedLLM:
    """Returns the given answers in order and records what it was asked."""

    def __init__(self, answers: Sequence[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, system: str, messages: list[dict[str, str]], schema: dict) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.answers:
            raise RuntimeError("RecordedLLM has no answers left")
        return self.answers.pop(0)


class AnthropicLLM:
    """Claude through the official SDK, with the answer constrained to the JSON schema.

    Server-side fallbacks are on: if the model declines a request, the API retries it on a
    fallback model in the same call.
    """

    def __init__(self, model: str = DEFAULT_MODEL, client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as error:
                raise RuntimeError(
                    'The Anthropic SDK is not installed: pip install "automl[llm]"'
                ) from error
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.client = client
        self.model = model

    def complete(self, system: str, messages: list[dict[str, str]], schema: dict) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined the request")
        return next(block.text for block in response.content if block.type == "text")
