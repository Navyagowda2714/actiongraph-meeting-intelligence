"""Provider-agnostic LLM interface."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface every provider implements."""

    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def schema_instruction(json_schema: dict[str, Any]) -> str:
    return (
        "Respond with a single JSON object only (no prose, no markdown fences) that validates "
        "against this JSON schema:\n" + json.dumps(json_schema, separators=(",", ":"))
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating markdown fences and leading prose."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~0.75 words per token) used for budgeting and offline accounting."""
    return max(1, int(len(text.split()) * 1.33))
