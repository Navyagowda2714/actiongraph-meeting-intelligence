"""Deterministic offline stand-in for an LLM.

`OfflineProvider` speaks the same prompt/JSON contract as a real model: it reads the turns
embedded in the extraction prompt, runs the rule-based extractor over them and returns JSON
that validates against `RawExtraction`. That keeps the whole pipeline (prompting, parsing,
evidence validation, owner/deadline normalisation, merging) on one code path for tests and demos.

It is a heuristic baseline, not a language model: quality numbers produced with it are rule-engine
numbers, and real quality requires a real model.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from typing import Any

from actiongraph.llm.base import LLMResponse, Message, approx_tokens

_BLOCK = re.compile(r"<turns_json>\s*(.*?)\s*</turns_json>", re.DOTALL)


class OfflineProvider:
    model = "offline-rules-v1"

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        from actiongraph.extract.rules import RuleBasedExtractor
        from actiongraph.models import Turn

        t0 = time.perf_counter()
        prompt = messages[-1].content
        m = _BLOCK.search(prompt)
        if not m:
            text = json.dumps({"action_items": [], "decisions": [], "risks": [], "summary": ""})
        else:
            payload = json.loads(m.group(1))
            turns = [Turn(**t) for t in payload["turns"]]
            extractor = RuleBasedExtractor(
                participants=payload.get("participants", []),
                meeting_date=date.fromisoformat(payload["meeting_date"]),
            )
            raw = extractor.extract(turns, focus_ids=set(payload.get("focus_turn_ids") or []))
            text = raw.model_dump_json()
        in_tok = approx_tokens((system or "") + prompt)
        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=approx_tokens(text),
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self.model,
        )


class ScriptedLLM:
    """Returns pre-baked replies in order; used in tests to simulate model behaviour/failures."""

    model = "scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else "{}"
        return LLMResponse(
            text=text,
            input_tokens=approx_tokens(messages[-1].content),
            output_tokens=approx_tokens(text),
            model=self.model,
        )
