"""Hosted / local LLM providers. SDK imports are lazy so the package works without extras."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from actiongraph.llm.base import LLMResponse, Message, schema_instruction


def _system_with_schema(system: str | None, json_schema: dict[str, Any] | None) -> str | None:
    if json_schema is None:
        return system
    return ((system or "") + "\n\n" + schema_instruction(json_schema)).strip()


class AnthropicProvider:
    """Claude via the official `anthropic` SDK (pip install actiongraph[anthropic])."""

    def __init__(
        self, model: str | None = None, api_key: str | None = None, max_tokens: int = 4096
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Install the 'anthropic' extra: pip install actiongraph[anthropic]"
            ) from exc
        self.model = model or os.getenv("LLM_MODEL") or "claude-sonnet-5-5"
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        sys_prompt = _system_with_schema(system, json_schema)
        if sys_prompt:
            kwargs["system"] = sys_prompt
        resp = self._client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self.model,
        )


class OpenAIProvider:
    """OpenAI chat completions (pip install actiongraph[openai])."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Install the 'openai' extra: pip install actiongraph[openai]"
            ) from exc
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        msgs: list[dict[str, str]] = []
        sys_prompt = _system_with_schema(system, json_schema)
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs += [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs, "temperature": temperature}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self.model,
        )


class OllamaProvider:
    """Local models through the Ollama HTTP API."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 120):
        self.model = model or os.getenv("LLM_MODEL") or "llama3.1"
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        msgs: list[dict[str, str]] = []
        sys_prompt = _system_with_schema(system, json_schema)
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs += [{"role": m.role, "content": m.content} for m in messages]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            payload["format"] = "json"
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self.model,
        )
