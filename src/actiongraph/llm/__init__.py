"""LLM layer: provider protocol, hosted providers, offline stand-ins and a factory."""

from __future__ import annotations

import logging
import os

from actiongraph.llm.base import LLMClient, LLMResponse, Message, extract_json
from actiongraph.llm.offline import OfflineProvider, ScriptedLLM

log = logging.getLogger(__name__)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "OfflineProvider",
    "ScriptedLLM",
    "extract_json",
    "get_llm",
]


def get_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build the provider named by `provider` / env `LLM_PROVIDER`.

    Falls back to the offline provider (with a warning) when the selected hosted provider has no
    API key or its SDK is not installed.
    """
    name = (provider or os.getenv("LLM_PROVIDER") or "auto").lower()
    if name == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "offline"
    if name == "offline":
        return OfflineProvider()
    try:
        if name == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            from actiongraph.llm.providers import AnthropicProvider

            return AnthropicProvider(model=model)
        if name == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            from actiongraph.llm.providers import OpenAIProvider

            return OpenAIProvider(model=model)
        if name == "ollama":
            from actiongraph.llm.providers import OllamaProvider

            return OllamaProvider(model=model)
    except RuntimeError as exc:
        log.warning("LLM provider %r unavailable (%s); falling back to offline rules.", name, exc)
        return OfflineProvider()
    raise ValueError(f"Unknown LLM provider: {name!r}")
