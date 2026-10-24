"""Runtime configuration via environment variables (pydantic-settings)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "auto"  # auto | anthropic | openai | ollama | offline
    llm_model: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    db_path: Path = Path("actiongraph.db")
    chunk_max_tokens: int = 1500
    chunk_overlap_turns: int = 2
    review_confidence_threshold: float = 0.6
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str | None = None) -> None:
    lvl = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
