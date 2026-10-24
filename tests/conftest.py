from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from actiongraph.extract.rules import RuleBasedExtractor
from actiongraph.models import Turn

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
GOLD = ROOT / "data" / "gold"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the network: force the offline provider."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "offline")


def make_turns(*pairs: tuple[str, str]) -> list[Turn]:
    return [Turn(id=f"t{i:03d}", speaker=s, text=t) for i, (s, t) in enumerate(pairs, 1)]


@pytest.fixture
def extract():
    """extract(("Speaker", "text"), ...) -> RawExtraction, meeting on Monday 2026-03-02."""

    def _run(*pairs: tuple[str, str], meeting_date: date = date(2026, 3, 2)):
        turns = make_turns(*pairs)
        participants = list(dict.fromkeys(t.speaker for t in turns))
        return RuleBasedExtractor(participants, meeting_date).extract(turns)

    return _run
