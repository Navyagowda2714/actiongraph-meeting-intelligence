from __future__ import annotations

import json
from datetime import date

from actiongraph.extract.chunking import chunk_turns
from actiongraph.extract.pipeline import ExtractionPipeline, quote_supported
from actiongraph.ingest import load_transcript
from actiongraph.llm import ScriptedLLM, extract_json, get_llm
from actiongraph.llm.offline import OfflineProvider
from actiongraph.models import Transcript

from .conftest import SAMPLES, make_turns


def _transcript(*pairs) -> Transcript:
    return Transcript(
        meeting_id="m", title="T", meeting_date=date(2026, 3, 2), turns=make_turns(*pairs)
    )


def test_hallucinated_evidence_is_dropped_and_owner_resolved():
    t = _transcript(("Ana Ruiz", "I'll send the deck by Friday."), ("Ben Ito", "Thanks."))
    reply = json.dumps(
        {
            "action_items": [
                {
                    "description": "Send the deck",
                    "owner": "speaker",
                    "deadline_text": "by Friday",
                    "evidence": [{"turn_id": "t001", "quote": "I'll send the deck by Friday."}],
                },
                {
                    "description": "Book a venue",
                    "owner": "Ben",
                    "evidence": [{"turn_id": "t002", "quote": "I will book the venue tomorrow"}],
                },
            ],
            "decisions": [],
            "risks": [],
            "summary": "Short sync.",
        }
    )
    a = ExtractionPipeline(ScriptedLLM([reply])).run(t)
    assert [i.description for i in a.action_items] == ["Send the deck"]
    item = a.action_items[0]
    assert item.owner == "Ana Ruiz" and item.owner_resolution == "speaker_self"
    assert item.deadline == date(2026, 3, 6)
    assert a.usage.dropped_unsupported == 1 and a.usage.raw_evidence_valid == 1
    assert a.summary == "Short sync."


def test_invalid_json_triggers_one_retry_with_feedback():
    t = _transcript(("Ana Ruiz", "I'll send the deck."))
    good = json.dumps(
        {
            "action_items": [
                {
                    "description": "Send the deck",
                    "owner": "Ana",
                    "evidence": [{"turn_id": "t001", "quote": "send the deck"}],
                }
            ]
        }
    )
    llm = ScriptedLLM(["not json at all", good])
    a = ExtractionPipeline(llm).run(t)
    assert len(llm.calls) == 2 and "invalid" in llm.calls[1][-1].content
    assert a.action_items[0].owner == "Ana Ruiz" and a.usage.llm_calls == 2


def test_review_flags_for_unowned_unresolved_deadline_and_outsider():
    t = _transcript(("Ana Ruiz", "Someone should fix the build after the next sync."))
    reply = json.dumps(
        {
            "action_items": [
                {
                    "description": "Fix the build",
                    "owner": None,
                    "deadline_text": "after the next sync",
                    "confidence": 0.4,
                    "evidence": [{"turn_id": "t001", "quote": "fix the build"}],
                },
                {
                    "description": "Review the release checklist",
                    "owner": "Zed",
                    "confidence": 0.9,
                    "evidence": [{"turn_id": "t001", "quote": "Someone should fix the build"}],
                },
            ]
        }
    )
    a = ExtractionPipeline(ScriptedLLM([reply])).run(t)
    first, second = a.action_items
    assert first.needs_review and any("no owner" in r for r in first.review_reasons)
    assert any("could not be resolved" in r for r in first.review_reasons)
    assert any("not a meeting participant" in r for r in second.review_reasons)


def test_chunked_run_matches_single_pass_on_long_meeting():
    t = load_transcript(SAMPLES / "atlas_sync_2026-03-02.vtt")
    single = ExtractionPipeline(OfflineProvider(), chunk_max_tokens=100_000).run(t)
    chunked = ExtractionPipeline(OfflineProvider(), chunk_max_tokens=250, overlap_turns=2).run(t)
    assert chunked.usage.chunks > 3

    def key(a):
        return sorted(
            (str(i.owner), str(i.deadline), i.evidence[0].turn_id) for i in a.action_items
        )

    assert key(chunked) == key(single)


def test_chunk_overlap_carries_context_turns():
    turns = make_turns(*[("A", "word " * 50)] * 6)
    chunks = chunk_turns(turns, max_tokens=170, overlap_turns=1)
    assert len(chunks) == 3 and chunks[1].context == [chunks[0].focus[-1]]


def test_quote_supported_tolerates_case_and_punctuation_only():
    assert quote_supported("ill SEND the deck", "I'll send the deck by Friday.")
    assert not quote_supported("I'll cancel the launch", "I'll send the deck by Friday.")


def test_llm_factory_falls_back_to_offline_without_keys(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_llm(), OfflineProvider)
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
