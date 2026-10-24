from __future__ import annotations

import json
from datetime import date

import pytest

from actiongraph.ingest import load_transcript, transcript_from_bytes
from actiongraph.ingest.audio import assign_speakers
from actiongraph.ingest.parsers import (
    Segment,
    parse_srt,
    parse_timestamp,
    parse_vtt,
    segments_to_turns,
)

from .conftest import FIXTURES, SAMPLES


def test_vtt_voice_tags_metadata_and_same_speaker_merge():
    t = load_transcript(SAMPLES / "atlas_sync_2026-03-02.vtt")
    assert t.meeting_date == date(2026, 3, 2)
    assert t.series == "atlas-checkout"
    # Priya's two consecutive cues are merged into one turn
    t004 = t.turn("t004")
    assert t004 is not None and t004.speaker == "Priya Raman"
    assert "eighty percent" in t004.text and "mockups ready by Wednesday" in t004.text
    assert t004.start is not None and t004.end is not None and t004.end > t004.start
    assert [x.id for x in t.turns[:3]] == ["t001", "t002", "t003"]


def test_srt_with_diarisation_sidecar_speaker_map():
    t = load_transcript(SAMPLES / "heldout_incident_review_2026-04-07.srt")
    assert "Hannah Weiss" in t.participants and not any(
        p.startswith("SPEAKER_") for p in t.participants
    )
    assert t.meeting_date == date(2026, 4, 7)  # inferred from the filename


def test_txt_with_timestamps_and_unlabelled_continuation():
    raw = (
        b"[00:00:01] Ana Ruiz: First point.\ncontinued on the next line\n[00:00:09] Ben Ito: Reply."
    )
    t = transcript_from_bytes(raw, "notes.txt", meeting_date=date(2026, 1, 5))
    assert len(t.turns) == 2
    assert t.turns[0].text == "First point. continued on the next line"
    assert t.turns[1].start == 9


def test_json_speaker_turn_format():
    doc = {
        "title": "X",
        "meeting_date": "2026-02-01",
        "turns": [
            {"speaker": "A", "text": "hi"},
            {"speaker": "B", "text": "hello"},
            {"speaker": "B", "text": "again"},
        ],
    }
    t = transcript_from_bytes(json.dumps(doc).encode(), "m.json")
    assert t.title == "X" and len(t.turns) == 2 and t.turns[1].text == "hello again"


def test_pdf_transcript_via_pypdf():
    t = load_transcript(FIXTURES / "vendor_review.pdf")
    assert t.source_format == "pdf"
    assert t.meeting_date == date(2026, 2, 10)
    assert [x.speaker for x in t.turns] == ["Maya Chen", "Daniel Okafor", "Maya Chen"]


def test_unsupported_format_and_empty_transcript_rejected():
    with pytest.raises(ValueError):
        transcript_from_bytes(b"data", "slides.pptx")
    with pytest.raises(ValueError):
        transcript_from_bytes(b"", "empty.txt")


def test_cue_parsing_without_voice_tags_and_timestamps():
    doc = parse_srt("1\n00:00:01,500 --> 00:00:03,000\nSPEAKER_01: Hello there\n")
    assert doc.segments[0].speaker == "SPEAKER_01" and doc.segments[0].start == 1.5
    assert parse_timestamp("01:02:03.250") == pytest.approx(3723.25)
    assert (
        parse_vtt("WEBVTT\n\n00:01.000 --> 00:02.000\n<v Kim>Hi</v>\n").segments[0].speaker == "Kim"
    )


def test_assign_speakers_by_max_overlap_then_merge():
    asr = [
        Segment("Unknown", "hello", 0.0, 2.0),
        Segment("Unknown", "and more", 2.0, 4.0),
        Segment("Unknown", "reply", 4.1, 6.0),
    ]
    diar = [(0.0, 3.9, "SPEAKER_00"), (3.9, 7.0, "SPEAKER_01")]
    labelled = assign_speakers(asr, diar)
    turns = segments_to_turns(labelled, {"SPEAKER_00": "Ana", "SPEAKER_01": "Ben"})
    assert [(t.speaker, t.text) for t in turns] == [("Ana", "hello and more"), ("Ben", "reply")]
