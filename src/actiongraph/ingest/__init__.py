"""Ingestion entry points: any supported file -> normalised `Transcript`."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from actiongraph.ingest.audio import DiarizationTrack, Transcriber, assign_speakers
from actiongraph.ingest.parsers import (
    AUDIO_EXTS,
    ParsedDocument,
    Segment,
    parse_bytes,
    parse_meeting_date,
    parse_pdf,
    segments_to_turns,
)
from actiongraph.models import Transcript

__all__ = ["Segment", "load_transcript", "transcript_from_bytes"]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:60] or "meeting"


def _build(
    doc: ParsedDocument,
    *,
    name: str,
    title: str | None,
    meeting_date: date | None,
    series: str | None,
    speaker_map: dict[str, str] | None,
) -> Transcript:
    turns = segments_to_turns(doc.segments, speaker_map)
    if not turns:
        raise ValueError("Transcript contains no speaker turns")
    name_date = re.search(r"\d{4}-\d{2}-\d{2}", name)
    mdate = (
        meeting_date
        or parse_meeting_date(doc.metadata.get("meeting_date"))
        or (parse_meeting_date(name_date.group(0)) if name_date else None)
        or date.today()
    )
    ttl = title or doc.metadata.get("title") or name
    digest = hashlib.sha1(" ".join(t.text for t in turns).encode()).hexdigest()[:6]
    return Transcript(
        meeting_id=f"{_slug(Path(name).stem)}-{digest}",
        title=ttl,
        meeting_date=mdate,
        series=series or doc.metadata.get("series") or "default",
        turns=turns,
        source_format=doc.source_format,
    )


def load_transcript(
    path: str | Path,
    *,
    title: str | None = None,
    meeting_date: date | None = None,
    series: str | None = None,
    speaker_map: dict[str, str] | None = None,
    transcriber: Transcriber | None = None,
    diarization: DiarizationTrack | None = None,
) -> Transcript:
    """Load a transcript (or transcribe audio/video) from disk.

    A sidecar `<stem>.speakers.json` mapping diarisation labels to names is picked up automatically.
    """
    p = Path(path)
    sidecar = p.with_name(p.stem + ".speakers.json")
    if speaker_map is None and sidecar.exists():
        speaker_map = json.loads(sidecar.read_text())
    ext = p.suffix.lower()
    if ext in AUDIO_EXTS:
        if transcriber is None:
            from actiongraph.ingest.audio import FasterWhisperTranscriber

            transcriber = FasterWhisperTranscriber()
        segments = transcriber.transcribe(p)
        if diarization:
            segments = assign_speakers(segments, diarization)
        doc = ParsedDocument(segments, {}, "audio")
    elif ext == ".pdf":
        doc = parse_pdf(p)
    else:
        doc = parse_bytes(p.read_bytes(), p.name)
    return _build(
        doc,
        name=p.name,
        title=title,
        meeting_date=meeting_date,
        series=series,
        speaker_map=speaker_map,
    )


def transcript_from_bytes(
    data: bytes,
    filename: str,
    *,
    title: str | None = None,
    meeting_date: date | None = None,
    series: str | None = None,
    speaker_map: dict[str, str] | None = None,
) -> Transcript:
    doc = parse_bytes(data, filename)
    return _build(
        doc,
        name=filename,
        title=title,
        meeting_date=meeting_date,
        series=series,
        speaker_map=speaker_map,
    )
