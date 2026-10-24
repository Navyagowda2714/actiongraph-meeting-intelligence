"""Transcript parsers: TXT, WebVTT, SRT, JSON speaker turns and PDF.

All parsers produce raw `Segment`s which are then normalised by `segments_to_turns`:
speaker labels are mapped through an optional diarisation speaker map (SPEAKER_00 -> "Hannah"),
consecutive segments from the same speaker are merged into one turn, and turns get stable ids
(t001, t002, ...) that every extracted item cites as evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from actiongraph.models import Turn

UNKNOWN_SPEAKER = "Unknown"


@dataclass
class Segment:
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None


@dataclass
class ParsedDocument:
    segments: list[Segment]
    metadata: dict[str, str]
    source_format: str


# --------------------------------------------------------------------------- helpers

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?")
_META = re.compile(r"^(?:#|NOTE)\s*(title|meeting_date|date|series)\s*:\s*(.+)$", re.IGNORECASE)


def parse_timestamp(value: str) -> float | None:
    m = _TS.search(value)
    if not m:
        return None
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + (int(ms.ljust(3, "0")) / 1000 if ms else 0)


_SPEAKER_PREFIX = re.compile(r"^(?P<speaker>[A-Z][\w.'\- ]{0,40}?|SPEAKER_\d+)\s*:\s+(?P<text>.+)$")


def split_speaker(line: str) -> tuple[str | None, str]:
    """Split 'Name: text' into (Name, text). Guards against 'Note: ...' style false positives."""
    m = _SPEAKER_PREFIX.match(line.strip())
    if not m:
        return None, line.strip()
    speaker = m.group("speaker").strip()
    if len(speaker.split()) > 4:
        return None, line.strip()
    return speaker, m.group("text").strip()


def _read_meta(line: str, meta: dict[str, str]) -> bool:
    m = _META.match(line.strip())
    if m:
        key = m.group(1).lower()
        meta["meeting_date" if key == "date" else key] = m.group(2).strip()
        return True
    return False


# --------------------------------------------------------------------------- formats


_TXT_LINE = re.compile(r"^\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s+(?P<rest>.+)$")


def parse_txt(text: str) -> ParsedDocument:
    """`[00:01:02] Name: text` or `Name: text` lines; unlabelled lines continue the previous turn."""
    meta: dict[str, str] = {}
    segs: list[Segment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _read_meta(line, meta):
            continue
        ts = None
        m = _TXT_LINE.match(line)
        if m:
            ts = parse_timestamp(m.group("ts"))
            line = m.group("rest")
        speaker, body = split_speaker(line)
        if speaker is None:
            if segs:
                segs[-1].text += " " + body
                continue
            speaker = UNKNOWN_SPEAKER
        segs.append(Segment(speaker=speaker, text=body, start=ts))
    return ParsedDocument(segs, meta, "txt")


_CUE_TIME = re.compile(r"(?P<a>[\d:.,]+)\s*-->\s*(?P<b>[\d:.,]+)")
_VOICE = re.compile(r"<v(?:\.[\w.]+)?\s+([^>]+)>(.*?)(?:</v>|$)", re.DOTALL)
_TAGS = re.compile(r"</?[^>]+>")


def _parse_cues(text: str, fmt: str) -> ParsedDocument:
    meta: dict[str, str] = {}
    segs: list[Segment] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("WEBVTT"):
            for ln in lines[1:]:
                _read_meta(ln, meta)
            continue
        if lines[0].startswith("NOTE"):
            for ln in lines:
                _read_meta(ln, meta)
            continue
        time_idx = next((i for i, ln in enumerate(lines) if _CUE_TIME.search(ln)), None)
        if time_idx is None:
            continue
        tm = _CUE_TIME.search(lines[time_idx])
        assert tm is not None
        start, end = parse_timestamp(tm.group("a")), parse_timestamp(tm.group("b"))
        payload = " ".join(lines[time_idx + 1 :]).strip()
        voice = _VOICE.search(payload)
        if voice:
            speaker, body = voice.group(1).strip(), voice.group(2)
        else:
            speaker, body = split_speaker(_TAGS.sub("", payload))
        body = _TAGS.sub("", body).strip()
        if not body:
            continue
        segs.append(Segment(speaker=speaker or UNKNOWN_SPEAKER, text=body, start=start, end=end))
    return ParsedDocument(segs, meta, fmt)


def parse_vtt(text: str) -> ParsedDocument:
    return _parse_cues(text, "vtt")


def parse_srt(text: str) -> ParsedDocument:
    return _parse_cues(text, "srt")


def parse_json(text: str) -> ParsedDocument:
    """Accepts `{"turns": [{"speaker", "text", "start"?, "end"?}], ...metadata}` or a bare list."""
    data: Any = json.loads(text)
    meta: dict[str, str] = {}
    if isinstance(data, dict):
        for key in ("title", "meeting_date", "series"):
            if key in data:
                meta[key] = str(data[key])
        items = data.get("turns") or data.get("segments") or data.get("utterances") or []
    else:
        items = data
    segs = []
    for it in items:
        speaker = it.get("speaker") or it.get("speaker_label") or it.get("name") or UNKNOWN_SPEAKER
        segs.append(
            Segment(
                speaker=str(speaker),
                text=str(it.get("text", "")).strip(),
                start=it.get("start"),
                end=it.get("end"),
            )
        )
    return ParsedDocument([s for s in segs if s.text], meta, "json")


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract text with pypdf, then reuse the TXT parser (PDF exports of transcripts)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    doc = parse_txt(text)
    doc.source_format = "pdf"
    return doc


PARSERS = {
    ".txt": parse_txt,
    ".md": parse_txt,
    ".vtt": parse_vtt,
    ".srt": parse_srt,
    ".json": parse_json,
}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac", ".mkv", ".mov"}


def parse_bytes(data: bytes, filename: str) -> ParsedDocument:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        doc = parse_txt(text)
        doc.source_format = "pdf"
        return doc
    if ext not in PARSERS:
        raise ValueError(f"Unsupported transcript format: {ext or filename}")
    return PARSERS[ext](data.decode("utf-8-sig"))


# --------------------------------------------------------------------------- normalisation


def segments_to_turns(
    segments: list[Segment], speaker_map: dict[str, str] | None = None, merge: bool = True
) -> list[Turn]:
    speaker_map = speaker_map or {}
    turns: list[Turn] = []
    for seg in segments:
        speaker = speaker_map.get(seg.speaker, seg.speaker)
        if merge and turns and turns[-1].speaker == speaker:
            prev = turns[-1]
            prev.text = f"{prev.text} {seg.text}".strip()
            prev.end = seg.end if seg.end is not None else prev.end
            continue
        turns.append(
            Turn(
                id=f"t{len(turns) + 1:03d}",
                speaker=speaker,
                text=seg.text,
                start=seg.start,
                end=seg.end,
            )
        )
    return turns


def parse_meeting_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        import dateparser

        dt = dateparser.parse(value)
        return dt.date() if dt else None
