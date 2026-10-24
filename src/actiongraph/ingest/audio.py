"""Audio/video transcription adapters (optional) and diarisation alignment.

Transcription back-ends are imported lazily:
  * `FasterWhisperTranscriber` -> `pip install actiongraph[whisper]` (runs locally)
  * `OpenAIWhisperTranscriber` -> `pip install actiongraph[openai]` + OPENAI_API_KEY

Neither ASR model ships with diarisation, so `assign_speakers` aligns ASR segments with an
external diarisation track (e.g. pyannote output, `[(start, end, "SPEAKER_00"), ...]`) by
maximum temporal overlap. The result feeds the same `segments_to_turns` normaliser as text formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from actiongraph.ingest.parsers import UNKNOWN_SPEAKER, Segment

DiarizationTrack = list[tuple[float, float, str]]


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> list[Segment]: ...


class FasterWhisperTranscriber:
    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "int8"):
        self.model_size, self.device, self.compute_type = model_size, device, compute_type
        self._model = None

    def transcribe(self, path: Path) -> list[Segment]:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "Install the whisper extra: pip install actiongraph[whisper]"
                ) from exc
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        segments, _info = self._model.transcribe(str(path), vad_filter=True)
        return [Segment(UNKNOWN_SPEAKER, s.text.strip(), s.start, s.end) for s in segments]


class OpenAIWhisperTranscriber:
    def __init__(self, model: str = "whisper-1"):
        self.model = model

    def transcribe(self, path: Path) -> list[Segment]:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Install the openai extra: pip install actiongraph[openai]") from exc
        client = openai.OpenAI()
        with open(path, "rb") as fh:
            resp = client.audio.transcriptions.create(
                model=self.model, file=fh, response_format="verbose_json"
            )
        return [
            Segment(UNKNOWN_SPEAKER, s["text"].strip(), s["start"], s["end"])
            for s in (resp.model_dump().get("segments") or [])
        ]


def assign_speakers(segments: list[Segment], diarization: DiarizationTrack) -> list[Segment]:
    """Label each ASR segment with the diarisation speaker it overlaps most."""
    out: list[Segment] = []
    for seg in segments:
        best, best_overlap = seg.speaker, 0.0
        if seg.start is not None and seg.end is not None:
            for d_start, d_end, label in diarization:
                overlap = min(seg.end, d_end) - max(seg.start, d_start)
                if overlap > best_overlap:
                    best, best_overlap = label, overlap
        out.append(Segment(best, seg.text, seg.start, seg.end))
    return out
