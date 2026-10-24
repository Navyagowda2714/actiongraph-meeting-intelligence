"""Application service: ingestion + extraction + tracking behind one facade (used by CLI and API)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from actiongraph.extract.pipeline import ExtractionPipeline
from actiongraph.ingest import load_transcript, transcript_from_bytes
from actiongraph.llm import LLMClient
from actiongraph.models import ItemStatus, MeetingAnalysis, Transcript
from actiongraph.tracking.store import Store
from actiongraph.tracking.tracker import Tracker, TrackingReport

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    transcript: Transcript
    analysis: MeetingAnalysis
    tracking: TrackingReport


class ActionGraphService:
    def __init__(self, store: Store, llm: LLMClient | None = None):
        self.store = store
        self.pipeline = ExtractionPipeline(llm)
        self.tracker = Tracker(store)

    def ingest_transcript(self, transcript: Transcript) -> IngestResult:
        analysis = self.pipeline.run(transcript)
        report = self.tracker.ingest(transcript, analysis)
        log.info(
            "Ingested %s: %d actions, %d decisions, %d risks, %d completions detected",
            transcript.meeting_id,
            len(analysis.action_items),
            len(analysis.decisions),
            len(analysis.risks),
            len(report.completed),
        )
        return IngestResult(transcript, analysis, report)

    def ingest_file(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        meeting_date: date | None = None,
        series: str | None = None,
    ) -> IngestResult:
        return self.ingest_transcript(
            load_transcript(path, title=title, meeting_date=meeting_date, series=series)
        )

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        title: str | None = None,
        meeting_date: date | None = None,
        series: str | None = None,
        speaker_map: dict[str, str] | None = None,
    ) -> IngestResult:
        return self.ingest_transcript(
            transcript_from_bytes(
                data,
                filename,
                title=title,
                meeting_date=meeting_date,
                series=series,
                speaker_map=speaker_map,
            )
        )

    def live_analysis(
        self, meeting_id: str
    ) -> tuple[Transcript, MeetingAnalysis, dict | None] | None:
        """Stored analysis with item statuses/owners overlaid from the tracking store."""
        got = self.store.get_meeting(meeting_id)
        if got is None:
            return None
        transcript, analysis, tracking = got
        for item in analysis.action_items:
            tracked = self.store.get_item(item.carried_over_from or item.id)
            if tracked:
                item.status = tracked.item.status
                item.owner = item.owner or tracked.item.owner
        return transcript, analysis, tracking

    def set_item_status(self, item_id: str, status: ItemStatus) -> str | None:
        """Update an item; ids of re-raised items resolve to the original tracked item."""
        target = item_id
        if self.store.get_item(item_id) is None:
            for m in self.store.list_meetings():
                got = self.store.get_meeting(m["id"])
                if not got:
                    continue
                for it in got[1].action_items:
                    if it.id == item_id and it.carried_over_from:
                        target = it.carried_over_from
        if not self.store.set_status(target, status):
            return None
        self.store.add_mention(target, "manual", None, None, f"manual:{status.value}")
        return target
