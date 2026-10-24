"""Ingest a meeting series into SQLite and show which items were closed in later meetings.

python examples/track_series.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from actiongraph.ingest import load_transcript
from actiongraph.llm import get_llm
from actiongraph.service import ActionGraphService
from actiongraph.tracking.store import Store

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def main() -> None:
    service = ActionGraphService(Store(":memory:"), get_llm())
    for path in sorted(SAMPLES.glob("atlas_sync_*")):
        result = service.ingest_transcript(load_transcript(path, series="atlas"))
        print(f"\n== {result.analysis.title}")
        print(f"   new items: {len(result.tracking.new_item_ids)}")
        for ev in result.tracking.completed:
            print(f'   closed: {ev.description}  <- {ev.turn_id} "{ev.quote}"')
    print("\nDue within 3 days of 2026-03-17:")
    for tracked in service.store.upcoming(date(2026, 3, 17), days=3):
        print(f"   {tracked.item.deadline}  {tracked.item.owner}: {tracked.item.description}")


if __name__ == "__main__":
    main()
