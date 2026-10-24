"""Analyze a single transcript with the Python API and print evidence-linked action items.

    python examples/analyze_transcript.py data/samples/atlas_sync_2026-03-02.vtt

Uses the provider selected by LLM_PROVIDER (offline rule engine when no API key is set).
"""

from __future__ import annotations

import sys

from actiongraph.extract.pipeline import ExtractionPipeline
from actiongraph.ingest import load_transcript
from actiongraph.llm import get_llm


def main(path: str) -> None:
    transcript = load_transcript(path)
    analysis = ExtractionPipeline(get_llm()).run(transcript)
    print(
        f"{analysis.title} - {len(analysis.action_items)} action items ({analysis.usage.model})\n"
    )
    for item in analysis.action_items:
        due = item.deadline.isoformat() if item.deadline else "no deadline"
        print(f"* {item.description}  [{item.owner or 'UNASSIGNED'}, {due}, {item.kind.value}]")
        for ev in item.evidence:
            print(f'    {ev.turn_id} {ev.speaker}: "{ev.quote}"')
        if item.needs_review:
            print(f"    needs review: {'; '.join(item.review_reasons)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/samples/atlas_sync_2026-03-02.vtt")
