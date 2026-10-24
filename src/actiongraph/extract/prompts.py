"""Prompt templates for LLM extraction."""

from __future__ import annotations

import json

from actiongraph.extract.chunking import Chunk
from actiongraph.models import Transcript

SYSTEM_PROMPT = """You are ActionGraph, an assistant that turns meeting transcripts into structured,
evidence-backed records. Extract ONLY what the transcript supports.

Action items
- Include explicit commitments ("I'll send the deck by Friday") AND implicit ones
  ("I'll take a look", "let me check", "I can pull that number", "leave it with me").
  Set kind="implicit" for hedged/soft commitments, "explicit" for clear ones.
- A request ("Priya, could you ...?") becomes an item for the addressee when they accept it.
  Cite both the request turn and the acceptance turn.
- owner: use "speaker" when the speaker commits themselves (I / me / my), "we" for team
  commitments, the participant's name for requests/assignments, or null if nobody owns it.
- deadline_text: copy the deadline phrase verbatim ("by end of next week", "tomorrow morning").
  Do NOT convert it to a date. null if none.
- Do not create items from recaps of items already listed, or from things already completed.

Decisions: conclusions the group reached ("let's go with Stripe for the beta").
Risks: risks and blockers (kind="blocker" when work cannot proceed until something happens).

Evidence: every item MUST cite turn ids and an exact verbatim quote from that turn.
Confidence: 0-1, lower it for hedged, ambiguous or unaccepted items.
Only cite turns from the FOCUS set; CONTEXT turns are there to help you understand references.
"""

SUMMARY_SYSTEM = (
    "You write concise, factual meeting summaries (3-5 sentences) from partial summaries. "
    "No speculation."
)


def build_extraction_prompt(transcript: Transcript, chunk: Chunk, n_chunks: int) -> str:
    payload = {
        "meeting_date": transcript.meeting_date.isoformat(),
        "participants": transcript.participants,
        "focus_turn_ids": [t.id for t in chunk.focus],
        "turns": [t.model_dump(include={"id", "speaker", "text"}) for t in chunk.turns],
    }
    return (
        f"Meeting: {transcript.title}\n"
        f"Meeting date: {transcript.meeting_date.isoformat()} "
        f"({transcript.meeting_date.strftime('%A')})\n"
        f"Participants: {', '.join(transcript.participants)}\n"
        f"Chunk {chunk.index + 1} of {n_chunks}. "
        f"Context turns: {[t.id for t in chunk.context]}.\n\n"
        f"<turns_json>\n{json.dumps(payload, ensure_ascii=False)}\n</turns_json>\n\n"
        "Return the JSON object now."
    )
