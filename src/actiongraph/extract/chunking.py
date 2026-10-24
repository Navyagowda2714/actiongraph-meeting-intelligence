"""Split long meetings into token-bounded chunks along turn boundaries.

Each chunk carries `context` turns (the tail of the previous chunk) so that a request at the end
of one chunk and its acceptance at the start of the next are both visible. Items are emitted only
if they cite at least one *focus* turn; duplicates across the overlap are merged downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from actiongraph.llm.base import approx_tokens
from actiongraph.models import Turn


@dataclass(frozen=True)
class Chunk:
    index: int
    context: list[Turn]
    focus: list[Turn]

    @property
    def turns(self) -> list[Turn]:
        return [*self.context, *self.focus]


def chunk_turns(turns: list[Turn], max_tokens: int = 1500, overlap_turns: int = 2) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    chunks: list[Chunk] = []
    current: list[Turn] = []
    budget = 0
    for turn in turns:
        cost = approx_tokens(f"{turn.speaker}: {turn.text}") + 12  # JSON overhead
        if current and budget + cost > max_tokens:
            chunks.append(Chunk(len(chunks), [], current))
            current, budget = [], 0
        current.append(turn)
        budget += cost
    if current:
        chunks.append(Chunk(len(chunks), [], current))
    if overlap_turns > 0:
        chunks = [
            Chunk(c.index, chunks[c.index - 1].focus[-overlap_turns:] if c.index else [], c.focus)
            for c in chunks
        ]
    return chunks
