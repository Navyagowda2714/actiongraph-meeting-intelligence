"""Domain models (pydantic v2) shared by every layer of ActionGraph."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CommitmentKind(str, Enum):
    """Explicit = a clear, owned commitment. Implicit = hedged / soft ("I'll take a look")."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class ItemStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"


OwnerResolution = Literal["named", "speaker_self", "addressee", "team_we", "unresolved"]


# --------------------------------------------------------------------------- transcript


class Turn(BaseModel):
    """One speaker turn. Consecutive segments from the same speaker are merged into one turn."""

    id: str
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None


class Transcript(BaseModel):
    meeting_id: str
    title: str
    meeting_date: date
    series: str = "default"
    turns: list[Turn]
    source_format: str = "txt"

    @property
    def participants(self) -> list[str]:
        seen: list[str] = []
        for t in self.turns:
            if t.speaker not in seen:
                seen.append(t.speaker)
        return seen

    def turn(self, turn_id: str) -> Turn | None:
        for t in self.turns:
            if t.id == turn_id:
                return t
        return None


# --------------------------------------------------------------------------- extraction


class Evidence(BaseModel):
    turn_id: str
    speaker: str = ""
    quote: str


class ActionItem(BaseModel):
    id: str
    description: str
    owner: str | None = None
    owner_resolution: OwnerResolution = "unresolved"
    deadline: date | None = None
    deadline_text: str | None = None
    kind: CommitmentKind = CommitmentKind.EXPLICIT
    priority: Priority = Priority.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    status: ItemStatus = ItemStatus.OPEN
    carried_over_from: str | None = None


class Decision(BaseModel):
    id: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class Risk(BaseModel):
    id: str
    description: str
    kind: Literal["risk", "blocker"] = "risk"
    severity: Priority = Priority.MEDIUM
    owner: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class UsageStats(BaseModel):
    model: str = "offline"
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    chunks: int = 0
    raw_items: int = 0
    dropped_unsupported: int = 0
    raw_evidence_total: int = 0
    raw_evidence_valid: int = 0


class MeetingAnalysis(BaseModel):
    meeting_id: str
    title: str
    meeting_date: date
    series: str = "default"
    participants: list[str]
    summary: str
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)

    @property
    def review_queue(self) -> list[ActionItem]:
        return [a for a in self.action_items if a.needs_review]


# --------------------------------------------------------------------------- raw LLM schema
# This is the contract between the LLM (or the offline rule engine) and the pipeline.
# Deadlines are returned as *phrases*; normalisation to ISO dates is deterministic code.


class RawEvidence(BaseModel):
    turn_id: str
    quote: str


class RawActionItem(BaseModel):
    description: str
    owner: str | None = Field(
        default=None,
        description="Participant name, or 'speaker' if the speaker committed themselves, or null",
    )
    deadline_text: str | None = Field(default=None, description="Deadline phrase verbatim")
    kind: CommitmentKind = CommitmentKind.EXPLICIT
    priority: Priority = Priority.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    evidence: list[RawEvidence]


class RawDecision(BaseModel):
    summary: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    evidence: list[RawEvidence]


class RawRisk(BaseModel):
    description: str
    kind: Literal["risk", "blocker"] = "risk"
    severity: Priority = Priority.MEDIUM
    owner: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    evidence: list[RawEvidence]


class RawExtraction(BaseModel):
    action_items: list[RawActionItem] = Field(default_factory=list)
    decisions: list[RawDecision] = Field(default_factory=list)
    risks: list[RawRisk] = Field(default_factory=list)
    summary: str = ""
