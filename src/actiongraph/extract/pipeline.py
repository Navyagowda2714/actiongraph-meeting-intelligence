"""Extraction pipeline: chunk -> LLM (or offline rules) -> validate -> normalise -> merge.

Everything after the LLM call is deterministic and provider-independent:
* evidence validation (quotes must actually appear in the cited turn; unsupported items dropped)
* owner resolution ("speaker" / "I" -> the turn's speaker, first names -> participants)
* deadline normalisation (phrase -> ISO date relative to the meeting date)
* confidence-based human-review flags
* de-duplication across chunk overlaps
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from difflib import SequenceMatcher

from pydantic import ValidationError

from actiongraph.config import get_settings
from actiongraph.extract.chunking import chunk_turns
from actiongraph.extract.deadlines import normalise_deadline
from actiongraph.extract.prompts import SUMMARY_SYSTEM, SYSTEM_PROMPT, build_extraction_prompt
from actiongraph.llm import LLMClient, Message, OfflineProvider, extract_json
from actiongraph.models import (
    ActionItem,
    Decision,
    Evidence,
    MeetingAnalysis,
    OwnerResolution,
    Priority,
    RawEvidence,
    RawExtraction,
    Risk,
    Transcript,
    UsageStats,
)
from actiongraph.similarity import similarity

log = logging.getLogger(__name__)

SELF_OWNERS = {"speaker", "i", "me", "myself", "self"}
TEAM_OWNERS = {"we", "us", "team", "our team", "the team"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_supported(quote: str, turn_text: str, threshold: float = 0.85) -> bool:
    """True if `quote` appears in `turn_text` (normalised), or fuzzily for minor model edits."""
    q, t = _norm(quote), _norm(turn_text)
    if not q:
        return False
    if q in t:
        return True
    if len(q) > len(t) * 1.1:
        return False
    sm = SequenceMatcher(None, t, q, autojunk=False)
    match = sum(b.size for b in sm.get_matching_blocks())
    return match / len(q) >= threshold


class ExtractionPipeline:
    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        chunk_max_tokens: int | None = None,
        overlap_turns: int | None = None,
        review_threshold: float | None = None,
        max_retries: int = 1,
    ):
        s = get_settings()
        self.llm = llm or OfflineProvider()
        self.chunk_max_tokens = chunk_max_tokens or s.chunk_max_tokens
        self.overlap_turns = s.chunk_overlap_turns if overlap_turns is None else overlap_turns
        self.review_threshold = review_threshold or s.review_confidence_threshold
        self.max_retries = max_retries

    # ------------------------------------------------------------------ LLM call
    def _call(self, prompt: str, usage: UsageStats) -> RawExtraction | None:
        messages = [Message("user", prompt)]
        schema = RawExtraction.model_json_schema()
        for attempt in range(self.max_retries + 1):
            resp = self.llm.complete(messages, system=SYSTEM_PROMPT, json_schema=schema)
            usage.llm_calls += 1
            usage.input_tokens += resp.input_tokens
            usage.output_tokens += resp.output_tokens
            usage.latency_ms += resp.latency_ms
            usage.model = resp.model or usage.model
            try:
                return RawExtraction.model_validate(extract_json(resp.text))
            except (ValueError, ValidationError) as exc:
                log.warning("Invalid extraction JSON (attempt %d): %s", attempt + 1, exc)
                messages = [
                    *messages,
                    Message("assistant", resp.text),
                    Message(
                        "user",
                        f"That output was invalid: {str(exc)[:500]}. Return valid JSON only.",
                    ),
                ]
        return None

    # ------------------------------------------------------------------ public
    def run(self, transcript: Transcript) -> MeetingAnalysis:
        usage = UsageStats(model=getattr(self.llm, "model", "unknown"))
        chunks = chunk_turns(transcript.turns, self.chunk_max_tokens, self.overlap_turns)
        usage.chunks = len(chunks)
        raws: list[RawExtraction] = []
        for chunk in chunks:
            raw = self._call(build_extraction_prompt(transcript, chunk, len(chunks)), usage)
            if raw is not None:
                raws.append(raw)

        actions: list[ActionItem] = []
        decisions: list[Decision] = []
        risks: list[Risk] = []
        for raw in raws:
            usage.raw_items += len(raw.action_items) + len(raw.decisions) + len(raw.risks)
            for ra in raw.action_items:
                ev = self._evidence(ra.evidence, transcript, usage)
                if not ev:
                    usage.dropped_unsupported += 1
                    continue
                actions.append(self._action(ra, ev, transcript, len(ev) < len(ra.evidence)))
            for rd in raw.decisions:
                ev = self._evidence(rd.evidence, transcript, usage)
                if not ev:
                    usage.dropped_unsupported += 1
                    continue
                decisions.append(
                    Decision(
                        id="",
                        summary=rd.summary,
                        confidence=rd.confidence,
                        evidence=ev,
                        needs_review=rd.confidence < self.review_threshold,
                        review_reasons=["low confidence"]
                        if rd.confidence < self.review_threshold
                        else [],
                    )
                )
            for rr in raw.risks:
                ev = self._evidence(rr.evidence, transcript, usage)
                if not ev:
                    usage.dropped_unsupported += 1
                    continue
                risks.append(
                    Risk(
                        id="",
                        description=rr.description,
                        kind=rr.kind,
                        severity=rr.severity,
                        owner=self._resolve_owner(rr.owner, ev, transcript)[0],
                        confidence=rr.confidence,
                        evidence=ev,
                        needs_review=rr.confidence < self.review_threshold,
                    )
                )

        actions = _dedupe_actions(actions)
        decisions = _dedupe(decisions, key=lambda d: d.summary)
        risks = _dedupe(risks, key=lambda r: r.description)
        mid = transcript.meeting_id
        for i, a in enumerate(actions, 1):
            a.id = f"{mid}-a{i}"
        for i, d in enumerate(decisions, 1):
            d.id = f"{mid}-d{i}"
        for i, r in enumerate(risks, 1):
            r.id = f"{mid}-r{i}"

        summary = self._summary(transcript, raws, actions, decisions, risks, usage)
        return MeetingAnalysis(
            meeting_id=mid,
            title=transcript.title,
            meeting_date=transcript.meeting_date,
            series=transcript.series,
            participants=transcript.participants,
            summary=summary,
            action_items=actions,
            decisions=decisions,
            risks=risks,
            usage=usage,
        )

    # ------------------------------------------------------------------ post-processing
    @staticmethod
    def _evidence(
        raw: list[RawEvidence], transcript: Transcript, usage: UsageStats
    ) -> list[Evidence]:
        out: list[Evidence] = []
        for e in raw:
            usage.raw_evidence_total += 1
            turn = transcript.turn(e.turn_id)
            if turn is None or not quote_supported(e.quote, turn.text):
                continue
            usage.raw_evidence_valid += 1
            out.append(Evidence(turn_id=turn.id, speaker=turn.speaker, quote=e.quote))
        return out

    @staticmethod
    def _resolve_owner(
        owner: str | None, evidence: list[Evidence], transcript: Transcript
    ) -> tuple[str | None, OwnerResolution]:
        if not owner or owner.strip().lower() in {"none", "null", "unassigned", "unknown"}:
            return None, "unresolved"
        key = owner.strip().lower()
        if key in SELF_OWNERS:
            # the committing speaker is the speaker of the last cited turn that isn't a request
            return evidence[-1].speaker, "speaker_self"
        if key in TEAM_OWNERS:
            return evidence[-1].speaker, "team_we"
        for p in transcript.participants:
            if key == p.lower() or key == p.split()[0].lower():
                return p, "named"
        return owner.strip(), "named"

    def _action(
        self, ra, evidence: list[Evidence], transcript: Transcript, partial_evidence: bool
    ) -> ActionItem:
        owner, resolution = self._resolve_owner(ra.owner, evidence, transcript)
        reasons: list[str] = []
        norm = normalise_deadline(ra.deadline_text, transcript.meeting_date)
        if ra.deadline_text and norm.value is None:
            reasons.append(f"deadline '{ra.deadline_text}' could not be resolved")
        elif norm.ambiguous:
            reasons.append(norm.note or "ambiguous deadline")
        if owner is None:
            reasons.append("no owner")
        elif resolution == "team_we":
            reasons.append("team commitment ('we') attributed to the speaker")
        elif owner not in transcript.participants:
            reasons.append(f"owner '{owner}' is not a meeting participant")
        if ra.confidence < self.review_threshold:
            reasons.append(f"low confidence ({ra.confidence:.2f})")
        if partial_evidence:
            reasons.append("some cited evidence could not be verified")
        priority = ra.priority
        if (
            norm.value is not None
            and priority == Priority.MEDIUM
            and norm.value - transcript.meeting_date <= timedelta(days=1)
        ):
            priority = Priority.HIGH
        return ActionItem(
            id="",
            description=ra.description,
            owner=owner,
            owner_resolution=resolution,
            deadline=norm.value,
            deadline_text=ra.deadline_text,
            kind=ra.kind,
            priority=priority,
            confidence=ra.confidence,
            needs_review=bool(reasons),
            review_reasons=reasons,
            evidence=evidence,
        )

    def _summary(self, transcript, raws, actions, decisions, risks, usage) -> str:
        parts = [r.summary.strip() for r in raws if r.summary.strip()]
        if len(parts) == 1:
            return parts[0]
        if len(parts) > 1 and not isinstance(self.llm, OfflineProvider):
            resp = self.llm.complete(
                [
                    Message(
                        "user", "Merge these partial meeting summaries:\n\n" + "\n\n".join(parts)
                    )
                ],
                system=SUMMARY_SYSTEM,
            )
            usage.llm_calls += 1
            usage.input_tokens += resp.input_tokens
            usage.output_tokens += resp.output_tokens
            return resp.text.strip()
        return template_summary(transcript, actions, decisions, risks)


def template_summary(transcript: Transcript, actions, decisions, risks) -> str:
    """Deterministic summary assembled from the extracted records (used in offline mode)."""
    lines = [
        f"{transcript.title} ({transcript.meeting_date.isoformat()}) with "
        f"{len(transcript.participants)} participants: {', '.join(transcript.participants)}."
    ]
    if decisions:
        lines.append("Decisions: " + "; ".join(d.summary.rstrip(".") for d in decisions) + ".")
    if actions:
        owners = sorted({a.owner for a in actions if a.owner})
        review = sum(a.needs_review for a in actions)
        lines.append(
            f"{len(actions)} action items across {len(owners)} owners"
            + (f", {review} flagged for review." if review else ".")
        )
    blockers = [r for r in risks if r.kind == "blocker"]
    if risks:
        lines.append(
            f"{len(risks)} risk{'s' if len(risks) != 1 else ''}/blocker{'s' if len(risks) != 1 else ''} raised"
            + (f", including blocker: {blockers[0].description}" if blockers else ".")
        )
    return " ".join(lines)


def _merge_evidence(a: list[Evidence], b: list[Evidence]) -> list[Evidence]:
    seen = {(e.turn_id, e.quote) for e in a}
    return a + [e for e in b if (e.turn_id, e.quote) not in seen]


def _dedupe_actions(items: list[ActionItem]) -> list[ActionItem]:
    out: list[ActionItem] = []
    for it in items:
        dup = None
        for prev in out:
            if it.owner and prev.owner and it.owner != prev.owner:
                continue
            sim = similarity(it.description, prev.description)
            overlap = {e.turn_id for e in it.evidence} & {e.turn_id for e in prev.evidence}
            if sim >= 0.6 or (overlap and sim >= 0.3):
                dup = prev
                break
        if dup is None:
            out.append(it)
            continue
        dup.evidence = _merge_evidence(dup.evidence, it.evidence)
        dup.deadline = dup.deadline or it.deadline
        dup.deadline_text = dup.deadline_text or it.deadline_text
        dup.owner = dup.owner or it.owner
        dup.confidence = max(dup.confidence, it.confidence)
    return out


def _dedupe(items, key):
    out = []
    for it in items:
        dup = next((p for p in out if similarity(key(it), key(p)) >= 0.5), None)
        if dup is None:
            out.append(it)
        else:
            dup.evidence = _merge_evidence(dup.evidence, it.evidence)
            dup.confidence = max(dup.confidence, it.confidence)
    return out
