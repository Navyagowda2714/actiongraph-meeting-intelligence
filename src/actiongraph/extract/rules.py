"""Rule-based extractor: the offline baseline behind `OfflineProvider`.

It is a transparent, deterministic heuristic system built around how commitments are phrased
in meetings. It handles:

* explicit self-commitments ("I'll have the mockups ready by Wednesday")
* implicit / hedged commitments ("I'll take a look", "let me dig into", "I can pull that number")
* team commitments ("We'll get staging refreshed") attributed to the speaker but flagged
* requests + acceptance across turns ("Daniel, can you ...?" -> "Sure, by end of next week")
* anaphora ("Could you set that up?") resolved against the preceding "we need X" sentence,
  the accepter's own restatement, or an item the addressee just volunteered for
* unassigned needs ("We should update the help center articles") -> owner = None
* recaps are skipped so summarised items are not double counted
* decisions and risks / blockers with negation handling ("that blocker is gone")

It returns a `RawExtraction` exactly like an LLM would, so downstream code is shared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from actiongraph.extract.deadlines import find_deadline
from actiongraph.models import (
    CommitmentKind,
    Priority,
    RawActionItem,
    RawDecision,
    RawEvidence,
    RawExtraction,
    RawRisk,
    Turn,
)
from actiongraph.similarity import content_tokens

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


# --------------------------------------------------------------------------- lexicons

_I = r"\bI(?:'ll| will|’ll)"
EXCLUDE = re.compile(
    r"\bI(?:'ll| will) (?:be (?:honest|out|off|away|back|there|around|in|on|fine|happy)|admit|say|"
    r"tell you|bet|grant)\b|\bI said I'd\b|\bI can(?:'t|not)\b|\bI could be wrong\b|\bI can see\b",
    re.IGNORECASE,
)
SUBORDINATE_BEFORE_I = re.compile(
    r"\b(?:before|until|so|if|unless|when|once|that|where|whether)\s+I\b(?=\s+(?:can|could|'ll|will))",
    re.IGNORECASE,
)
CONDITIONAL = re.compile(r"(?:^|,\s*|\s)if\b", re.IGNORECASE)

IMPLICIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bleave it with me\b[,.]?\s*(?:I(?:'ll| will)\s+)?(?P<task>[^.?!]*)", re.I),
    re.compile(
        _I + r" (?P<task>(?:take a look at|take a look|have a look at|look into|try to|try and|"
        r"see if|see whether|keep an eye on|circle back on|check on|check|think about|probably)\b"
        r"[^.?!]*)",
        re.I,
    ),
    re.compile(r"\blet me (?P<task>(?!know\b)[^.?!]+)", re.I),
    re.compile(
        r"\b(?:maybe )?I (?:can|could|should be able to|might be able to|may be able to)"
        r"(?: probably)? (?P<task>(?!see\b|say\b|tell\b|imagine\b)[^.?!]+)",
        re.I,
    ),
]
EXPLICIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(_I + r" (?P<task>[^.?!]+)", re.I),
    re.compile(r"\bI(?:'m| am) going to (?P<task>[^.?!]+)", re.I),
    re.compile(r"\bI need to (?P<task>[^.?!]+)", re.I),
]
TEAM_PATTERN = re.compile(r"\bwe(?:'ll| will|'re going to| are going to) (?P<task>[^.?!]+)", re.I)
UNASSIGNED_PATTERN = re.compile(
    r"\b(?:we|someone|somebody|somebody) (?:should|needs? to|ought to|has to|must)\s+"
    r"(?:probably\s+|also\s+|really\s+|definitely\s+)*(?P<task>[^.?!]+)",
    re.I,
)
ANTECEDENT_PATTERN = re.compile(
    r"\b(?:we need|we could|we should|someone should|someone needs to|it would be good to|"
    r"we'd need)\s+(?:probably\s+|also\s+|really\s+)*(?:to\s+)?(?P<x>[^.?!]+)",
    re.I,
)
ACTION_LABEL = re.compile(r"\b(?:action item|todo|to-do)\s*[:\-]\s*(?P<rest>.+)", re.I)

REQUEST_PATTERN = re.compile(
    r"\b(?:can|could|would|will) you\s+(?:please\s+)?(?:also\s+)?(?P<task>[^?]+)"
    r"|\bplease\s+(?!do\b)(?P<task2>[^.?!]+)|\bcan someone\s+(?P<task3>[^?]+)",
    re.I,
)
AFFIRM = re.compile(
    r"^(?:sure|yes|yeah|yep|yup|will do|on it|absolutely|of course|can do|ok|okay|no problem|"
    r"definitely|leave it with me|happy to|i can|i'll|i will|consider it done)\b",
    re.I,
)
REPLY_HEDGE = re.compile(
    r"\b(?:probably|maybe|might|try|let me see|see what i can|not sure|should be able|"
    r"leave it with me|figure out|look into|take a look)\b",
    re.I,
)
SOFT_HEDGE = re.compile(r"\b(?:probably|maybe|might|try to|try and|hopefully|if i can)\b", re.I)

DECISION_PATTERN = re.compile(
    r"\b(?:let's go with|we(?:'ll| will) go with|decision made|decision\s*:|we(?:'ve| have)? decided|"
    r"let's (?:freeze|stick with|keep|cut|do that|do it|move forward|ship|go ahead|drop)|"
    r"we(?:'ll| will) stick with|we(?:'re| are) going (?:to go )?with|final decision|"
    r"the plan is to|agreed[,:]? (?:we|let's)|we(?:'re| are) not going to)(?:\b|(?<=:))",
    re.I,
)
STRONG_DECISION = re.compile(r"\b(?:decision made|decision\s*:|decided|final decision)\b", re.I)

RISK_PATTERN = re.compile(
    r"\b(?:risks?|risky|worr(?:y|ied|ies)|concerns?|concerned|might slip|could slip|at risk|"
    r"pushes everything|capacity|we don't (?:really )?know|bottleneck|"
    r"(?:hasn't|haven't) (?:been )?\w+ed\b[^.]*\b(?:yet|at all))\b",
    re.I,
)
BLOCKER_PATTERN = re.compile(
    r"\b(?:blocked (?:on|by)|blocker|waiting on|waiting for|until we (?:have|get)|before I can|"
    r"depends on|dependent on|can't (?:test|start|proceed|move|check)\b)",
    re.I,
)
RISK_NEGATION = re.compile(r"\b(?:blocker is gone|unblocked|no longer blocked|is resolved)\b", re.I)
HIGH_SEVERITY = re.compile(
    r"\b(?:beta date|launch|pushes everything|slip|month-end|deadline)\b", re.I
)

RECAP = re.compile(
    r"^(?:okay,?\s*|so,?\s*)*(?:to recap|recap|to summari[sz]e|quick recap|just to recap)\b", re.I
)
URGENT = re.compile(r"\b(?:urgent|asap|critical|right away|immediately|top priority)\b", re.I)
LOW = re.compile(
    r"\b(?:nothing urgent|no rush|when you get a chance|at some point|nice to have|whenever)\b",
    re.I,
)

ANAPHORS = {"it", "that", "this", "them", "those"}
_CLAUSE_CUT = re.compile(
    r",?\s+(?:so that|so we|because|since|which means|but it|but i|, and that)\b.*$", re.I
)
_LEADING_FILLER = re.compile(r"^(?:also|just|quickly|go ahead and|then|and)\s+", re.I)
_TRAILING_FILLER = re.compile(
    r"[\s,]*(?:too|as well|also|by the way|at some point|nothing urgent|for now|after this)\s*$",
    re.I,
)


@dataclass
class _Item:
    turn_index: int
    raw: RawActionItem
    owner_key: str | None  # resolved speaker / addressee name for anaphora merges
    extra_turns: set[str] = field(default_factory=set)


def _is_anaphoric(task: str) -> bool:
    words = re.findall(r"[a-z']+", task.lower())
    return any(w in ANAPHORS for w in words) and len(content_tokens(task)) <= 3


def clean_task(task: str, deadline_text: str | None = None) -> str:
    t = task.strip().strip(",;:- ")
    t = re.sub(r"\s+and I(?:'ll| will)\s+", " and ", t, flags=re.I)
    t = _CLAUSE_CUT.sub("", t)
    if deadline_text:
        t = re.sub(r"\s*" + re.escape(deadline_text) + r"\b", "", t, flags=re.I)
    for _ in range(2):
        t = _LEADING_FILLER.sub("", t).strip()
        t = _TRAILING_FILLER.sub("", t).strip()
    t = re.sub(r"^probably\s+", "", t, flags=re.I).strip(" ,.")
    return t[:1].upper() + t[1:] if t else t


def _priority(text: str) -> Priority:
    if LOW.search(text):
        return Priority.LOW
    if URGENT.search(text):
        return Priority.HIGH
    return Priority.MEDIUM


class RuleBasedExtractor:
    def __init__(self, participants: list[str], meeting_date: date):
        self.participants = participants
        self.meeting_date = meeting_date
        self.first_names = {p.split()[0].lower(): p for p in participants if p}

    # ------------------------------------------------------------------ helpers
    def _vocative(self, sentences: list[str], upto: int) -> str | None:
        """Name addressed directly: 'Priya, could you...' or '... could you, Priya?'."""
        for s in reversed(sentences[: upto + 1]):
            for first, full in self.first_names.items():
                pat = re.compile(
                    rf"(?:^(?:\w+,\s*)?{re.escape(first)}\b\s*,|,\s*{re.escape(first)}\s*[?.!]?$)",
                    re.I,
                )
                if pat.search(s):
                    return full
        return None

    def _self_commitment(self, sentence: str) -> tuple[str, CommitmentKind, str] | None:
        """Return (task, kind, mode) where mode in {'self', 'team'}."""
        if sentence.rstrip().endswith("?") or SUBORDINATE_BEFORE_I.search(sentence):
            return None
        stripped = EXCLUDE.sub("", sentence)
        for pat in IMPLICIT_PATTERNS:
            m = pat.search(stripped)
            if m and m.group("task").strip():
                return m.group("task"), CommitmentKind.IMPLICIT, "self"
        for pat in EXPLICIT_PATTERNS:
            m = pat.search(stripped)
            if m and m.group("task").strip():
                kind = (
                    CommitmentKind.IMPLICIT
                    if SOFT_HEDGE.search(sentence)
                    else CommitmentKind.EXPLICIT
                )
                return m.group("task"), kind, "self"
        m = TEAM_PATTERN.search(stripped)
        if m:
            return m.group("task"), CommitmentKind.EXPLICIT, "team"
        return None

    @staticmethod
    def _antecedent(sentences: list[str], idx: int) -> tuple[int, str] | None:
        for j in range(idx - 1, max(-1, idx - 3), -1):
            m = ANTECEDENT_PATTERN.search(sentences[j])
            if m and not sentences[j].rstrip().endswith("?"):
                return j, clean_task(m.group("x"))
        return None

    # ------------------------------------------------------------------ main
    def extract(self, turns: list[Turn], focus_ids: set[str] | None = None) -> RawExtraction:
        items: list[_Item] = []
        decisions: list[RawDecision] = []
        risks: list[RawRisk] = []
        consumed: set[tuple[int, int]] = set()
        sents = [split_sentences(t.text) for t in turns]

        for ti, turn in enumerate(turns):
            ss = sents[ti]
            recap = bool(ss) and bool(RECAP.search(ss[0]))
            decision_idx = {si for si, s in enumerate(ss) if DECISION_PATTERN.search(s)}

            # ---------- decisions
            for si in sorted(decision_idx):
                summary = re.sub(
                    r"^(?:okay|ok|so|great|right|fine)[,.]?\s*", "", ss[si], flags=re.I
                )
                summary = re.sub(
                    r"^(?:so\s+)?(?:decision(?: made)?\s*[:.,]?\s*)", "", summary, flags=re.I
                )
                quote_parts = [ss[si]]
                if len(content_tokens(summary)) < 4 and si + 1 < len(ss):
                    summary = ss[si + 1]
                    quote_parts.append(ss[si + 1])
                conf = 0.85 if STRONG_DECISION.search(ss[si]) else 0.75
                decisions.append(
                    RawDecision(
                        summary=clean_task(summary) or summary,
                        confidence=conf,
                        evidence=[RawEvidence(turn_id=turn.id, quote=" ".join(quote_parts))],
                    )
                )

            # ---------- risks / blockers (one per turn)
            matched = [
                si
                for si, s in enumerate(ss)
                if (RISK_PATTERN.search(s) or BLOCKER_PATTERN.search(s))
                and not RISK_NEGATION.search(s)
                and not s.rstrip().endswith("?")
            ]
            if matched:
                parts: list[str] = []
                for si in matched:
                    if ss[si] not in parts:
                        parts.append(ss[si])
                    if (
                        len(content_tokens(ss[si])) < 4
                        and si + 1 < len(ss)
                        and ss[si + 1] not in parts
                    ):
                        parts.append(ss[si + 1])
                text = " ".join(parts)
                is_blocker = any(BLOCKER_PATTERN.search(p) for p in parts)
                risks.append(
                    RawRisk(
                        description=text,
                        kind="blocker" if is_blocker else "risk",
                        severity=Priority.HIGH
                        if (is_blocker or HIGH_SEVERITY.search(text))
                        else Priority.MEDIUM,
                        confidence=0.8 if is_blocker else 0.7,
                        evidence=[RawEvidence(turn_id=turn.id, quote=p) for p in parts],
                    )
                )

            if recap:
                continue

            # ---------- self / team commitments
            for si, s in enumerate(ss):
                if (ti, si) in consumed or si in decision_idx:
                    continue
                label = ACTION_LABEL.search(s)
                if label:
                    self._labelled(label.group("rest"), ti, turn, s, items)
                    consumed.add((ti, si))
                    continue
                if CONDITIONAL.search(s) and re.search(
                    r"\bif\b.*\b(?:we|I)(?:'ll| will)\b", s, re.I
                ):
                    continue  # "if X changes, we'll have to redo Y" is a risk, not a commitment
                found = self._self_commitment(s)
                if not found:
                    continue
                task, kind, mode = found
                dl = find_deadline(s)
                desc = clean_task(task, dl.text if dl else None)
                if not desc:
                    continue
                evid = [RawEvidence(turn_id=turn.id, quote=s)]
                if _is_anaphoric(task):
                    ante = self._antecedent(ss, si)
                    if ante:
                        consumed.add((ti, ante[0]))
                        evid.insert(0, RawEvidence(turn_id=turn.id, quote=ss[ante[0]]))
                        desc = f"{desc} ({ante[1]})"
                hedged = bool(SOFT_HEDGE.search(s))
                if mode == "team":
                    conf, owner = 0.75, "we"
                elif kind == CommitmentKind.IMPLICIT:
                    conf, owner = (0.55 if hedged else 0.7), "speaker"
                else:
                    conf, owner = 0.9, "speaker"
                consumed.add((ti, si))
                items.append(
                    _Item(
                        ti,
                        RawActionItem(
                            description=desc,
                            owner=owner,
                            deadline_text=dl.text if dl else None,
                            kind=kind,
                            priority=_priority(s),
                            confidence=conf,
                            evidence=evid,
                        ),
                        owner_key=turn.speaker,
                    )
                )

            # ---------- unassigned needs ("we should ...")
            for si, s in enumerate(ss):
                if (ti, si) in consumed or si in decision_idx or s.rstrip().endswith("?"):
                    continue
                m = UNASSIGNED_PATTERN.search(s)
                if not m or re.search(r"\bif\b", s[: m.start()], re.I):
                    continue
                dl = find_deadline(s)
                desc = clean_task(m.group("task"), dl.text if dl else None)
                if not desc:
                    continue
                consumed.add((ti, si))
                items.append(
                    _Item(
                        ti,
                        RawActionItem(
                            description=desc,
                            owner=None,
                            deadline_text=dl.text if dl else None,
                            kind=CommitmentKind.IMPLICIT,
                            priority=_priority(s),
                            confidence=0.45,
                            evidence=[RawEvidence(turn_id=turn.id, quote=s)],
                        ),
                        owner_key=None,
                    )
                )

            # ---------- requests to others
            for si, s in enumerate(ss):
                if (ti, si) in consumed or si in decision_idx:
                    continue
                m = REQUEST_PATTERN.search(s)
                if not m or re.search(r"\b(?:did|do|have) you\b", s[: m.start() + 12], re.I):
                    continue
                task = next(
                    g
                    for g in (m.group("task"), m.group("task2"), m.group("task3"))
                    if g is not None
                )
                self._request(ti, si, s, task, turns, sents, items, consumed)

        result_items = [it.raw for it in items]
        if focus_ids:
            result_items = [
                r for r in result_items if any(e.turn_id in focus_ids for e in r.evidence)
            ]
            decisions = [d for d in decisions if any(e.turn_id in focus_ids for e in d.evidence)]
            risks = [r for r in risks if any(e.turn_id in focus_ids for e in r.evidence)]
        return RawExtraction(
            action_items=result_items, decisions=decisions, risks=risks, summary=""
        )

    # ------------------------------------------------------------------ sub-handlers
    def _labelled(self, rest: str, ti: int, turn: Turn, sentence: str, items: list[_Item]) -> None:
        m = re.match(
            r"(?P<name>[A-Z][\w'-]+)\s+(?:to|will|is going to)\s+(?P<task>.+)", rest.strip()
        )
        owner, task = (m.group("name"), m.group("task")) if m else (None, rest)
        if owner and owner.lower() in self.first_names:
            owner = self.first_names[owner.lower()]
        dl = find_deadline(sentence)
        items.append(
            _Item(
                ti,
                RawActionItem(
                    description=clean_task(task, dl.text if dl else None),
                    owner=owner,
                    deadline_text=dl.text if dl else None,
                    kind=CommitmentKind.EXPLICIT,
                    priority=_priority(sentence),
                    confidence=0.9 if owner else 0.5,
                    evidence=[RawEvidence(turn_id=turn.id, quote=sentence)],
                ),
                owner_key=owner,
            )
        )

    def _request(
        self,
        ti: int,
        si: int,
        sentence: str,
        task: str,
        turns: list[Turn],
        sents: list[list[str]],
        items: list[_Item],
        consumed: set[tuple[int, int]],
    ) -> None:
        requester = turns[ti].speaker
        addressee = self._vocative(sents[ti], si)
        reply_idx = None
        for tj in range(ti + 1, min(len(turns), ti + 4)):
            spk = turns[tj].speaker
            if spk == requester:
                continue
            if addressee is None or spk == addressee:
                reply_idx = tj
                break
        if addressee is None and reply_idx is not None:
            addressee = turns[reply_idx].speaker
        if addressee is None:
            return

        reply_sents = sents[reply_idx] if reply_idx is not None else []
        accepted = bool(reply_sents) and bool(AFFIRM.search(reply_sents[0]))
        reply_text = " ".join(reply_sents)
        evid = [RawEvidence(turn_id=turns[ti].id, quote=sentence)]
        reply_task = None
        reply_quote: list[str] = []
        if accepted and reply_idx is not None:
            for rj, rs in enumerate(reply_sents):
                if rj == 0 or self._self_commitment(rs):
                    consumed.add((reply_idx, rj))
                    reply_quote.append(rs)
                    found = self._self_commitment(rs)
                    if found and not _is_anaphoric(found[0]) and reply_task is None:
                        reply_task = found[0]
            evid.append(RawEvidence(turn_id=turns[reply_idx].id, quote=" ".join(reply_quote)))

        dl = find_deadline(sentence) or (
            find_deadline(" ".join(reply_quote)) if reply_quote else None
        )
        deadline_text = dl.text if dl else None
        hedged = accepted and bool(REPLY_HEDGE.search(reply_text))
        desc = clean_task(task, deadline_text)

        if _is_anaphoric(task):
            ante = self._antecedent(sents[ti], si)
            if ante:
                consumed.add((ti, ante[0]))
                evid.insert(0, RawEvidence(turn_id=turns[ti].id, quote=sents[ti][ante[0]]))
                desc = f"{desc} ({ante[1]})"
            elif reply_task:
                desc = clean_task(reply_task, deadline_text)
            else:
                # e.g. "Can you get it to me by Friday?" right after the addressee volunteered
                for prev in reversed(items):
                    if prev.owner_key == addressee and ti - prev.turn_index <= 3:
                        prev.raw.evidence.extend(evid)
                        if deadline_text and not prev.raw.deadline_text:
                            prev.raw.deadline_text = deadline_text
                        if accepted and not hedged:
                            prev.raw.kind = CommitmentKind.EXPLICIT
                            prev.raw.confidence = max(prev.raw.confidence, 0.85)
                        return
        if not desc:
            return
        if accepted:
            kind = CommitmentKind.IMPLICIT if hedged else CommitmentKind.EXPLICIT
            conf = 0.65 if hedged else 0.92
        else:
            kind, conf = CommitmentKind.EXPLICIT, 0.55
        items.append(
            _Item(
                ti,
                RawActionItem(
                    description=desc,
                    owner=addressee,
                    deadline_text=deadline_text,
                    kind=kind,
                    priority=_priority(sentence + " " + reply_text),
                    confidence=conf,
                    evidence=evid,
                ),
                owner_key=addressee,
            )
        )
