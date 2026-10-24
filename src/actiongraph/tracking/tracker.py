"""Cross-meeting tracking: carry open items forward and detect completions / re-mentions.

When a new meeting in a series is ingested:
1. Open items from *earlier* meetings in the series are loaded.
2. Each clause of the new transcript is compared (hashed bag-of-words cosine) with every open item.
   A clause in the past tense with a completion cue ("is fixed", "I shared", "went out", "is live")
   that matches an item closes it, with the clause stored as completion evidence.
   Speaker == owner and owner-name mentions add a small bonus to the match score.
3. Non-completion clauses that match strongly are recorded as mentions ("I've started on the note").
4. Newly extracted items that duplicate an open item are linked (`carried_over_from`) instead of
   being tracked twice; they can fill in a missing owner or deadline ("I can take the help center
   articles" assigns a previously unowned item).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from actiongraph.extract.rules import split_sentences
from actiongraph.models import ItemStatus, MeetingAnalysis, Transcript
from actiongraph.similarity import Embedder, cosine_matrix, stem
from actiongraph.tracking.store import Store, TrackedItem

COMPLETION_CUE = re.compile(
    r"\b(?:done|finished|complete|completed|fixed|merged|shipped|sent|shared|filed|triaged|"
    r"escalated|refreshed|resolved|closed|wrapped up|went out|(?:is|are) live|pulled|ran|"
    r"set up|written|delivered|submitted|published|deployed|launched|landed|sorted)\b",
    re.I,
)
FUTURE_OR_NEGATED = re.compile(
    r"\b(?:I'll|we'll|will|going to|should|could|can you|could you|not|n't|yet to|haven't|hasn't|"
    r"need to|plan to|try to|once)\b|\?$",
    re.I,
)
_CLAUSE = re.compile(r";\s+|,\s+(?=(?:and |but |so )?(?:I|we|it|the|and|but|so)\b)", re.I)


@dataclass
class TrackingConfig:
    complete_threshold: float = 0.3
    mention_threshold: float = 0.45
    reraise_threshold: float = 0.45
    owner_bonus: float = 0.1
    verb_bonus: float = 0.15


GENERIC_VERBS = frozenset(
    {"have", "get", "take", "make", "look", "set", "do", "keep", "see", "check"}
)


def main_verb(description: str) -> str | None:
    """Stemmed leading verb of an imperative task description ("Triage them" -> "triag")."""
    words = re.findall(r"[a-z]+", description.lower())
    if not words or words[0] in GENERIC_VERBS:
        return None
    return stem(words[0])


class TrackEvent(BaseModel):
    item_id: str
    description: str
    owner: str | None
    from_meeting_id: str
    turn_id: str | None = None
    quote: str | None = None
    score: float = 0.0


class TrackingReport(BaseModel):
    meeting_id: str
    series: str
    completed: list[TrackEvent] = Field(default_factory=list)
    mentioned: list[TrackEvent] = Field(default_factory=list)
    reraised: list[TrackEvent] = Field(default_factory=list)
    still_open: list[TrackEvent] = Field(default_factory=list)
    new_item_ids: list[str] = Field(default_factory=list)


def clauses(text: str) -> list[str]:
    out: list[str] = []
    for s in split_sentences(text):
        out += [c.strip(" ,.") for c in _CLAUSE.split(s) if c and c.strip(" ,.")]
    return out


class Tracker:
    def __init__(
        self, store: Store, config: TrackingConfig | None = None, embedder: Embedder | None = None
    ):
        self.store = store
        self.cfg = config or TrackingConfig()
        self.embedder = embedder

    def ingest(self, transcript: Transcript, analysis: MeetingAnalysis) -> TrackingReport:
        series = analysis.series
        open_items = self.store.open_items_before(series, analysis.meeting_date)
        report = TrackingReport(meeting_id=analysis.meeting_id, series=series)
        self.store.save_meeting(transcript, analysis)

        closed: set[str] = set()
        mentioned: set[str] = set()
        if open_items:
            units = [(t, c) for t in transcript.turns for c in clauses(t.text)]
            if units:
                sims = cosine_matrix(
                    [c for _, c in units], [o.item.description for o in open_items], self.embedder
                )
                verbs = [main_verb(o.item.description) for o in open_items]
                for ui, (turn, clause) in enumerate(units):
                    clause_stems = {stem(w) for w in re.findall(r"[a-z]+", clause.lower())}
                    is_completion = bool(
                        COMPLETION_CUE.search(clause)
                    ) and not FUTURE_OR_NEGATED.search(clause)
                    scores = []
                    for oi, o in enumerate(open_items):
                        if o.item.id in closed:
                            scores.append(-1.0)
                            continue
                        s = float(sims[ui, oi])
                        if (
                            s > 0
                            and o.item.owner
                            and (
                                turn.speaker == o.item.owner
                                or o.item.owner.split()[0].lower() in clause.lower()
                            )
                        ):
                            s += self.cfg.owner_bonus
                        if s > 0 and verbs[oi] and verbs[oi] in clause_stems:
                            s += self.cfg.verb_bonus
                        scores.append(s)
                    best = max(range(len(scores)), key=scores.__getitem__)
                    score, target = scores[best], open_items[best]
                    if is_completion and score >= self.cfg.complete_threshold:
                        closed.add(target.item.id)
                        self.store.set_status(
                            target.item.id, ItemStatus.COMPLETED, analysis.meeting_id
                        )
                        self.store.add_mention(
                            target.item.id, analysis.meeting_id, turn.id, clause, "completed", score
                        )
                        report.completed.append(self._event(target, turn.id, clause, score))
                    elif (
                        not is_completion
                        and score >= self.cfg.mention_threshold
                        and target.item.id not in mentioned
                    ):
                        mentioned.add(target.item.id)
                        self.store.add_mention(
                            target.item.id, analysis.meeting_id, turn.id, clause, "mentioned", score
                        )
                        report.mentioned.append(self._event(target, turn.id, clause, score))

        # Link newly extracted items that re-raise an existing open item.
        still_open = [o for o in open_items if o.item.id not in closed]
        new_items = []
        if still_open and analysis.action_items:
            sims = cosine_matrix(
                [a.description for a in analysis.action_items],
                [o.item.description for o in still_open],
                self.embedder,
            )
            for ai, item in enumerate(analysis.action_items):
                best = int(sims[ai].argmax())
                score = float(sims[ai, best])
                target = still_open[best]
                compatible = (
                    not (item.owner and target.item.owner and item.owner != target.item.owner)
                    or score >= 0.7
                )
                if score >= self.cfg.reraise_threshold and compatible:
                    item.carried_over_from = target.item.id
                    self.store.update_fields(
                        target.item.id,
                        owner=item.owner if not target.item.owner else None,
                        deadline=item.deadline,
                    )
                    self.store.add_mention(
                        target.item.id,
                        analysis.meeting_id,
                        item.evidence[0].turn_id,
                        item.evidence[0].quote,
                        "reraised",
                        score,
                    )
                    report.reraised.append(
                        self._event(target, item.evidence[0].turn_id, item.evidence[0].quote, score)
                    )
                else:
                    new_items.append(item)
        else:
            new_items = list(analysis.action_items)

        self.store.add_items(new_items, series, analysis.meeting_id, analysis.meeting_date)
        self.store.update_analysis(analysis)
        report.new_item_ids = [i.id for i in new_items]
        reraised_ids = {e.item_id for e in report.reraised}
        report.mentioned = [e for e in report.mentioned if e.item_id not in reraised_ids]
        report.still_open = [
            self._event(o)
            for o in still_open
            if o.item.id not in {e.item_id for e in report.reraised}
        ]
        self.store.save_tracking(analysis.meeting_id, report.model_dump_json())
        return report

    @staticmethod
    def _event(
        o: TrackedItem, turn_id: str | None = None, quote: str | None = None, score: float = 0.0
    ) -> TrackEvent:
        return TrackEvent(
            item_id=o.item.id,
            description=o.item.description,
            owner=o.item.owner,
            from_meeting_id=o.meeting_id,
            turn_id=turn_id,
            quote=quote,
            score=round(score, 3),
        )
