"""Deadline phrase detection and normalisation to ISO dates relative to the meeting date.

Normalisation is deliberately deterministic code rather than an LLM job: models are unreliable at
calendar arithmetic, and "by Friday" must mean the same date every time the pipeline runs.

Conventions (documented in docs/ARCHITECTURE.md):
* "by/before/on <weekday>" -> next occurrence of that weekday strictly after the meeting date.
* "next <weekday>" -> same as above when the meeting weekday is the same or later in the week
  (Monday meeting: "next Monday" = +7 days); otherwise the occurrence in the following week,
  flagged as ambiguous.
* "end of (the) week" -> Friday of the meeting week; "end of next week" -> Friday of next week.
* "today / tonight / this afternoon / EOD" -> meeting date; "tomorrow ..." -> +1 day.
* Everything else goes through `dateparser` with the meeting date as RELATIVE_BASE and future
  preference; unresolvable phrases ("after the next sync", "Q3", "at some point") return None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_WD = "|".join(WEEKDAYS)

_CORE = (
    rf"(?:end of (?:the )?(?:day|week|next week|month)|eod|eow|cob"
    rf"|today|tonight|this (?:afternoon|evening|morning|week)"
    rf"|tomorrow(?: morning| afternoon| evening)?"
    rf"|next (?:{_WD}|week)|(?:this )?(?:{_WD})(?: morning| afternoon| evening)?"
    rf"|(?:{_MONTHS}) \d{{1,2}}(?:st|nd|rd|th)?|the \d{{1,2}}(?:st|nd|rd|th)"
    rf"|in (?:a|one|two|three|\d+) (?:days?|weeks?))"
)
DEADLINE_RE = re.compile(
    rf"\b(?:(?:by|before|on|until|no later than|for)\s+)?(?P<core>{_CORE})\b", re.IGNORECASE
)
_BARE_OK = re.compile(
    r"^(?:today|tonight|tomorrow.*|this (?:afternoon|evening|morning)|eod|end of .*|in .*)$", re.I
)


@dataclass(frozen=True)
class DeadlineMatch:
    text: str
    start: int
    end: int


def find_deadline(sentence: str) -> DeadlineMatch | None:
    """Return the first deadline-like phrase in a sentence.

    Bare weekdays ("Thursday") only count when introduced by by/before/on/until, to avoid
    treating narrative references ("the mockups went out Wednesday") as deadlines.
    """
    for m in DEADLINE_RE.finditer(sentence):
        full = m.group(0)
        core = m.group("core")
        has_prep = full.lower() != core.lower()
        if has_prep or _BARE_OK.match(core) or core.lower().startswith("next "):
            return DeadlineMatch(full.strip(), m.start(), m.end())
    return None


@dataclass(frozen=True)
class NormalisedDeadline:
    value: date | None
    ambiguous: bool = False
    note: str = ""


def _next_weekday(base: date, wd: int) -> date:
    delta = (wd - base.weekday()) % 7
    return base + timedelta(days=delta or 7)


def normalise_deadline(text: str | None, meeting_date: date) -> NormalisedDeadline:
    if not text:
        return NormalisedDeadline(None)
    raw = text.strip().lower().rstrip(".?!")
    # ISO date passed straight through (some models return one)
    try:
        return NormalisedDeadline(date.fromisoformat(raw[:10]))
    except ValueError:
        pass
    phrase = re.sub(r"^(?:by|before|on|until|no later than|for)\s+", "", raw)
    phrase = (
        re.sub(r"\s+(?:morning|afternoon|evening)$", "", phrase)
        if not phrase.startswith(("this ", "tomorrow"))
        else phrase
    )

    if phrase in {"today", "tonight", "eod", "cob", "end of day", "end of the day"} or phrase in {
        "this afternoon",
        "this evening",
        "this morning",
    }:
        return NormalisedDeadline(meeting_date)
    if phrase.startswith("tomorrow"):
        return NormalisedDeadline(meeting_date + timedelta(days=1))
    if phrase in {"end of week", "end of the week", "eow", "this week"}:
        friday = meeting_date + timedelta(days=(4 - meeting_date.weekday()) % 7)
        return NormalisedDeadline(friday)
    if phrase in {"end of next week", "next week"}:
        friday_next = meeting_date + timedelta(days=(4 - meeting_date.weekday()) % 7 + 7)
        return NormalisedDeadline(
            friday_next,
            ambiguous=phrase == "next week",
            note="'next week' interpreted as Friday of next week",
        )
    if phrase in {"end of month", "end of the month"}:
        nxt = (meeting_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        return NormalisedDeadline(nxt - timedelta(days=1))
    m = re.fullmatch(rf"(next |this )?({_WD})", phrase)
    if m:
        wd = WEEKDAYS.index(m.group(2))
        d = _next_weekday(meeting_date, wd)
        if m.group(1) == "next " and wd > meeting_date.weekday():
            return NormalisedDeadline(
                d + timedelta(days=7),
                ambiguous=True,
                note=f"'next {m.group(2)}' could mean {d.isoformat()}",
            )
        return NormalisedDeadline(d)
    try:
        import dateparser

        parsed = dateparser.parse(
            phrase,
            settings={
                "RELATIVE_BASE": datetime.combine(meeting_date, datetime.min.time()),
                "PREFER_DATES_FROM": "future",
                "REQUIRE_PARTS": ["day"],
            },
        )
    except Exception:  # pragma: no cover - dateparser edge cases
        parsed = None
    if parsed:
        return NormalisedDeadline(parsed.date())
    return NormalisedDeadline(None, ambiguous=True, note=f"could not resolve '{text}' to a date")
