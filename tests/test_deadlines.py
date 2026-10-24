from __future__ import annotations

from datetime import date

import pytest

from actiongraph.extract.deadlines import find_deadline, normalise_deadline

MONDAY = date(2026, 3, 2)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("by Wednesday", date(2026, 3, 4)),
        ("by Friday", date(2026, 3, 6)),
        ("by next Monday", date(2026, 3, 9)),  # Monday meeting -> the following Monday
        ("by end of next week", date(2026, 3, 13)),
        ("end of the week", date(2026, 3, 6)),
        ("today", MONDAY),
        ("tonight", MONDAY),
        ("this afternoon", MONDAY),
        ("tomorrow morning", date(2026, 3, 3)),
        ("before Monday", date(2026, 3, 9)),
        ("by March 20th", date(2026, 3, 20)),
        ("in two weeks", date(2026, 3, 16)),
        ("2026-04-01", date(2026, 4, 1)),
    ],
)
def test_normalise_relative_to_meeting_date(phrase, expected):
    assert normalise_deadline(phrase, MONDAY).value == expected


def test_next_weekday_later_in_week_is_flagged_ambiguous():
    res = normalise_deadline("next Thursday", date(2026, 5, 12))  # Tuesday meeting
    assert res.value == date(2026, 5, 21)
    assert res.ambiguous and "2026-05-14" in res.note


def test_unresolvable_phrase_returns_none():
    res = normalise_deadline("after the next sync", MONDAY)
    assert res.value is None and res.ambiguous


def test_find_deadline_ignores_narrative_weekdays():
    assert find_deadline("The mockups went out Wednesday as planned.") is None
    m = find_deadline("I'll have the full mockups ready by Wednesday.")
    assert m is not None and m.text == "by Wednesday"
    assert find_deadline("I'll kick off the backfill tonight.").text == "tonight"
