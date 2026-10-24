from __future__ import annotations

from actiongraph.models import CommitmentKind


def test_explicit_self_commitment_with_deadline(extract):
    r = extract(
        ("Priya Raman", "The flow is mostly done. I'll have the full mockups ready by Wednesday.")
    )
    [a] = r.action_items
    assert a.owner == "speaker" and a.kind == CommitmentKind.EXPLICIT
    assert a.deadline_text == "by Wednesday" and a.description == "Have the full mockups ready"
    assert a.evidence[0].turn_id == "t001"


def test_implicit_commitments_are_detected_and_labelled(extract):
    r = extract(
        ("Daniel Okafor", "Hmm, I wasn't aware. I'll take a look at the promo code reset bug."),
        (
            "Lucas Ferreira",
            "Let me dig into the country breakdown and I'll share it in the channel.",
        ),
    )
    kinds = {a.description: a.kind for a in r.action_items}
    assert kinds == {
        "Take a look at the promo code reset bug": CommitmentKind.IMPLICIT,
        "Dig into the country breakdown and share it in the channel": CommitmentKind.IMPLICIT,
    }


def test_request_and_acceptance_assigns_addressee_and_takes_reply_deadline(extract):
    r = extract(
        (
            "Maya Chen",
            "Daniel, can you write up a short note on the Adyen migration, so we have it for Q3?",
        ),
        ("Daniel Okafor", "Sure, I can do that by end of next week."),
    )
    [a] = r.action_items  # acceptance is not double counted as a second item
    assert a.owner == "Daniel Okafor" and a.kind == CommitmentKind.EXPLICIT
    assert a.deadline_text == "by end of next week"
    assert [e.turn_id for e in a.evidence] == ["t001", "t002"]
    assert a.description == "Write up a short note on the Adyen migration"


def test_hedged_acceptance_is_implicit_and_anaphora_resolved_from_same_turn(extract):
    r = extract(
        (
            "Maya Chen",
            "Lucas, one more thing. We need a dashboard tracking conversion by step. Could you set that up?",
        ),
        (
            "Lucas Ferreira",
            "Yeah, I can probably get a first version together. Let me see what I can do.",
        ),
    )
    [a] = r.action_items
    assert a.owner == "Lucas Ferreira" and a.kind == CommitmentKind.IMPLICIT
    assert "dashboard tracking conversion by step" in a.description


def test_followup_request_merges_into_volunteered_item(extract):
    r = extract(
        ("Grace Liu", "We could add schema checks. I can draft a proposal for that."),
        ("Hannah Weiss", "Please do. Can you get it to me by Friday?"),
        ("Grace Liu", "Yes, Friday works."),
    )
    [a] = r.action_items
    assert a.deadline_text == "by Friday" and a.kind == CommitmentKind.EXPLICIT
    assert {e.turn_id for e in a.evidence} == {"t001", "t002", "t003"}


def test_non_commitments_are_ignored(extract):
    r = extract(
        ("Lucas Ferreira", "I'll be honest, I don't think it matters much."),
        ("Daniel Okafor", "I'll be out Thursday afternoon, by the way."),
        (
            "Sofia Alvarez",
            "I need the staging environment refreshed before I can start the regression suite.",
        ),
        ("Priya Raman", "If the screens change again, we'll have to redo the usability test."),
        ("Maya Chen", "Lucas, did you get a chance to look at the data?"),
    )
    assert r.action_items == []


def test_team_and_unassigned_commitments(extract):
    r = extract(
        ("Daniel Okafor", "Yeah, that's on us. We'll get staging refreshed by Wednesday."),
        ("Sofia Alvarez", "We should probably update the help center articles at some point too."),
    )
    team, unassigned = r.action_items
    assert team.owner == "we" and team.description == "Get staging refreshed"
    assert unassigned.owner is None and unassigned.priority.value == "low"
    assert unassigned.description == "Update the help center articles"


def test_recaps_are_not_double_counted(extract):
    r = extract(
        ("Maya Chen", "Okay, to recap. Priya has mockups by Wednesday, and I'll chase the vendor.")
    )
    assert r.action_items == []


def test_decisions_and_risk_negation(extract):
    r = extract(
        ("Maya Chen", "Great, decision made. Stripe for the beta, Adyen revisit in Q3."),
        ("Maya Chen", "We got the credentials, so that blocker is gone."),
        ("Sofia Alvarez", "Also, I'm blocked on test accounts for the EU stores."),
    )
    assert [d.summary for d in r.decisions] == ["Stripe for the beta, Adyen revisit in Q3"]
    assert [(k.kind, k.evidence[0].turn_id) for k in r.risks] == [("blocker", "t003")]
