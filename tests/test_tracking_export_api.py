from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from actiongraph.evaluation import PRF, run_eval
from actiongraph.export import to_ics, to_markdown
from actiongraph.ingest import load_transcript
from actiongraph.llm import OfflineProvider
from actiongraph.models import ItemStatus
from actiongraph.service import ActionGraphService
from actiongraph.tracking.store import Store
from actiongraph.web.app import create_app

from .conftest import GOLD, SAMPLES


@pytest.fixture(scope="module")
def series():
    svc = ActionGraphService(Store(":memory:"), OfflineProvider())
    results = [
        svc.ingest_transcript(load_transcript(p, series="atlas"))
        for p in sorted(SAMPLES.glob("atlas_sync_*"))
    ]
    return svc, results


def test_completions_detected_across_meetings(series):
    _, (_m1, m2, m3) = series
    assert len(m2.tracking.completed) == 5 and len(m3.tracking.completed) == 9
    promo = next(e for e in m2.tracking.completed if "promo" in e.description)
    assert promo.turn_id == "t003" and "is fixed" in promo.quote


def test_reraised_item_is_linked_and_gets_an_owner(series):
    svc, (_, m2, m3) = series
    help_item = next(i for i in m3.analysis.action_items if "help center" in i.description)
    original = next(i for i in m2.analysis.action_items if "help center" in i.description)
    assert help_item.carried_over_from == original.id
    tracked = svc.store.get_item(original.id)
    assert tracked is not None and tracked.item.owner == "Sofia Alvarez"
    assert help_item.id not in m3.tracking.new_item_ids


def test_upcoming_deadline_notifications(series):
    svc, _ = series
    due = svc.store.upcoming(date(2026, 3, 17), days=3, series="atlas")
    assert [t.item.deadline.isoformat() for t in due] == ["2026-03-18", "2026-03-19", "2026-03-20"]
    assert all(t.item.status == ItemStatus.OPEN for t in due)


def test_markdown_and_ics_exports(series):
    _, (m1, _, _) = series
    md = to_markdown(m1.analysis)
    assert "| # | Action | Owner | Due |" in md and "`t017`" in md
    ics = to_ics(m1.analysis.action_items, "Atlas")
    lines = ics.split("\r\n")
    assert ics.count("BEGIN:VEVENT") == sum(1 for a in m1.analysis.action_items if a.deadline)
    assert "DTSTART;VALUE=DATE:20260304" in lines
    assert all(len(ln.encode()) <= 75 for ln in lines)


def test_api_upload_view_complete_export(tmp_path):
    client = TestClient(create_app(tmp_path / "t.db", OfflineProvider()))
    body = (SAMPLES / "atlas_sync_2026-03-09.json").read_bytes()
    r = client.post(
        "/api/meetings",
        files={"file": ("sync.json", body, "application/json")},
        data={"series": "demo"},
    )
    assert r.status_code == 201
    mid = r.json()["meeting_id"]
    data = client.get(f"/api/meetings/{mid}").json()
    item = data["analysis"]["action_items"][0]
    assert client.patch(f"/api/items/{item['id']}", json={"status": "completed"}).status_code == 200
    again = client.get(f"/api/meetings/{mid}").json()["analysis"]["action_items"][0]
    assert again["status"] == "completed"
    assert (
        client.get(f"/api/meetings/{mid}/export/ics")
        .headers["content-type"]
        .startswith("text/calendar")
    )
    assert client.get("/api/meetings/nope").status_code == 404
    bad = client.post(
        "/api/meetings", files={"file": ("x.docx", b"zz", "application/octet-stream")}
    )
    assert bad.status_code == 422
    assert "ActionGraph" in client.get("/").text


def test_eval_runs_and_prf_math():
    p = PRF(tp=8, fp=2, fn=2)
    assert p.precision == pytest.approx(0.8) and p.f1 == pytest.approx(0.8)
    res = run_eval(SAMPLES, GOLD, OfflineProvider())
    assert res["dev"]["meetings"] == 3 and res["heldout"]["meetings"] == 3
    assert res["all"]["evidence_valid"]["value"] == 1.0
    assert 0.0 < res["heldout"]["actions"]["f1"] <= 1.0
    json.dumps(res)  # serialisable
