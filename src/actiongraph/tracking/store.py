"""SQLite persistence for meetings, tracked action items and cross-meeting mentions."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

from actiongraph.models import ActionItem, ItemStatus, MeetingAnalysis, Transcript

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    series TEXT NOT NULL,
    title TEXT NOT NULL,
    meeting_date TEXT NOT NULL,
    transcript_json TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    tracking_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    series TEXT NOT NULL,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    meeting_date TEXT NOT NULL,
    description TEXT NOT NULL,
    owner TEXT,
    deadline TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    completed_meeting_id TEXT,
    completed_at TEXT,
    item_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_series_status ON items(series, status);
CREATE INDEX IF NOT EXISTS idx_items_deadline ON items(deadline);
CREATE TABLE IF NOT EXISTS mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    meeting_id TEXT NOT NULL,
    turn_id TEXT,
    quote TEXT,
    kind TEXT NOT NULL,             -- mentioned | completed | reraised | manual
    score REAL,
    created_at TEXT NOT NULL
);
"""


class TrackedItem(BaseModel):
    item: ActionItem
    series: str
    meeting_id: str
    meeting_date: date
    completed_meeting_id: str | None = None
    completed_at: str | None = None


class Mention(BaseModel):
    item_id: str
    meeting_id: str
    turn_id: str | None
    quote: str | None
    kind: str
    score: float | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ meetings
    def save_meeting(self, transcript: Transcript, analysis: MeetingAnalysis) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO meetings(id, series, title, meeting_date, transcript_json,"
                " analysis_json, tracking_json, created_at) VALUES (?,?,?,?,?,?,NULL,?)",
                (
                    analysis.meeting_id,
                    analysis.series,
                    analysis.title,
                    analysis.meeting_date.isoformat(),
                    transcript.model_dump_json(),
                    analysis.model_dump_json(),
                    _now(),
                ),
            )

    def update_analysis(self, analysis: MeetingAnalysis) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE meetings SET analysis_json=? WHERE id=?",
                (analysis.model_dump_json(), analysis.meeting_id),
            )

    def save_tracking(self, meeting_id: str, report_json: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE meetings SET tracking_json=? WHERE id=?", (report_json, meeting_id)
            )

    def list_meetings(self, series: str | None = None) -> list[dict]:
        q = "SELECT id, series, title, meeting_date FROM meetings"
        args: tuple = ()
        if series:
            q += " WHERE series=?"
            args = (series,)
        rows = self.conn.execute(q + " ORDER BY meeting_date, created_at", args).fetchall()
        return [dict(r) for r in rows]

    def get_meeting(
        self, meeting_id: str
    ) -> tuple[Transcript, MeetingAnalysis, dict | None] | None:
        row = self.conn.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone()
        if row is None:
            return None
        tracking = json.loads(row["tracking_json"]) if row["tracking_json"] else None
        return (
            Transcript.model_validate_json(row["transcript_json"]),
            MeetingAnalysis.model_validate_json(row["analysis_json"]),
            tracking,
        )

    def delete_meeting(self, meeting_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))

    # ------------------------------------------------------------------ items
    def add_items(
        self, items: list[ActionItem], series: str, meeting_id: str, meeting_date: date
    ) -> None:
        with self._lock, self.conn:
            for it in items:
                self.conn.execute(
                    "INSERT OR REPLACE INTO items(id, series, meeting_id, meeting_date, description,"
                    " owner, deadline, status, item_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        it.id,
                        series,
                        meeting_id,
                        meeting_date.isoformat(),
                        it.description,
                        it.owner,
                        it.deadline.isoformat() if it.deadline else None,
                        it.status.value,
                        it.model_dump_json(),
                        _now(),
                    ),
                )

    def _row_to_tracked(self, r: sqlite3.Row) -> TrackedItem:
        item = ActionItem.model_validate_json(r["item_json"])
        item.status = ItemStatus(r["status"])
        item.owner = r["owner"]
        item.deadline = date.fromisoformat(r["deadline"]) if r["deadline"] else None
        return TrackedItem(
            item=item,
            series=r["series"],
            meeting_id=r["meeting_id"],
            meeting_date=date.fromisoformat(r["meeting_date"]),
            completed_meeting_id=r["completed_meeting_id"],
            completed_at=r["completed_at"],
        )

    def get_item(self, item_id: str) -> TrackedItem | None:
        r = self.conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return self._row_to_tracked(r) if r else None

    def items(
        self,
        series: str | None = None,
        status: ItemStatus | None = None,
        meeting_id: str | None = None,
    ) -> list[TrackedItem]:
        clauses, args = [], []
        for col, val in (
            ("series", series),
            ("status", status.value if status else None),
            ("meeting_id", meeting_id),
        ):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        q = "SELECT * FROM items" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        rows = self.conn.execute(q + " ORDER BY meeting_date, id", args).fetchall()
        return [self._row_to_tracked(r) for r in rows]

    def open_items_before(self, series: str, meeting_date: date) -> list[TrackedItem]:
        rows = self.conn.execute(
            "SELECT * FROM items WHERE series=? AND status='open' AND meeting_date < ? ORDER BY meeting_date",
            (series, meeting_date.isoformat()),
        ).fetchall()
        return [self._row_to_tracked(r) for r in rows]

    def set_status(self, item_id: str, status: ItemStatus, meeting_id: str | None = None) -> bool:
        with self._lock, self.conn:
            cur = self.conn.execute(
                "UPDATE items SET status=?, completed_meeting_id=?, completed_at=?, updated_at=? WHERE id=?",
                (
                    status.value,
                    meeting_id if status == ItemStatus.COMPLETED else None,
                    _now() if status == ItemStatus.COMPLETED else None,
                    _now(),
                    item_id,
                ),
            )
        return cur.rowcount > 0

    def update_fields(
        self, item_id: str, *, owner: str | None = None, deadline: date | None = None
    ) -> None:
        with self._lock, self.conn:
            if owner is not None:
                self.conn.execute(
                    "UPDATE items SET owner=?, updated_at=? WHERE id=?", (owner, _now(), item_id)
                )
            if deadline is not None:
                self.conn.execute(
                    "UPDATE items SET deadline=?, updated_at=? WHERE id=?",
                    (deadline.isoformat(), _now(), item_id),
                )

    # ------------------------------------------------------------------ mentions
    def add_mention(
        self,
        item_id: str,
        meeting_id: str,
        turn_id: str | None,
        quote: str | None,
        kind: str,
        score: float | None = None,
    ) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO mentions(item_id, meeting_id, turn_id, quote, kind, score, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (item_id, meeting_id, turn_id, quote, kind, score, _now()),
            )

    def mentions(self, item_id: str) -> list[Mention]:
        rows = self.conn.execute(
            "SELECT * FROM mentions WHERE item_id=? ORDER BY id", (item_id,)
        ).fetchall()
        return [
            Mention(
                item_id=r["item_id"],
                meeting_id=r["meeting_id"],
                turn_id=r["turn_id"],
                quote=r["quote"],
                kind=r["kind"],
                score=r["score"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ notifications
    def upcoming(self, today: date, days: int = 7, series: str | None = None) -> list[TrackedItem]:
        """Open items due within `days` of `today`, including overdue ones, soonest first."""
        q = "SELECT * FROM items WHERE status='open' AND deadline IS NOT NULL AND deadline <= ?"
        args: list = [(today + timedelta(days=days)).isoformat()]
        if series:
            q += " AND series=?"
            args.append(series)
        rows = self.conn.execute(q + " ORDER BY deadline", args).fetchall()
        return [self._row_to_tracked(r) for r in rows]
