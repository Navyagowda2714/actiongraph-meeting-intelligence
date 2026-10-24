"""Exporters: Markdown report, JSON, and ICS calendar (one all-day VTODO-style event per deadline)."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone

from actiongraph.models import ActionItem, MeetingAnalysis


def to_json(analysis: MeetingAnalysis) -> str:
    return analysis.model_dump_json(indent=2)


def _evidence_md(item) -> str:
    return "; ".join(f'`{e.turn_id}` {e.speaker}: "{e.quote}"' for e in item.evidence)


def to_markdown(analysis: MeetingAnalysis) -> str:
    lines = [
        f"# {analysis.title}",
        "",
        f"**Date:** {analysis.meeting_date.isoformat()}  ",
        f"**Participants:** {', '.join(analysis.participants)}  ",
        f"**Extraction model:** {analysis.usage.model}",
        "",
        "## Summary",
        "",
        analysis.summary,
        "",
        "## Action items",
        "",
        "| # | Action | Owner | Due | Priority | Type | Confidence | Review |",
        "|---|--------|-------|-----|----------|------|-----------:|--------|",
    ]
    for i, a in enumerate(analysis.action_items, 1):
        review = "; ".join(a.review_reasons) if a.needs_review else ""
        status = " (done)" if a.status.value == "completed" else ""
        lines.append(
            f"| {i} | {a.description}{status} | {a.owner or '_unassigned_'} | "
            f"{a.deadline.isoformat() if a.deadline else (a.deadline_text or '-')} | {a.priority.value} | "
            f"{a.kind.value} | {a.confidence:.2f} | {review} |"
        )
    lines += ["", "### Evidence", ""]
    for i, a in enumerate(analysis.action_items, 1):
        lines.append(f"{i}. {_evidence_md(a)}")
    lines += ["", "## Decisions", ""]
    for d in analysis.decisions:
        lines.append(f"- **{d.summary}** - {_evidence_md(d)}")
    lines += ["", "## Risks and blockers", ""]
    for r in analysis.risks:
        lines.append(f"- [{r.kind}, {r.severity.value}] {r.description} - {_evidence_md(r)}")
    return "\n".join(lines) + "\n"


def _ics_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> list[str]:
    """RFC 5545 line folding at 75 octets."""
    out, cur = [], line
    while len(cur.encode()) > 75:
        cut = 75
        while len(cur[:cut].encode()) > 75:
            cut -= 1
        out.append(cur[:cut])
        cur = " " + cur[cut:]
    out.append(cur)
    return out


def to_ics(
    items: list[ActionItem], calendar_name: str = "ActionGraph", now: datetime | None = None
) -> str:
    """All-day events on each item's deadline, with a reminder the day before."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//ActionGraph//EN",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{_ics_escape(calendar_name)}",
    ]
    for it in items:
        if it.deadline is None:
            continue
        uid = hashlib.sha1(it.id.encode()).hexdigest()[:16] + "@actiongraph"
        start: date = it.deadline
        desc = f"Owner: {it.owner or 'unassigned'}\nPriority: {it.priority.value}\n" + "\n".join(
            f'{e.speaker}: "{e.quote}"' for e in it.evidence
        )
        event = [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{start.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{(start + timedelta(days=1)).strftime('%Y%m%d')}",
            f"SUMMARY:{_ics_escape(('[done] ' if it.status.value == 'completed' else '') + it.description)}",
            f"DESCRIPTION:{_ics_escape(desc)}",
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            "TRIGGER:-P1D",
            f"DESCRIPTION:{_ics_escape('Due tomorrow: ' + it.description)}",
            "END:VALARM",
            "END:VEVENT",
        ]
        for ln in event:
            lines += _fold(ln)
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"
