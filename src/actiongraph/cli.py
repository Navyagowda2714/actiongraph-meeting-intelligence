"""`actiongraph` command line interface."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import typer

from actiongraph.config import configure_logging, get_settings
from actiongraph.export import to_ics, to_json, to_markdown
from actiongraph.ingest import load_transcript
from actiongraph.ingest.parsers import AUDIO_EXTS, PARSERS
from actiongraph.llm import get_llm
from actiongraph.models import MeetingAnalysis
from actiongraph.service import ActionGraphService, IngestResult
from actiongraph.tracking.store import Store

app = typer.Typer(add_completion=False, help="ActionGraph: turn meetings into meaningful action.")
ROOT = Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    """Sample/gold data: $ACTIONGRAPH_DATA_DIR, ./data, or the source checkout's data/."""
    env = os.getenv("ACTIONGRAPH_DATA_DIR")
    if env:
        return Path(env)
    for cand in (Path.cwd() / "data", ROOT / "data"):
        if (cand / "samples").exists():
            return cand
    return ROOT / "data"


SAMPLES = _data_dir() / "samples"
GOLD = _data_dir() / "gold"


def _print_analysis(a: MeetingAnalysis) -> None:
    typer.secho(f"\n{a.title}  ({a.meeting_date})", bold=True)
    typer.echo(f"Participants: {', '.join(a.participants)}")
    typer.echo(f"Summary: {a.summary}\n")
    typer.secho(f"Action items ({len(a.action_items)})", bold=True)
    for i in a.action_items:
        due = (
            i.deadline.isoformat()
            if i.deadline
            else (f'"{i.deadline_text}"' if i.deadline_text else "-")
        )
        flag = "  [REVIEW: " + "; ".join(i.review_reasons) + "]" if i.needs_review else ""
        carry = f"  (re-raises {i.carried_over_from})" if i.carried_over_from else ""
        turns = ",".join(dict.fromkeys(e.turn_id for e in i.evidence))
        typer.echo(
            f"  - [{i.priority.value:<6}] {i.description}\n"
            f"      owner={i.owner or 'UNASSIGNED'}  due={due}  {i.kind.value}  "
            f"conf={i.confidence:.2f}  evidence={turns}{flag}{carry}"
        )
    typer.secho(f"Decisions ({len(a.decisions)})", bold=True)
    for d in a.decisions:
        typer.echo(f"  - {d.summary}  [{','.join(e.turn_id for e in d.evidence)}]")
    typer.secho(f"Risks & blockers ({len(a.risks)})", bold=True)
    for r in a.risks:
        typer.echo(f"  - ({r.kind}/{r.severity.value}) {r.description}  [{r.evidence[0].turn_id}]")


def _print_tracking(res: IngestResult) -> None:
    tr = res.tracking
    if not (tr.completed or tr.mentioned or tr.reraised or tr.still_open):
        return
    typer.secho("Cross-meeting tracking", bold=True)
    for e in tr.completed:
        typer.echo(f'  [done]    {e.description} ({e.owner}) <- {e.turn_id}: "{e.quote}"')
    for e in tr.reraised:
        typer.echo(f'  [linked]  {e.description} <- {e.turn_id}: "{e.quote}"')
    for e in tr.mentioned:
        typer.echo(f'  [mention] {e.description} <- {e.turn_id}: "{e.quote}"')
    for e in tr.still_open:
        typer.echo(f"  [open]    {e.description} ({e.owner or 'unassigned'})")


def _service(db: Path | None, provider: str | None) -> ActionGraphService:
    return ActionGraphService(Store(db or get_settings().db_path), get_llm(provider))


@app.command()
def analyze(
    path: Path = typer.Argument(
        ..., exists=True, help="Transcript (txt/vtt/srt/json/pdf) or audio"
    ),
    title: str | None = typer.Option(None),
    meeting_date: str | None = typer.Option(None, "--date", help="ISO date, e.g. 2026-03-02"),
    series: str | None = typer.Option(None, help="Meeting series for cross-meeting tracking"),
    provider: str | None = typer.Option(None, help="anthropic | openai | ollama | offline"),
    fmt: str = typer.Option("text", "--format", help="text | md | json | ics"),
    out: Path | None = typer.Option(None, help="Write the export to this file"),
    db: Path | None = typer.Option(None, help="SQLite DB (enables tracking); default: in-memory"),
) -> None:
    """Analyze one meeting transcript."""
    configure_logging()
    svc = ActionGraphService(Store(db or ":memory:"), get_llm(provider))
    res = svc.ingest_transcript(
        load_transcript(
            path,
            title=title,
            meeting_date=date.fromisoformat(meeting_date) if meeting_date else None,
            series=series,
        )
    )
    a = res.analysis
    if fmt == "text":
        _print_analysis(a)
        _print_tracking(res)
        return
    body = {"md": to_markdown, "json": to_json}.get(fmt, lambda x: to_ics(x.action_items, x.title))(
        a
    )
    if out:
        out.write_text(body)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(body)


@app.command("ingest-series")
def ingest_series(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    series: str = typer.Option(..., help="Series name"),
    db: Path | None = typer.Option(None),
    provider: str | None = typer.Option(None),
    pattern: str = typer.Option("*", help="Glob to select transcripts"),
) -> None:
    """Analyze every transcript in a folder (chronologically) and track items across them."""
    configure_logging("WARNING")
    svc = _service(db, provider)
    exts = set(PARSERS) | {".pdf"} | AUDIO_EXTS
    files = [
        p
        for p in directory.glob(pattern)
        if p.suffix in exts and not p.name.endswith(".speakers.json")
    ]
    transcripts = sorted(
        (load_transcript(p, series=series) for p in files), key=lambda t: t.meeting_date
    )
    for t in transcripts:
        res = svc.ingest_transcript(t)
        _print_analysis(res.analysis)
        _print_tracking(res)


@app.command()
def notify(
    today: str | None = typer.Option(None, help="Reference date (default: today)"),
    days: int = typer.Option(7, help="Look-ahead window in days"),
    series: str | None = typer.Option(None),
    db: Path | None = typer.Option(None),
) -> None:
    """List open items that are overdue or due within the window."""
    store = Store(db or get_settings().db_path)
    ref = date.fromisoformat(today) if today else date.today()
    items = store.upcoming(ref, days, series)
    if not items:
        typer.echo("Nothing due.")
    for t in items:
        assert t.item.deadline is not None
        delta = (t.item.deadline - ref).days
        when = (
            f"OVERDUE by {-delta}d"
            if delta < 0
            else ("due TODAY" if delta == 0 else f"due in {delta}d")
        )
        typer.echo(
            f"  {when:<16} {t.item.deadline}  {t.item.owner or 'unassigned':<16} {t.item.description}"
        )


@app.command()
def export(
    meeting_id: str,
    fmt: str = typer.Option("md", "--format", help="md | json | ics"),
    out: Path | None = typer.Option(None),
    db: Path | None = typer.Option(None),
) -> None:
    """Export a stored meeting."""
    svc = _service(db, "offline")
    got = svc.live_analysis(meeting_id)
    if got is None:
        raise typer.BadParameter(f"unknown meeting {meeting_id}")
    a = got[1]
    body = {"md": to_markdown, "json": to_json}.get(fmt, lambda x: to_ics(x.action_items, x.title))(
        a
    )
    if out:
        out.write_text(body)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(body)


@app.command()
def demo(
    db: Path = typer.Option(Path("demo.db"), help="SQLite DB to (re)create"),
    provider: str | None = typer.Option(None),
) -> None:
    """Ingest the bundled 3-meeting series, show tracking and deadline notifications."""
    configure_logging("WARNING")
    if db.exists():
        db.unlink()
    svc = _service(db, provider)
    files = sorted(SAMPLES.glob("atlas_sync_*"))
    for p in files:
        res = svc.ingest_transcript(load_transcript(p, series="atlas-checkout"))
        _print_analysis(res.analysis)
        _print_tracking(res)
    typer.secho("\nNotifications as of 2026-03-17 (next 7 days)", bold=True)
    for t in svc.store.upcoming(date(2026, 3, 17), 7, "atlas-checkout"):
        assert t.item.deadline is not None
        delta = (t.item.deadline - date(2026, 3, 17)).days
        typer.echo(
            f"  {'OVERDUE' if delta < 0 else f'in {delta}d':<8} {t.item.deadline}  "
            f"{t.item.owner or 'unassigned':<15} {t.item.description}"
        )
    typer.echo(
        f"\nStored in {db}. Run `actiongraph serve --db {db}` and open http://127.0.0.1:8000/?asof=2026-03-17"
    )


@app.command("eval")
def run_evaluation(
    provider: str | None = typer.Option("offline"),
    out_dir: Path = typer.Option(Path("evals/results")),
) -> None:
    """Score extraction + tracking against the gold annotations in data/gold."""
    from actiongraph.evaluation import results_markdown, run_eval

    configure_logging("WARNING")
    res = run_eval(SAMPLES, GOLD, get_llm(provider))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(res, indent=2))
    md = results_markdown(res)
    (out_dir / "latest.md").write_text(md)
    typer.echo(md)
    for meeting, errs in res["errors"].items():
        for e in errs:
            typer.echo(f"  {meeting}: {e}")
    typer.echo(f"\nWrote {out_dir / 'latest.json'} and latest.md")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    db: Path | None = typer.Option(None),
    provider: str | None = typer.Option(None),
) -> None:
    """Run the web dashboard + API."""
    import uvicorn

    from actiongraph.web.app import create_app

    configure_logging()
    uvicorn.run(create_app(db, get_llm(provider)), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
