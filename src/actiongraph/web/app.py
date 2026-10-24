"""FastAPI app: JSON API + a single-page dashboard (Jinja2 shell + vanilla JS)."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from actiongraph import __version__
from actiongraph.config import get_settings
from actiongraph.export import to_ics, to_json, to_markdown
from actiongraph.llm import LLMClient, get_llm
from actiongraph.models import ItemStatus
from actiongraph.service import ActionGraphService
from actiongraph.tracking.store import Store

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class StatusUpdate(BaseModel):
    status: ItemStatus


def create_app(db_path: str | Path | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = get_settings()
    store = Store(db_path or settings.db_path)
    service = ActionGraphService(store, llm or get_llm())
    app = FastAPI(
        title="ActionGraph",
        version=__version__,
        description="Turn meetings into meaningful action.",
    )
    app.state.service = service
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def _meeting_summary(m: dict[str, Any]) -> dict[str, Any]:
        got = service.live_analysis(m["id"])
        assert got is not None
        _, analysis, _ = got
        items = analysis.action_items
        return {
            **m,
            "actions": len(items),
            "open": sum(i.status == ItemStatus.OPEN for i in items),
            "review": sum(i.needs_review for i in items),
            "risks": len(analysis.risks),
        }

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {"version": __version__})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model": getattr(service.pipeline.llm, "model", "unknown")}

    @app.get("/api/meetings")
    def list_meetings(series: str | None = None) -> list[dict[str, Any]]:
        return [_meeting_summary(m) for m in store.list_meetings(series)]

    @app.post("/api/meetings", status_code=201)
    async def upload_meeting(
        file: UploadFile = File(...),
        title: str | None = Form(None),
        meeting_date: str | None = Form(None),
        series: str | None = Form(None),
        speaker_map: str | None = Form(None),
    ) -> dict[str, Any]:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (10 MB max)")
        try:
            mdate = date.fromisoformat(meeting_date) if meeting_date else None
            smap = json.loads(speaker_map) if speaker_map else None
            result = service.ingest_bytes(
                data,
                file.filename or "upload.txt",
                title=title or None,
                meeting_date=mdate,
                series=series or None,
                speaker_map=smap,
            )
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(422, f"Could not process transcript: {exc}") from exc
        return {
            "meeting_id": result.analysis.meeting_id,
            "analysis": result.analysis.model_dump(mode="json"),
            "tracking": result.tracking.model_dump(mode="json"),
        }

    @app.get("/api/meetings/{meeting_id}")
    def get_meeting(meeting_id: str) -> dict[str, Any]:
        got = service.live_analysis(meeting_id)
        if got is None:
            raise HTTPException(404, "Meeting not found")
        transcript, analysis, tracking = got
        return {
            "transcript": transcript.model_dump(mode="json"),
            "analysis": analysis.model_dump(mode="json"),
            "tracking": tracking,
        }

    @app.delete("/api/meetings/{meeting_id}", status_code=204)
    def delete_meeting(meeting_id: str) -> Response:
        store.delete_meeting(meeting_id)
        return Response(status_code=204)

    @app.get("/api/meetings/{meeting_id}/export/{fmt}")
    def export(meeting_id: str, fmt: str) -> Response:
        got = service.live_analysis(meeting_id)
        if got is None:
            raise HTTPException(404, "Meeting not found")
        _, analysis, _ = got
        if fmt == "md":
            body, media = to_markdown(analysis), "text/markdown"
        elif fmt == "json":
            body, media = to_json(analysis), "application/json"
        elif fmt == "ics":
            body, media = to_ics(analysis.action_items, analysis.title), "text/calendar"
        else:
            raise HTTPException(400, "fmt must be one of md, json, ics")
        return Response(
            body,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{meeting_id}.{fmt}"'},
        )

    @app.get("/api/items")
    def list_items(
        series: str | None = None, status: ItemStatus | None = None
    ) -> list[dict[str, Any]]:
        return [t.model_dump(mode="json") for t in store.items(series=series, status=status)]

    @app.get("/api/items/{item_id}/history")
    def item_history(item_id: str) -> list[dict[str, Any]]:
        return [m.model_dump() for m in store.mentions(item_id)]

    @app.patch("/api/items/{item_id}")
    def update_item(item_id: str, body: StatusUpdate) -> dict[str, str]:
        target = service.set_item_status(item_id, body.status)
        if target is None:
            raise HTTPException(404, "Item not found")
        return {"id": target, "status": body.status.value}

    @app.get("/api/notifications")
    def notifications(
        today: date | None = None, days: int = 7, series: str | None = None
    ) -> list[dict[str, Any]]:
        ref = today or date.today()
        out = []
        for t in store.upcoming(ref, days, series):
            assert t.item.deadline is not None
            delta = (t.item.deadline - ref).days
            out.append({**t.model_dump(mode="json"), "days_left": delta, "overdue": delta < 0})
        return out

    return app
