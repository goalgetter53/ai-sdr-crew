"""FastAPI approval queue with a server-rendered UI (approve / edit / reject) and CSV export.

Run locally only (`sdr-crew serve` binds 127.0.0.1). There is deliberately no endpoint that
sends email: approved sequences leave the system only as a CSV a human imports elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from ai_sdr_crew.guardrails.citations import render
from ai_sdr_crew.hitl.export import ExportFormat, to_csv
from ai_sdr_crew.hitl.store import ApprovalBlocked, ApprovalStore, ItemStatus, QueueItem
from ai_sdr_crew.models import EmailStep, Issue, Sequence

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.filters["render"] = render


def create_app(store: ApprovalStore | None = None) -> FastAPI:
    store = store or ApprovalStore(os.environ.get("SDR_DB_PATH", "data/queue.db"))
    app = FastAPI(title="ai-sdr-crew approval queue", docs_url=None, redoc_url=None)
    app.state.store = store

    def _get(item_id: int) -> QueueItem:
        try:
            return store.get(item_id)
        except KeyError as e:
            raise HTTPException(404, "item not found") from e

    def _detail(
        request: Request, item: QueueItem, errors: list[Issue] | None = None, code: int = 200
    ) -> HTMLResponse:
        blocking, warnings = store.check(item)
        ctx: dict[str, Any] = {
            "item": item,
            "blocking": blocking,
            "warnings": warnings,
            "errors": errors or [],
        }
        return TEMPLATES.TemplateResponse(request, "item.html", ctx, status_code=code)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def queue(request: Request, status: ItemStatus | None = None) -> HTMLResponse:
        items = store.items(status)
        counts = {s: len(store.items(s)) for s in ("pending", "approved", "rejected")}
        return TEMPLATES.TemplateResponse(
            request, "queue.html", {"items": items, "status": status, "counts": counts}
        )

    @app.get("/items/{item_id}", response_class=HTMLResponse)
    def detail(request: Request, item_id: int) -> HTMLResponse:
        return _detail(request, _get(item_id))

    @app.post("/items/{item_id}/approve", response_model=None)
    def approve(
        request: Request, item_id: int, reviewer: Annotated[str, Form()] = "sdr"
    ) -> Response:
        item = _get(item_id)
        try:
            store.approve(item_id, reviewer)
        except ApprovalBlocked as e:
            return _detail(request, item, e.issues, code=409)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return RedirectResponse("/?status=pending", status_code=303)

    @app.post("/items/{item_id}/reject", response_model=None)
    def reject(
        item_id: int,
        reviewer: Annotated[str, Form()] = "sdr",
        note: Annotated[str, Form()] = "",
    ) -> Response:
        _get(item_id)
        try:
            store.reject(item_id, reviewer, note)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return RedirectResponse("/?status=pending", status_code=303)

    @app.post("/items/{item_id}/edit", response_model=None)
    def edit(
        request: Request,
        item_id: int,
        step1_subject: Annotated[str, Form()],
        step1_body: Annotated[str, Form()],
        step2_subject: Annotated[str, Form()],
        step2_body: Annotated[str, Form()],
        step3_subject: Annotated[str, Form()],
        step3_body: Annotated[str, Form()],
        linkedin_note: Annotated[str, Form()],
        reviewer: Annotated[str, Form()] = "sdr",
    ) -> Response:
        item = _get(item_id)
        subjects = [step1_subject, step2_subject, step3_subject]
        bodies = [step1_body, step2_body, step3_body]
        try:
            seq = Sequence(
                steps=[
                    EmailStep(
                        step=i + 1,
                        delay_days=item.sequence.steps[i].delay_days,
                        subject=subjects[i].strip(),
                        body=bodies[i].replace("\r\n", "\n").strip(),
                    )
                    for i in range(3)
                ],
                linkedin_note=linkedin_note.strip(),
            )
            updated, _ = store.edit(item_id, seq, reviewer)
        except ValueError as e:
            err = Issue(source="linter", code="INVALID_EDIT", message=str(e)[:300])
            return _detail(request, item, [err], code=422)
        return _detail(request, updated)

    @app.get("/export.csv")
    def export(fmt: ExportFormat = "instantly") -> Response:
        csv_text = to_csv(store.approved(), fmt)
        return Response(
            csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="approved_{fmt}.csv"'},
        )

    @app.get("/api/items")
    def api_items(status: ItemStatus | None = None) -> list[dict[str, Any]]:
        return [i.model_dump() for i in store.items(status)]

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory ai_sdr_crew.hitl.app:app_factory`."""
    return create_app()
