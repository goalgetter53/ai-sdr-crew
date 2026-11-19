from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from ai_sdr_crew.hitl.app import create_app
from ai_sdr_crew.hitl.export import COLUMNS, safe_cell, to_csv
from ai_sdr_crew.hitl.store import ApprovalBlocked, ApprovalStore
from ai_sdr_crew.models import Account, Contact, RunResult, RunStatus
from helpers import make_fact, make_seq


def run_result(status: RunStatus = RunStatus.READY, seq=None, run_id: str = "acme-1") -> RunResult:
    return RunResult(
        run_id=run_id,
        account=Account(
            domain="acme.example",
            company="Acme",
            website="https://www.acme.example/",
            industry="x",
            employees=100,
            contact=Contact(
                first_name="Maya", last_name="Chen", title="CFO", email="maya@acme.example"
            ),
        ),
        prompt_version="v2",
        status=status,
        sequence=seq or make_seq(),
        facts=[make_fact()],
    )


@pytest.fixture
def store() -> ApprovalStore:
    return ApprovalStore(":memory:")


@pytest.fixture
def client(store: ApprovalStore) -> TestClient:
    return TestClient(create_app(store))


def edit_form(**over: str) -> dict[str, str]:
    seq = make_seq()
    form = {f"step{s.step}_{k}": getattr(s, k) for s in seq.steps for k in ("subject", "body")}
    form["linkedin_note"] = seq.linkedin_note
    form.update(over)
    return form


def test_halted_runs_are_never_queued(store: ApprovalStore) -> None:
    assert store.enqueue(run_result(RunStatus.HALTED)) is None
    assert store.items() == []


def test_needs_human_items_carry_flags(store: ApprovalStore) -> None:
    r = run_result(RunStatus.NEEDS_HUMAN)
    r.reason = "unresolved issues after 3 revisions"
    item = store.get(store.enqueue(r) or 0)
    assert item.flags[0].startswith("NEEDS_HUMAN")


def test_queue_page_and_detail_render(client: TestClient, store: ApprovalStore) -> None:
    item_id = store.enqueue(run_result())
    assert "Acme" in client.get("/").text
    page = client.get(f"/items/{item_id}")
    assert page.status_code == 200 and "Approve" in page.text and "[F1]" in page.text


def test_html_is_escaped(client: TestClient, store: ApprovalStore) -> None:
    seq = make_seq(step2="Hi Maya,\n\n<script>alert(1)</script> Open to it?")
    item_id = store.enqueue(run_result(seq=seq))
    assert "<script>alert(1)</script>" not in client.get(f"/items/{item_id}").text


def test_approve_then_export(client: TestClient, store: ApprovalStore) -> None:
    item_id = store.enqueue(run_result())
    r = client.post(
        f"/items/{item_id}/approve", data={"reviewer": "hussain"}, follow_redirects=False
    )
    assert r.status_code == 303
    assert store.get(item_id or 0).status == "approved"
    out = client.get("/export.csv?fmt=instantly")
    rows = list(csv.DictReader(io.StringIO(out.text)))
    assert rows[0]["email"] == "maya@acme.example"
    assert "[F1]" not in rows[0]["step1_body"] and "$5M" in rows[0]["step1_body"]
    assert list(rows[0]) == COLUMNS["instantly"]


def test_approval_blocked_by_lint_errors(client: TestClient, store: ApprovalStore) -> None:
    bad = make_seq(step1="Hi {{first_name}},\n\nThis is free. See https://x.example ok?")
    item_id = store.enqueue(run_result(seq=bad))
    r = client.post(f"/items/{item_id}/approve", data={"reviewer": "sdr"})
    assert r.status_code == 409 and "UNFILLED_TOKEN" in r.text
    assert store.get(item_id or 0).status == "pending"
    with pytest.raises(ApprovalBlocked):
        store.approve(item_id or 0, "sdr")


def test_reject_removes_from_export(client: TestClient, store: ApprovalStore) -> None:
    item_id = store.enqueue(run_result())
    client.post(f"/items/{item_id}/reject", data={"reviewer": "sdr", "note": "wrong persona"})
    item = store.get(item_id or 0)
    assert item.status == "rejected" and item.note == "wrong persona"
    assert len(client.get("/export.csv").text.strip().splitlines()) == 1  # header only
    assert client.post(f"/items/{item_id}/approve").status_code == 409


def test_edit_relints_and_marks_edited(client: TestClient, store: ApprovalStore) -> None:
    item_id = store.enqueue(run_result())
    r = client.post(f"/items/{item_id}/edit", data=edit_form(step2_body="Hi Maya,\n\nAct now! Ok?"))
    assert r.status_code == 200 and "SPAM_WORD" in r.text
    item = store.get(item_id or 0)
    assert item.edited and "Act now" in item.sequence.steps[1].body
    with pytest.raises(ApprovalBlocked):
        store.approve(item_id or 0, "sdr")
    client.post(f"/items/{item_id}/edit", data=edit_form())
    assert store.approve(item_id or 0, "sdr").status == "approved"


def test_invalid_edit_rejected(client: TestClient, store: ApprovalStore) -> None:
    item_id = store.enqueue(run_result())
    r = client.post(f"/items/{item_id}/edit", data=edit_form(step1_body="too short"))
    assert r.status_code == 422


def test_export_refuses_unapproved_items(store: ApprovalStore) -> None:
    item_id = store.enqueue(run_result())
    with pytest.raises(ValueError, match="refusing"):
        to_csv([store.get(item_id or 0)])


def test_smartlead_format(client: TestClient, store: ApprovalStore) -> None:
    store.approve(store.enqueue(run_result()) or 0, "sdr")
    header = client.get("/export.csv?fmt=smartlead").text.splitlines()[0]
    assert header.split(",") == COLUMNS["smartlead"]


def test_no_send_route_exists(client: TestClient) -> None:
    paths = {getattr(r, "path", "") for r in client.app.routes}
    assert not any("send" in p for p in paths)
    assert client.get("/healthz").json() == {"status": "ok"}


@given(st.text(min_size=1, max_size=40))
def test_csv_cells_never_start_with_formula(value: str) -> None:
    assert not safe_cell(value).startswith(("=", "+", "-", "@"))
