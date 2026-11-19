"""SQLite-backed approval queue. Nothing leaves this queue except via an explicit CSV export of
human-approved items; there is no send path anywhere in the codebase."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_sdr_crew.config import Seller
from ai_sdr_crew.guardrails.citations import check_citations
from ai_sdr_crew.guardrails.linter import lint_sequence
from ai_sdr_crew.guardrails.output_guard import guard_sequence
from ai_sdr_crew.models import Contact, Fact, Issue, RunResult, RunStatus, Sequence

ItemStatus = Literal["pending", "approved", "rejected"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    company TEXT NOT NULL,
    website TEXT NOT NULL,
    contact_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    run_status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    flags_json TEXT NOT NULL DEFAULT '[]',
    sequence_json TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '[]',
    edited INTEGER NOT NULL DEFAULT 0,
    reviewer TEXT,
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class ApprovalBlocked(Exception):
    def __init__(self, issues: list[Issue]) -> None:
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in issues))
        self.issues = issues


class QueueItem(BaseModel):
    id: int
    run_id: str
    domain: str
    company: str
    website: str
    contact: Contact
    status: ItemStatus
    run_status: str
    reason: str
    flags: list[str] = Field(default_factory=list)
    sequence: Sequence
    facts: list[Fact] = Field(default_factory=list)
    edited: bool
    reviewer: str | None = None
    note: str | None = None
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ApprovalStore:
    def __init__(self, path: Path | str = ":memory:", seller: Seller | None = None) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.seller = seller or Seller()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def enqueue(self, result: RunResult) -> int | None:
        """Queue READY and NEEDS_HUMAN runs. HALTED runs (DNC, customers) are never queued."""
        if result.status == RunStatus.HALTED or result.sequence is None:
            return None
        flags = [f"{i.code}: {i.message}" for i in result.issues if i.severity == "error"]
        if result.status == RunStatus.NEEDS_HUMAN and result.reason:
            flags.insert(0, f"NEEDS_HUMAN: {result.reason}")
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO items (run_id, domain, company, website, contact_json, status, "
                "run_status, reason, flags_json, sequence_json, facts_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.run_id,
                    result.account.domain,
                    result.account.company,
                    result.account.website,
                    result.account.contact.model_dump_json(),
                    result.status.value,
                    result.reason,
                    json.dumps(flags),
                    result.sequence.model_dump_json(),
                    json.dumps([f.model_dump() for f in result.facts]),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def _row(self, row: sqlite3.Row) -> QueueItem:
        return QueueItem(
            id=row["id"],
            run_id=row["run_id"],
            domain=row["domain"],
            company=row["company"],
            website=row["website"],
            contact=Contact.model_validate_json(row["contact_json"]),
            status=row["status"],
            run_status=row["run_status"],
            reason=row["reason"],
            flags=json.loads(row["flags_json"]),
            sequence=Sequence.model_validate_json(row["sequence_json"]),
            facts=[Fact.model_validate(f) for f in json.loads(row["facts_json"])],
            edited=bool(row["edited"]),
            reviewer=row["reviewer"],
            note=row["note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def items(self, status: ItemStatus | None = None) -> list[QueueItem]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM items WHERE status = ? ORDER BY id", (status,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM items ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    def get(self, item_id: int) -> QueueItem:
        with self._lock:
            row = self._conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(item_id)
        return self._row(row)

    def check(self, item: QueueItem) -> tuple[list[Issue], list[Issue]]:
        """(blocking, warnings). Lint and output-guard errors block approval; citation issues
        are warnings once a human has edited the copy (the human vouches for it)."""
        blocking = [i for i in lint_sequence(item.sequence) if i.severity == "error"]
        blocking += guard_sequence(item.sequence, (self.seller.domain, item.domain))
        cite_issues, _ = check_citations(item.sequence, item.facts, item.company)
        if item.edited:
            return blocking, cite_issues
        return blocking + cite_issues, []

    def _update(self, item_id: int, **fields: object) -> QueueItem:
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE items SET {cols} WHERE id = ?", (*fields.values(), item_id))
            self._conn.commit()
        return self.get(item_id)

    def approve(self, item_id: int, reviewer: str) -> QueueItem:
        item = self.get(item_id)
        if item.status != "pending":
            raise ValueError(f"item {item_id} is {item.status}, not pending")
        blocking, _ = self.check(item)
        if blocking:
            raise ApprovalBlocked(blocking)
        return self._update(item_id, status="approved", reviewer=reviewer)

    def reject(self, item_id: int, reviewer: str, note: str = "") -> QueueItem:
        item = self.get(item_id)
        if item.status != "pending":
            raise ValueError(f"item {item_id} is {item.status}, not pending")
        return self._update(item_id, status="rejected", reviewer=reviewer, note=note)

    def edit(
        self, item_id: int, sequence: Sequence, reviewer: str
    ) -> tuple[QueueItem, list[Issue]]:
        item = self.get(item_id)
        if item.status != "pending":
            raise ValueError(f"item {item_id} is {item.status}, not pending")
        updated = self._update(
            item_id, sequence_json=sequence.model_dump_json(), edited=1, reviewer=reviewer
        )
        blocking, warnings = self.check(updated)
        return updated, blocking + warnings

    def approved(self) -> list[QueueItem]:
        return self.items("approved")
