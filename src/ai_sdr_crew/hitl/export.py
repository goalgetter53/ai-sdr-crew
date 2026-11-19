"""CSV export of human-approved sequences for Instantly or Smartlead lead import.

Copy is exported as custom variables (step1_subject, step1_body, ...); map them in the sending
tool's sequence editor, e.g. {{step1_body}}. Citation markers are stripped. Cells that a
spreadsheet would interpret as formulas are neutralised (CSV injection).
"""

from __future__ import annotations

import csv
import io
from typing import Literal

from ai_sdr_crew.guardrails.citations import render
from ai_sdr_crew.hitl.store import QueueItem

ExportFormat = Literal["instantly", "smartlead"]

COPY_FIELDS = [
    "step1_subject", "step1_body", "step2_subject", "step2_body",
    "step3_subject", "step3_body", "linkedin_note",
]  # fmt: skip
COLUMNS: dict[str, list[str]] = {
    "instantly": ["email", "first_name", "last_name", "company_name", "website", *COPY_FIELDS],
    "smartlead": ["first_name", "last_name", "email", "company_name", "website", *COPY_FIELDS],
}
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: str) -> str:
    return "'" + value if value.startswith(FORMULA_PREFIXES) else value


def export_rows(items: list[QueueItem]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        if item.status != "approved":
            raise ValueError(f"refusing to export item {item.id} with status {item.status}")
        row = {
            "email": item.contact.email,
            "first_name": item.contact.first_name or "",
            "last_name": item.contact.last_name,
            "company_name": item.company,
            "website": item.website,
            "linkedin_note": render(item.sequence.linkedin_note),
        }
        for step in item.sequence.steps:
            row[f"step{step.step}_subject"] = render(step.subject)
            row[f"step{step.step}_body"] = render(step.body)
        rows.append({k: safe_cell(v) for k, v in row.items()})
    return rows


def to_csv(items: list[QueueItem], fmt: ExportFormat = "instantly") -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS[fmt], extrasaction="ignore")
    writer.writeheader()
    for row in export_rows(items):
        writer.writerow(row)
    return buf.getvalue()
