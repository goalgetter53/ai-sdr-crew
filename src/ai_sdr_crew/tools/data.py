"""Fixture-backed signal provider and CRM. Swap these for Clearbit/Apollo/HubSpot adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

CrmStatus = Literal["new", "open_opportunity", "customer", "do_not_contact"]
HALT_STATUSES: frozenset[str] = frozenset({"open_opportunity", "customer", "do_not_contact"})


class Signal(BaseModel):
    kind: Literal["funding", "expansion", "tech_change", "hiring", "leadership", "launch"]
    date: str
    text: str
    url: str


class CrmRecord(BaseModel):
    domain: str
    status: CrmStatus = "new"
    owner: str | None = None
    last_touch: str | None = None
    notes: str = ""


class SignalsDB:
    def __init__(self, data: dict[str, list[Signal]] | None = None) -> None:
        self._data = {k.lower(): v for k, v in (data or {}).items()}

    def lookup(self, domain: str) -> list[Signal]:
        return list(self._data.get(domain.lower(), []))

    def render(self, domain: str) -> str:
        signals = self.lookup(domain)
        if not signals:
            return f"No signals found for {domain}."
        lines = [f"Signals for {domain}:"]
        lines += [f"- [{s.kind}] {s.date} | {s.text} | source: {s.url}" for s in signals]
        return "\n".join(lines)


class CRM:
    def __init__(self, records: dict[str, CrmRecord] | None = None) -> None:
        self._records = {k.lower(): v for k, v in (records or {}).items()}
        self.activity_log: list[dict[str, str]] = []

    def lookup(self, domain: str) -> CrmRecord:
        return self._records.get(domain.lower(), CrmRecord(domain=domain))

    def render(self, domain: str) -> str:
        r = self.lookup(domain)
        return "\n".join(
            [
                f"CRM record for {r.domain}:",
                f"status: {r.status}",
                f"owner: {r.owner or 'unassigned'}",
                f"last_touch: {r.last_touch or 'never'}",
                f"notes: {r.notes or 'none'}",
            ]
        )

    def log_activity(self, domain: str, activity: str) -> str:
        """A WRITE operation. No LLM role is allowed to call it; it exists to prove that."""
        self.activity_log.append({"domain": domain, "activity": activity[:500]})
        return f"logged activity for {domain}"
