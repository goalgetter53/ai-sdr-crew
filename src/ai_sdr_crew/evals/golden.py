"""Golden set loader and the offline fixture environment (web pages, signals, CRM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ai_sdr_crew.config import PROJECT_ROOT
from ai_sdr_crew.models import Account, RunStatus
from ai_sdr_crew.tools.data import CRM, CrmRecord, Signal, SignalsDB
from ai_sdr_crew.tools.fetch import SafeFetcher
from ai_sdr_crew.tools.web_fixture import FixtureWeb

GOLDEN_PATH = PROJECT_ROOT / "fixtures" / "golden" / "accounts.json"


class SiteContent(BaseModel):
    tagline: str
    about: list[str] = Field(default_factory=list)
    news: list[str] = Field(default_factory=list)


class Expectation(BaseModel):
    status: RunStatus
    angle_kind: str | None = None
    note: str = ""


class GoldenCase(BaseModel):
    account: Account
    site: SiteContent | None = None
    signals: list[Signal] = Field(default_factory=list)
    crm: CrmRecord
    expect: Expectation


def load_golden(path: Path | None = None) -> list[GoldenCase]:
    raw = json.loads((path or GOLDEN_PATH).read_text(encoding="utf-8"))
    return [GoldenCase.model_validate(c) for c in raw]


def render_site(company: str, site: SiteContent) -> str:
    about = "".join(f"<p>{escape(p)}</p>" for p in site.about)
    news = "".join(f"<li>{escape(n)}</li>" for n in site.news)
    return (
        f"<!doctype html><html><head><title>{escape(company)}</title>"
        f'<meta name="description" content="{escape(site.tagline)}"></head><body>'
        f"<header><h1>{escape(company)}</h1><p>{escape(site.tagline)}</p></header>"
        f"<main><section>{about}</section><section><h2>News</h2><ul>{news}</ul></section></main>"
        f"<footer><p>Copyright {escape(company)}.</p></footer></body></html>"
    )


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


@dataclass
class FixtureEnv:
    web: FixtureWeb
    signals: SignalsDB
    crm: CRM

    def fetcher(self) -> SafeFetcher:
        return SafeFetcher(resolver=self.web.resolver, transport=self.web.transport())


def build_env(cases: list[GoldenCase], extra_pages: dict[str, str] | None = None) -> FixtureEnv:
    pages: dict[str, str] = {}
    for c in cases:
        if c.site is not None:
            pages[host_of(c.account.website)] = render_site(c.account.company, c.site)
    pages.update(extra_pages or {})
    return FixtureEnv(
        web=FixtureWeb(pages),
        signals=SignalsDB({c.account.domain: c.signals for c in cases}),
        crm=CRM({c.account.domain: c.crm for c in cases}),
    )
