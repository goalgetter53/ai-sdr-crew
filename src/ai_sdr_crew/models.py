"""Domain models. Every agent output is validated against one of these pydantic schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_sdr_crew.config import Pricing

FactKind = Literal[
    "funding", "expansion", "tech_change", "hiring", "leadership", "launch", "website", "crm"
]
SIGNAL_KINDS: frozenset[str] = frozenset(
    {"funding", "expansion", "tech_change", "hiring", "leadership", "launch"}
)
SourceTool = Literal["lookup_signals", "crm_lookup", "fetch_url"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def cost_usd(self, p: Pricing) -> float:
        return (
            self.input_tokens * p.input
            + self.output_tokens * p.output
            + self.cache_creation_input_tokens * p.cache_write
            + self.cache_read_input_tokens * p.cache_read
        ) / 1_000_000

    def uncached_cost_usd(self, p: Pricing) -> float:
        """What the same traffic would cost with prompt caching disabled."""
        prompt = self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
        return (prompt * p.input + self.output_tokens * p.output) / 1_000_000


class Contact(BaseModel):
    first_name: str | None = None
    last_name: str
    title: str
    email: str


class Account(BaseModel):
    domain: str
    company: str
    website: str
    industry: str
    employees: int
    contact: Contact


class Fact(BaseModel):
    id: str = Field(pattern=r"^F\d+$", description="Stable id such as F1. Copy cites it as [F1].")
    kind: FactKind
    text: str = Field(min_length=5, max_length=400, description="The fact in plain words.")
    quote: str = Field(
        min_length=5,
        max_length=400,
        description="Verbatim excerpt from the tool output that supports the fact.",
    )
    source_tool: SourceTool
    source_ref: str = Field(description="URL or record id the quote came from.")


class ResearchBrief(BaseModel):
    facts: list[Fact] = Field(default_factory=list, max_length=12)
    summary: str = Field(default="", max_length=800)

    @model_validator(mode="after")
    def _unique_ids(self) -> ResearchBrief:
        ids = [f.id for f in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("fact ids must be unique")
        return self


class Angle(BaseModel):
    signal_fact_id: str = Field(pattern=r"^F\d+$")
    kind: FactKind
    hypothesis: str = Field(min_length=10, max_length=400)
    value_prop: str = Field(min_length=10, max_length=400)


class EmailStep(BaseModel):
    step: int = Field(ge=1, le=3)
    delay_days: int = Field(ge=0, le=30)
    subject: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=20, max_length=3000)


class Sequence(BaseModel):
    steps: list[EmailStep] = Field(min_length=3, max_length=3)
    linkedin_note: str = Field(min_length=10, max_length=600)

    @model_validator(mode="after")
    def _ordered(self) -> Sequence:
        if [s.step for s in self.steps] != [1, 2, 3]:
            raise ValueError("steps must be numbered 1, 2, 3 in order")
        return self

    def texts(self) -> list[tuple[str, str]]:
        """(location, text) pairs for every piece of copy, subjects included."""
        out: list[tuple[str, str]] = []
        for s in self.steps:
            out.append((f"subject{s.step}", s.subject))
            out.append((f"step{s.step}", s.body))
        out.append(("linkedin", self.linkedin_note))
        return out


class RubricScores(BaseModel):
    relevance: int = Field(ge=1, le=5)
    personalisation: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    cta: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)

    def as_dict(self) -> dict[str, int]:
        return self.model_dump()

    def minimum(self) -> int:
        return min(self.as_dict().values())

    def mean(self) -> float:
        vals = list(self.as_dict().values())
        return sum(vals) / len(vals)


class CriticReview(BaseModel):
    scores: RubricScores
    issues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Each issue starts with a code, e.g. 'CRITIC_CTA: ...'.",
    )
    verdict: Literal["pass", "revise"]


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["linter", "citations", "output_guard", "critic", "grounding"]
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"
    location: str | None = None
    evidence: str | None = None


class RunStatus(StrEnum):
    READY = "ready"
    NEEDS_HUMAN = "needs_human"
    HALTED = "halted"


class CitationStats(BaseModel):
    claims: int = 0
    cited: int = 0

    @property
    def coverage(self) -> float:
        return 1.0 if self.claims == 0 else self.cited / self.claims


class RunResult(BaseModel):
    run_id: str
    account: Account
    prompt_version: str
    status: RunStatus
    reason: str = ""
    stages: list[str] = Field(default_factory=list)
    brief: ResearchBrief | None = None
    facts: list[Fact] = Field(default_factory=list)
    dropped_facts: list[Fact] = Field(default_factory=list)
    angle: Angle | None = None
    sequence: Sequence | None = None
    review: CriticReview | None = None
    issues: list[Issue] = Field(default_factory=list)
    first_pass_issues: list[Issue] = Field(default_factory=list)
    revisions: int = 0
    citation_stats: CitationStats = Field(default_factory=CitationStats)
    injection_findings: list[str] = Field(default_factory=list)
    tools_executed: list[str] = Field(default_factory=list)
    tools_blocked: list[str] = Field(default_factory=list)
    fetched_urls: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    uncached_cost_usd: float = 0.0
    latency_ms: float = 0.0
