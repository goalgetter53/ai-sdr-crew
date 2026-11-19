"""Explicit state machine that orchestrates the crew.

    START -> GATE -> RESEARCH -> STRATEGY -> COPY -> CRITIQUE -> DONE
                                                   ^         |
                                                   +-REVISE<-+
    Any non-terminal stage may exit to NEEDS_HUMAN (budget, max revisions, no signal) and the
    deterministic CRM gate may exit to HALTED before a single token is spent.

Every exit except HALTED lands in the human approval queue. Nothing is ever sent.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from ai_sdr_crew.agents import roles
from ai_sdr_crew.agents.runtime import AgentContext, AgentError
from ai_sdr_crew.budget import Budget, BudgetExceeded
from ai_sdr_crew.config import Settings
from ai_sdr_crew.guardrails.citations import check_citations
from ai_sdr_crew.guardrails.grounding import ground_facts
from ai_sdr_crew.guardrails.linter import lint_sequence
from ai_sdr_crew.guardrails.output_guard import guard_sequence
from ai_sdr_crew.llm.types import LLMBackend
from ai_sdr_crew.models import (
    SIGNAL_KINDS,
    Account,
    CitationStats,
    Fact,
    Issue,
    RunResult,
    RunStatus,
    Sequence,
)
from ai_sdr_crew.prompts import PromptSet, load_prompts
from ai_sdr_crew.tools.data import CRM, HALT_STATUSES, SignalsDB
from ai_sdr_crew.tools.fetch import SafeFetcher
from ai_sdr_crew.tools.registry import Toolbox
from ai_sdr_crew.tracing.tracer import Tracer


class Stage(StrEnum):
    START = "start"
    GATE = "gate"
    RESEARCH = "research"
    STRATEGY = "strategy"
    COPY = "copy"
    CRITIQUE = "critique"
    REVISE = "revise"
    DONE = "done"
    NEEDS_HUMAN = "needs_human"
    HALTED = "halted"


TERMINAL = frozenset({Stage.DONE, Stage.NEEDS_HUMAN, Stage.HALTED})
TRANSITIONS: dict[Stage, frozenset[Stage]] = {
    Stage.START: frozenset({Stage.GATE}),
    Stage.GATE: frozenset({Stage.RESEARCH, Stage.HALTED}),
    Stage.RESEARCH: frozenset({Stage.STRATEGY, Stage.NEEDS_HUMAN}),
    Stage.STRATEGY: frozenset({Stage.COPY, Stage.NEEDS_HUMAN}),
    Stage.COPY: frozenset({Stage.CRITIQUE, Stage.NEEDS_HUMAN}),
    Stage.CRITIQUE: frozenset({Stage.DONE, Stage.REVISE, Stage.NEEDS_HUMAN}),
    Stage.REVISE: frozenset({Stage.CRITIQUE, Stage.NEEDS_HUMAN}),
    Stage.DONE: frozenset(),
    Stage.NEEDS_HUMAN: frozenset(),
    Stage.HALTED: frozenset(),
}


class InvalidTransition(Exception):
    pass


@dataclass
class StateMachine:
    stage: Stage = Stage.START
    history: list[Stage] = field(default_factory=lambda: [Stage.START])

    def advance(self, to: Stage) -> None:
        if to not in TRANSITIONS[self.stage]:
            raise InvalidTransition(f"{self.stage} -> {to} is not allowed")
        self.stage = to
        self.history.append(to)

    def abort(self) -> None:
        """Escape hatch from any non-terminal stage (budget exhausted, agent failure)."""
        if self.stage in TERMINAL:
            raise InvalidTransition(f"cannot abort from terminal stage {self.stage}")
        self.stage = Stage.NEEDS_HUMAN
        self.history.append(Stage.NEEDS_HUMAN)

    @property
    def terminal(self) -> bool:
        return self.stage in TERMINAL


def deterministic_checks(
    seq: Sequence, facts: list[Fact], account: Account, settings: Settings
) -> tuple[list[Issue], CitationStats]:
    issues = lint_sequence(seq)
    cite_issues, stats = check_citations(seq, facts, account.company)
    issues += cite_issues
    if settings.guardrails.output_guard:
        issues += guard_sequence(seq, (settings.seller.domain, account.domain))
    return issues, stats


def _critic_issue(raw: str) -> Issue:
    code, _, msg = raw.partition(":")
    code = code.strip().upper().replace(" ", "_") if msg else "CRITIC_NOTE"
    return Issue(source="critic", code=code, message=(msg or raw).strip(), location="step1")


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        backend: LLMBackend,
        fetcher: SafeFetcher,
        signals: SignalsDB,
        crm: CRM,
        tracer: Tracer | None = None,
        prompts: PromptSet | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.fetcher = fetcher
        self.signals = signals
        self.crm = crm
        self.tracer = tracer or Tracer()
        self.prompts = prompts or load_prompts(settings.prompt_version)

    def run(self, account: Account) -> RunResult:
        run_id = f"{account.domain}-{uuid.uuid4().hex[:8]}"
        self.tracer.start_trace(run_id)
        t0 = time.perf_counter()
        sm = StateMachine()
        budget = Budget(self.settings.max_tokens_per_run, self.settings.max_cost_usd_per_run)
        toolbox = Toolbox(
            self.fetcher, self.signals, self.crm, self.settings.guardrails, self.tracer
        )
        ctx = AgentContext(self.settings, self.backend, self.prompts, self.tracer, budget, toolbox)
        result = RunResult(
            run_id=run_id,
            account=account,
            prompt_version=self.prompts.version,
            status=RunStatus.NEEDS_HUMAN,
        )

        with self.tracer.span(
            "run", kind="run", account=account.domain, prompt_version=self.prompts.version
        ) as span:
            try:
                self._run(account, sm, ctx, result)
            except (BudgetExceeded, AgentError) as e:
                sm.abort()
                result.status = RunStatus.NEEDS_HUMAN
                result.reason = f"{type(e).__name__}: {e}"
            result.stages = [s.value for s in sm.history]
            result.usage = budget.usage
            result.cost_usd = round(budget.cost_usd, 6)
            result.uncached_cost_usd = round(
                budget.usage.uncached_cost_usd(self.settings.pricing), 6
            )
            result.tools_executed = list(toolbox.executed)
            result.tools_blocked = list(toolbox.blocked)
            result.fetched_urls = list(toolbox.fetched_urls)
            result.injection_findings = list(toolbox.injection_findings)
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            span.set(
                status_out=result.status.value,
                reason=result.reason,
                revisions=result.revisions,
                cost_usd=result.cost_usd,
                total_tokens=budget.usage.total_tokens,
            )
        return result

    def _finish(self, sm: StateMachine, result: RunResult, stage: Stage, reason: str = "") -> None:
        sm.advance(stage)
        result.status = {
            Stage.DONE: RunStatus.READY,
            Stage.HALTED: RunStatus.HALTED,
            Stage.NEEDS_HUMAN: RunStatus.NEEDS_HUMAN,
        }[stage]
        result.reason = reason

    def _run(
        self, account: Account, sm: StateMachine, ctx: AgentContext, result: RunResult
    ) -> None:
        settings = self.settings
        sm.advance(Stage.GATE)
        record = self.crm.lookup(account.domain)
        with self.tracer.span("crm_gate", kind="guardrail", crm_status=record.status):
            if record.status in HALT_STATUSES:
                self._finish(sm, result, Stage.HALTED, f"crm status is {record.status}")
                return

        sm.advance(Stage.RESEARCH)
        brief = roles.research(ctx, account)
        result.brief = brief
        facts = brief.facts
        if settings.guardrails.grounding:
            with self.tracer.span("grounding", kind="guardrail") as gspan:
                facts, dropped = ground_facts(brief.facts, ctx.toolbox.outputs)
                result.dropped_facts = [d.fact for d in dropped]
                gspan.set(
                    kept=len(facts), dropped=len(dropped), reasons=[d.reason for d in dropped]
                )
        result.facts = facts
        signal_ids = {f.id for f in facts if f.kind in SIGNAL_KINDS}
        if not signal_ids:
            self._finish(sm, result, Stage.NEEDS_HUMAN, "no verifiable buying signal")
            return

        sm.advance(Stage.STRATEGY)
        angle = roles.strategize(ctx, account, facts)
        result.angle = angle
        if angle.signal_fact_id not in signal_ids:
            self._finish(sm, result, Stage.NEEDS_HUMAN, "angle is not tied to a verified signal")
            return

        sm.advance(Stage.COPY)
        seq = roles.write_copy(ctx, account, facts, angle)
        first_pass = True
        while True:
            sm.advance(Stage.CRITIQUE)
            with self.tracer.span("deterministic_checks", kind="guardrail") as cspan:
                issues, stats = deterministic_checks(seq, facts, account, settings)
                cspan.set(issues=[i.code for i in issues])
            errors = [i for i in issues if i.severity == "error"]
            if not errors:
                review = roles.critique(ctx, account, facts, angle, seq)
                result.review = review
                if review.verdict == "pass" and review.scores.minimum() >= 3:
                    result.sequence, result.issues, result.citation_stats = seq, issues, stats
                    if first_pass:
                        result.first_pass_issues = []
                    self._finish(sm, result, Stage.DONE)
                    return
                issues = issues + [_critic_issue(i) for i in review.issues]
            if first_pass:
                result.first_pass_issues = issues
                first_pass = False
            if result.revisions >= settings.max_revisions:
                result.sequence, result.issues, result.citation_stats = seq, issues, stats
                self._finish(
                    sm,
                    result,
                    Stage.NEEDS_HUMAN,
                    f"unresolved issues after {result.revisions} revisions",
                )
                return
            sm.advance(Stage.REVISE)
            seq = roles.revise_copy(ctx, account, facts, angle, seq, issues)
            result.revisions += 1
            result.sequence = seq
