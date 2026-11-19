"""Golden-set evaluation: deterministic checks + rubric judge, aggregated into an EvalReport."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ai_sdr_crew.agents.runtime import AgentContext
from ai_sdr_crew.budget import Budget
from ai_sdr_crew.config import Settings
from ai_sdr_crew.evals.golden import GoldenCase, build_env, load_golden
from ai_sdr_crew.evals.judge import judge_sequence
from ai_sdr_crew.factory import make_backend, make_pipeline
from ai_sdr_crew.llm.types import LLMBackend
from ai_sdr_crew.models import RunResult, RunStatus
from ai_sdr_crew.tools.registry import Toolbox
from ai_sdr_crew.tracing.tracer import Tracer


class CaseResult(BaseModel):
    domain: str
    expected_status: str
    status: str
    status_ok: bool
    expected_angle: str | None
    angle_kind: str | None
    angle_ok: bool
    reached_critique: bool
    first_pass_clean: bool
    first_pass_issue_codes: list[str] = Field(default_factory=list)
    revisions: int
    final_error_count: int
    citation_coverage: float
    ungrounded_facts_dropped: int
    judge: dict[str, int] | None = None
    judge_mean: float | None = None
    total_tokens: int
    cache_read_tokens: int
    cost_usd: float
    uncached_cost_usd: float
    latency_ms: float
    success: bool


class EvalReport(BaseModel):
    mode: str
    prompt_version: str
    model: str
    generated_at: str
    n_cases: int
    aggregates: dict[str, float]
    cases: list[CaseResult]


def _case_result(case: GoldenCase, r: RunResult, judge: dict[str, int] | None) -> CaseResult:
    status_ok = r.status.value == case.expect.status.value
    angle_kind = r.angle.kind if r.angle else None
    angle_ok = case.expect.angle_kind is None or angle_kind == case.expect.angle_kind
    final_errors = sum(1 for i in r.issues if i.severity == "error")
    reached = r.sequence is not None
    success = status_ok and angle_ok and (r.status != RunStatus.READY or final_errors == 0)
    return CaseResult(
        domain=case.account.domain,
        expected_status=case.expect.status.value,
        status=r.status.value,
        status_ok=status_ok,
        expected_angle=case.expect.angle_kind,
        angle_kind=angle_kind,
        angle_ok=angle_ok,
        reached_critique=reached,
        first_pass_clean=reached and not r.first_pass_issues,
        first_pass_issue_codes=sorted({i.code for i in r.first_pass_issues}),
        revisions=r.revisions,
        final_error_count=final_errors,
        citation_coverage=round(r.citation_stats.coverage, 4),
        ungrounded_facts_dropped=len(r.dropped_facts),
        judge=judge,
        judge_mean=None if judge is None else round(sum(judge.values()) / len(judge), 3),
        total_tokens=r.usage.total_tokens,
        cache_read_tokens=r.usage.cache_read_input_tokens,
        cost_usd=r.cost_usd,
        uncached_cost_usd=r.uncached_cost_usd,
        latency_ms=r.latency_ms,
        success=success,
    )


def _aggregate(cases: list[CaseResult]) -> dict[str, float]:
    n = len(cases)
    reached = [c for c in cases if c.reached_critique]
    judged = [c for c in cases if c.judge]
    cost = sum(c.cost_usd for c in cases)
    uncached = sum(c.uncached_cost_usd for c in cases)
    agg: dict[str, float] = {
        "success_rate": sum(c.success for c in cases) / n,
        "status_accuracy": sum(c.status_ok for c in cases) / n,
        "angle_accuracy": sum(c.angle_ok for c in cases) / n,
        "first_pass_clean_rate": (
            sum(c.first_pass_clean for c in reached) / len(reached) if reached else 0.0
        ),
        "mean_revisions": statistics.mean(c.revisions for c in reached) if reached else 0.0,
        "citation_coverage": (
            statistics.mean(c.citation_coverage for c in reached) if reached else 0.0
        ),
        "final_lint_errors": float(sum(c.final_error_count for c in cases)),
        "ungrounded_facts_dropped": float(sum(c.ungrounded_facts_dropped for c in cases)),
        "total_tokens": float(sum(c.total_tokens for c in cases)),
        "cost_usd_est": round(cost, 4),
        "cost_per_ready_sequence_usd_est": round(
            cost / max(1, sum(1 for c in cases if c.status == "ready")), 4
        ),
        "cache_savings_pct": round(100 * (1 - cost / uncached), 2) if uncached else 0.0,
        "latency_p50_ms": statistics.median(c.latency_ms for c in cases),
    }
    for dim in ("relevance", "personalisation", "clarity", "cta", "tone"):
        if judged:
            agg[f"judge_{dim}"] = round(statistics.mean(c.judge[dim] for c in judged if c.judge), 3)
    if judged:
        agg["judge_mean"] = round(statistics.mean(c.judge_mean or 0 for c in judged), 3)
    return {k: round(v, 4) for k, v in agg.items()}


def run_eval(
    prompt_version: str,
    live: bool = False,
    trace_path: Path | None = None,
    backend: LLMBackend | None = None,
    cases: list[GoldenCase] | None = None,
) -> EvalReport:
    settings = Settings.from_env(prompt_version=prompt_version, live=live)
    cases = cases or load_golden()
    env = build_env(cases)
    tracer = Tracer(trace_path)
    backend = backend or make_backend(settings)
    pipeline = make_pipeline(settings, env, tracer=tracer, backend=backend)
    results: list[CaseResult] = []
    for case in cases:
        r = pipeline.run(case.account)
        judge: dict[str, int] | None = None
        if r.sequence is not None and r.angle is not None:
            ctx = AgentContext(
                settings,
                backend,
                pipeline.prompts,
                tracer,
                Budget(settings.max_tokens_per_run, settings.max_cost_usd_per_run),
                Toolbox(pipeline.fetcher, env.signals, env.crm, settings.guardrails, tracer),
            )
            judge = judge_sequence(ctx, case.account, r.facts, r.angle, r.sequence).as_dict()
        results.append(_case_result(case, r, judge))
    return EvalReport(
        mode="live" if live else "mock",
        prompt_version=prompt_version,
        model=settings.model if live else f"mock (prompt-sensitive, {settings.model} not called)",
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_cases=len(cases),
        aggregates=_aggregate(results),
        cases=results,
    )


def report_markdown(r: EvalReport) -> str:
    label = "MOCK MODE (deterministic offline model)" if r.mode == "mock" else "LIVE MODE"
    lines = [
        f"# Eval report: prompt {r.prompt_version} [{label}]",
        "",
        f"- model: {r.model}",
        f"- cases: {r.n_cases}",
        f"- generated: {r.generated_at}",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in r.aggregates.items()]
    lines += [
        "",
        "| account | status (exp) | angle (exp) | first-pass issues | revisions | judge mean |",
        "|---|---|---|---|---|---|",
    ]
    for c in r.cases:
        lines.append(
            f"| {c.domain} | {c.status} ({c.expected_status}) | {c.angle_kind} "
            f"({c.expected_angle}) | {', '.join(c.first_pass_issue_codes) or '-'} | "
            f"{c.revisions} | {c.judge_mean if c.judge_mean is not None else '-'} |"
        )
    return "\n".join(lines) + "\n"
