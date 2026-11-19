"""Regression report: compare two EvalReports (e.g. prompt v1 vs v2) per metric and per case."""

from __future__ import annotations

from dataclasses import dataclass

from ai_sdr_crew.evals.runner import EvalReport

# +1: higher is better, -1: lower is better
DIRECTION: dict[str, int] = {
    "success_rate": 1,
    "status_accuracy": 1,
    "angle_accuracy": 1,
    "first_pass_clean_rate": 1,
    "mean_revisions": -1,
    "citation_coverage": 1,
    "final_lint_errors": -1,
    "ungrounded_facts_dropped": -1,
    "total_tokens": -1,
    "cost_usd_est": -1,
    "cost_per_ready_sequence_usd_est": -1,
    "cache_savings_pct": 1,
    "judge_relevance": 1,
    "judge_personalisation": 1,
    "judge_clarity": 1,
    "judge_cta": 1,
    "judge_tone": 1,
    "judge_mean": 1,
}
TOLERANCE = 1e-9
GATED = ("success_rate", "status_accuracy", "angle_accuracy", "final_lint_errors", "judge_mean")


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def verdict(self) -> str:
        d = DIRECTION.get(self.metric, 0) * self.delta
        if abs(self.delta) <= TOLERANCE or d == 0:
            return "same"
        return "better" if d > 0 else "WORSE"


def compare(baseline: EvalReport, candidate: EvalReport) -> list[MetricDelta]:
    keys = [k for k in baseline.aggregates if k in candidate.aggregates]
    return [MetricDelta(k, baseline.aggregates[k], candidate.aggregates[k]) for k in keys]


def case_regressions(baseline: EvalReport, candidate: EvalReport) -> list[str]:
    base = {c.domain: c for c in baseline.cases}
    out: list[str] = []
    for c in candidate.cases:
        b = base.get(c.domain)
        if b is None:
            continue
        if b.success and not c.success:
            out.append(f"{c.domain}: success -> failure ({c.status})")
        if b.judge_mean is not None and c.judge_mean is not None and c.judge_mean < b.judge_mean:
            worse = [
                f"{k} {b.judge[k]}->{v}"
                for k, v in (c.judge or {}).items()
                if b.judge and v < b.judge[k]
            ]
            out.append(f"{c.domain}: judge {b.judge_mean} -> {c.judge_mean} ({', '.join(worse)})")
    return out


def gate_failures(deltas: list[MetricDelta]) -> list[str]:
    return [
        f"{d.metric}: {d.baseline} -> {d.candidate}"
        for d in deltas
        if d.metric in GATED and d.verdict == "WORSE"
    ]


def regression_markdown(baseline: EvalReport, candidate: EvalReport) -> str:
    deltas = compare(baseline, candidate)
    mode = baseline.mode if baseline.mode == candidate.mode else f"{baseline.mode}/{candidate.mode}"
    lines = [
        f"# Regression report: {baseline.prompt_version} -> {candidate.prompt_version} "
        f"[{mode.upper()} MODE]",
        "",
        f"Golden set: {candidate.n_cases} fictional accounts. "
        + (
            "Numbers come from deterministic offline runs of the mock model; they measure the "
            "harness, guardrails and prompt-rule adherence of the mock, not real model quality. "
            "Token and cost figures are estimates (~4 chars/token, assumed Sonnet-class prices)."
            if mode == "mock"
            else "Numbers come from live model calls."
        ),
        "",
        f"| metric | {baseline.prompt_version} | {candidate.prompt_version} | delta | verdict |",
        "|---|---|---|---|---|",
    ]
    for d in deltas:
        lines.append(
            f"| {d.metric} | {d.baseline:g} | {d.candidate:g} | {d.delta:+.4g} | {d.verdict} |"
        )
    regs = case_regressions(baseline, candidate)
    lines += ["", "## Per-account regressions", ""]
    lines += [f"- {r}" for r in regs] or ["- none"]
    gates = gate_failures(deltas)
    lines += ["", "## Release gate", ""]
    lines += (
        [f"- FAIL {g}" for g in gates]
        if gates
        else ["- PASS (no gated metric got worse: " + ", ".join(GATED) + ")"]
    )
    return "\n".join(lines) + "\n"
