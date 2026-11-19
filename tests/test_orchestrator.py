from __future__ import annotations

from typing import Any

import pytest

from ai_sdr_crew.llm.mock import MockBackend
from ai_sdr_crew.llm.types import LLMRequest, LLMResponse
from ai_sdr_crew.models import RunStatus
from ai_sdr_crew.orchestrator import InvalidTransition, Stage, StateMachine
from ai_sdr_crew.tracing.tracer import Tracer


class CountingBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def create(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req.role)
        return super().create(req)


class StubbornCopywriter(CountingBackend):
    """Always re-submits a draft with a link in email 1, so the revise loop can never finish."""

    def _revise(self, draft: dict[str, Any], issues: list[dict[str, Any]], account: dict[str, Any]):
        out = super()._revise(draft, issues, account)
        out["steps"][0]["body"] += "\n\nSee https://tallyhaus.example/demo."
        return out

    def _compose(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        out = super()._compose(*args, **kwargs)
        out["steps"][0]["body"] += "\n\nSee https://tallyhaus.example/demo."
        return out


def test_state_machine_happy_path_and_invalid_transition() -> None:
    sm = StateMachine()
    for s in (Stage.GATE, Stage.RESEARCH, Stage.STRATEGY, Stage.COPY, Stage.CRITIQUE, Stage.DONE):
        sm.advance(s)
    assert sm.terminal
    with pytest.raises(InvalidTransition):
        sm.advance(Stage.REVISE)
    with pytest.raises(InvalidTransition):
        StateMachine().advance(Stage.COPY)


def test_abort_only_from_non_terminal() -> None:
    sm = StateMachine()
    sm.advance(Stage.GATE)
    sm.abort()
    assert sm.stage == Stage.NEEDS_HUMAN
    with pytest.raises(InvalidTransition):
        sm.abort()


@pytest.mark.parametrize("domain", ["kestrel.example", "lumenfield.example", "amberly.example"])
def test_crm_gate_halts_before_any_llm_call(pipeline_factory, case_by_domain, domain: str) -> None:
    backend = CountingBackend()
    r = pipeline_factory(backend=backend).run(case_by_domain(domain).account)
    assert r.status == RunStatus.HALTED
    assert backend.calls == [] and r.cost_usd == 0


def test_happy_path_ready_with_citations(pipeline_factory, case_by_domain) -> None:
    r = pipeline_factory().run(case_by_domain("brightforge.example").account)
    assert r.status == RunStatus.READY
    assert r.stages[:5] == ["start", "gate", "research", "strategy", "copy"]
    assert r.angle and r.angle.kind == "funding"
    assert r.citation_stats.coverage == 1.0 and not r.issues
    assert r.sequence and "[F" in r.sequence.steps[0].body


def test_no_signal_routes_to_human(pipeline_factory, case_by_domain) -> None:
    r = pipeline_factory().run(case_by_domain("ridgeway.example").account)
    assert r.status == RunStatus.NEEDS_HUMAN and "signal" in r.reason


def test_fetch_failure_is_tolerated(pipeline_factory, case_by_domain) -> None:
    r = pipeline_factory().run(case_by_domain("birchmont.example").account)
    assert r.status == RunStatus.READY and r.fetched_urls == []


def test_v1_prompt_needs_revisions_and_loop_fixes_them(pipeline_factory, case_by_domain) -> None:
    r = pipeline_factory(prompt_version="v1").run(case_by_domain("quarrystone.example").account)
    assert "UNFILLED_TOKEN" in {i.code for i in r.first_pass_issues}
    assert r.status == RunStatus.READY and r.revisions >= 1
    assert r.sequence and "{{" not in r.sequence.model_dump_json()


def test_v1_hallucinated_fact_is_dropped_by_grounding(pipeline_factory, case_by_domain) -> None:
    r = pipeline_factory(prompt_version="v1").run(case_by_domain("brightforge.example").account)
    assert any("IPO" in f.text for f in r.dropped_facts)
    assert all("IPO" not in f.text for f in r.facts)


def test_max_revisions_routes_to_human(pipeline_factory, case_by_domain) -> None:
    backend = StubbornCopywriter()
    r = pipeline_factory(backend=backend, max_revisions=2).run(
        case_by_domain("fernway.example").account
    )
    assert r.status == RunStatus.NEEDS_HUMAN and r.revisions == 2
    assert "LINK_IN_FIRST_EMAIL" in {i.code for i in r.issues}
    assert r.stages.count("revise") == 2


def test_token_budget_routes_to_human(pipeline_factory, case_by_domain) -> None:
    r = pipeline_factory(max_tokens_per_run=3000).run(case_by_domain("fernway.example").account)
    assert r.status == RunStatus.NEEDS_HUMAN and "BudgetExceeded" in r.reason


def test_cost_budget_routes_to_human(pipeline_factory, case_by_domain) -> None:
    r = pipeline_factory(max_cost_usd_per_run=0.001).run(case_by_domain("fernway.example").account)
    assert r.status == RunStatus.NEEDS_HUMAN and "cost budget" in r.reason


def test_critic_only_runs_when_deterministic_checks_pass(pipeline_factory, case_by_domain) -> None:
    backend = CountingBackend()
    r = pipeline_factory(backend=backend, prompt_version="v1").run(
        case_by_domain("copperline.example").account
    )
    assert r.revisions >= 1
    assert backend.calls.count("critic") == 1  # only the final, clean draft is reviewed


def test_run_emits_nested_trace(pipeline_factory, case_by_domain, tmp_path) -> None:
    tracer = Tracer(tmp_path / "t.jsonl")
    r = pipeline_factory(tracer=tracer).run(case_by_domain("xenith.example").account)
    kinds = {rec["kind"] for rec in tracer.records}
    assert {"run", "agent", "llm", "tool", "guardrail"} <= kinds
    run_span = next(rec for rec in tracer.records if rec["kind"] == "run")
    assert run_span["trace_id"] == r.run_id and run_span["parent_id"] is None
    assert run_span["attrs"]["status_out"] == "ready"


def test_prompt_caching_reads_cache_across_accounts(pipeline_factory, golden) -> None:
    p = pipeline_factory()
    first = p.run(golden[0].account)
    second = p.run(golden[1].account)
    assert first.usage.cache_creation_input_tokens > 0
    assert second.usage.cache_read_input_tokens > 0
    assert second.cost_usd < second.uncached_cost_usd
