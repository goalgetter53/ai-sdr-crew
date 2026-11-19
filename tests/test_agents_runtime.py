from __future__ import annotations

from collections.abc import Callable

import pytest

from ai_sdr_crew.agents.runtime import AgentContext, AgentError, run_agent, system_blocks
from ai_sdr_crew.budget import Budget, BudgetExceeded
from ai_sdr_crew.config import Guardrails, Settings
from ai_sdr_crew.llm.mock import MockBackend
from ai_sdr_crew.llm.types import LLMRequest, LLMResponse, TextBlock, ToolUseBlock
from ai_sdr_crew.models import Angle, Usage
from ai_sdr_crew.prompts import load_prompts
from ai_sdr_crew.tools.registry import ROLE_ALLOWLIST, TOOL_SPECS, Toolbox
from ai_sdr_crew.tracing.tracer import Tracer

Block = TextBlock | ToolUseBlock
GOOD_ANGLE = {
    "signal_fact_id": "F1",
    "kind": "funding",
    "hypothesis": "A new round means more invoices.",
    "value_prop": "Scale AP without headcount.",
}


class ScriptedBackend:
    name = "scripted"

    def __init__(
        self, script: list[Callable[[LLMRequest], list[TextBlock | ToolUseBlock]]]
    ) -> None:
        self.script = script
        self.requests: list[LLMRequest] = []

    def create(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        content = self.script[min(len(self.requests) - 1, len(self.script) - 1)](req)
        return LLMResponse(content=content, usage=Usage(input_tokens=100, output_tokens=20))


def ctx_for(backend, env, guardrails: Guardrails | None = None, **kw) -> AgentContext:
    settings = Settings(guardrails=guardrails or Guardrails(), **kw)
    tracer = Tracer()
    return AgentContext(
        settings=settings,
        backend=backend,
        prompts=load_prompts("v2"),
        tracer=tracer,
        budget=Budget(settings.max_tokens_per_run, settings.max_cost_usd_per_run),
        toolbox=Toolbox(env.fetcher(), env.signals, env.crm, settings.guardrails, tracer),
    )


def submit(payload: dict) -> Callable[[LLMRequest], list[TextBlock | ToolUseBlock]]:
    return lambda req: [ToolUseBlock(id="t1", name="submit_angle", input=payload)]


def test_schema_validation_error_is_fed_back_and_retried(env) -> None:
    backend = ScriptedBackend([submit({"signal_fact_id": "nope"}), submit(GOOD_ANGLE)])
    angle = run_agent(ctx_for(backend, env), "strategist", "angle", Angle, "pick")
    assert angle.kind == "funding"
    feedback = backend.requests[1].messages[-1].content[0]
    assert feedback.is_error and "Schema validation failed" in feedback.content


def test_text_only_reply_gets_nudged(env) -> None:
    backend = ScriptedBackend([lambda r: [TextBlock(text="thinking")], submit(GOOD_ANGLE)])
    assert run_agent(ctx_for(backend, env), "strategist", "angle", Angle, "pick").kind == "funding"
    assert "submit_angle" in backend.requests[1].messages[-1].content[0].text


def test_turn_limit_raises(env) -> None:
    backend = ScriptedBackend([submit({"bad": True})])
    with pytest.raises(AgentError):
        run_agent(ctx_for(backend, env, max_agent_turns=3), "strategist", "angle", Angle, "x")
    assert len(backend.requests) == 3


def test_disallowed_tool_is_blocked_and_recorded(env) -> None:
    hijack = lambda r: [
        ToolUseBlock(
            id="t0", name="crm_log_activity", input={"domain": "a.example", "activity": "x"}
        )
    ]
    backend = ScriptedBackend([hijack, submit(GOOD_ANGLE)])
    ctx = ctx_for(backend, env)
    run_agent(ctx, "strategist", "angle", Angle, "x")
    assert ctx.toolbox.blocked == ["crm_log_activity"] and ctx.toolbox.executed == []
    assert env.crm.activity_log == []
    result = backend.requests[1].messages[-1].content[0]
    assert result.is_error and "not permitted" in result.content


def test_unknown_tool_is_blocked(env) -> None:
    ghost = lambda r: [ToolUseBlock(id="t0", name="send_email", input={})]
    backend = ScriptedBackend([ghost, submit(GOOD_ANGLE)])
    ctx = ctx_for(backend, env, guardrails=Guardrails.all_off())
    run_agent(ctx, "strategist", "angle", Angle, "x")
    assert ctx.toolbox.blocked == ["send_email"]


def test_tool_specs_follow_allowlist(env) -> None:
    ctx = ctx_for(MockBackend(), env)
    assert {t.name for t in ctx.toolbox.specs_for("researcher")} == ROLE_ALLOWLIST["researcher"]
    assert ctx.toolbox.specs_for("copywriter") == []
    assert "crm_log_activity" in TOOL_SPECS
    assert not any("crm_log_activity" in tools for tools in ROLE_ALLOWLIST.values())


def test_non_tool_roles_are_forced_to_submit(env) -> None:
    backend = ScriptedBackend([submit(GOOD_ANGLE)])
    run_agent(ctx_for(backend, env), "strategist", "angle", Angle, "x")
    req = backend.requests[0]
    assert req.tool_choice == {"type": "tool", "name": "submit_angle"}
    assert [t.name for t in req.tools] == ["submit_angle"]


def test_shared_prompt_is_the_cache_breakpoint() -> None:
    blocks = system_blocks(load_prompts("v2"), "critic")
    assert blocks[0].cache and not blocks[1].cache
    assert "TH-SHARED-PROMPT-v2" in blocks[0].text and "ROLE: Critic" in blocks[1].text


def test_budget_checked_before_calls(env) -> None:
    ctx = ctx_for(ScriptedBackend([submit(GOOD_ANGLE)]), env)
    ctx.budget.add(Usage(input_tokens=10**6), 0.0)
    with pytest.raises(BudgetExceeded):
        run_agent(ctx, "strategist", "angle", Angle, "x")


def test_mock_cache_accounting_follows_minimum_prefix(env) -> None:
    backend = MockBackend()
    prompts = load_prompts("v2")
    req = LLMRequest(
        role="judge",
        model="m",
        system=system_blocks(prompts, "judge"),
        messages=[],
        tools=[],
    )
    # judge with no tools: the shared prompt alone is above 1024 estimated tokens
    first = backend._usage(req, [])
    second = backend._usage(req, [])
    assert first.cache_creation_input_tokens > 0 and second.cache_read_input_tokens > 0
    tiny = req.model_copy(update={"system": [req.system[1].model_copy(update={"cache": True})]})
    assert backend._usage(tiny, []).cache_creation_input_tokens == 0
