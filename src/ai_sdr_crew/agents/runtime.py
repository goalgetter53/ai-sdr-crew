"""Minimal agent runtime: tool-use loop + pydantic-validated structured output.

Structured output is obtained by giving each agent a `submit_<name>` tool whose input_schema is
the pydantic model's JSON schema. The arguments are validated with pydantic; on failure the
validation errors are returned to the model as an is_error tool_result so it can self-correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ai_sdr_crew.budget import Budget
from ai_sdr_crew.config import Settings
from ai_sdr_crew.llm.types import (
    ContentBlock,
    LLMBackend,
    LLMRequest,
    Message,
    SystemBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from ai_sdr_crew.prompts import PromptSet
from ai_sdr_crew.tools.registry import Toolbox
from ai_sdr_crew.tracing.tracer import Tracer

T = TypeVar("T", bound=BaseModel)


class AgentError(Exception):
    pass


@dataclass
class AgentContext:
    settings: Settings
    backend: LLMBackend
    prompts: PromptSet
    tracer: Tracer
    budget: Budget
    toolbox: Toolbox


def submit_tool(name: str, model: type[BaseModel]) -> ToolSpec:
    return ToolSpec(
        name=f"submit_{name}",
        description=f"Return the final {name.replace('_', ' ')}. Call exactly once when done.",
        input_schema=model.model_json_schema(),
    )


def system_blocks(prompts: PromptSet, role: str) -> list[SystemBlock]:
    """Shared prompt first with a cache breakpoint; the small role prompt after it."""
    return [SystemBlock(text=prompts.shared, cache=True), SystemBlock(text=prompts.role(role))]


def run_agent(
    ctx: AgentContext, role: str, output_name: str, output_model: type[T], user_text: str
) -> T:
    submit = submit_tool(output_name, output_model)
    role_tools = ctx.toolbox.specs_for(role)
    tools = [*role_tools, submit]
    tool_choice = {"type": "any"} if role_tools else {"type": "tool", "name": submit.name}
    messages: list[Message] = [Message(role="user", content=[TextBlock(text=user_text)])]
    pricing = ctx.settings.pricing

    with ctx.tracer.span(f"agent:{role}", kind="agent", role=role) as agent_span:
        validation_errors = 0
        for turn in range(ctx.settings.max_agent_turns):
            ctx.budget.check()
            req = LLMRequest(
                role=role,
                model=ctx.settings.model,
                system=system_blocks(ctx.prompts, role),
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            with ctx.tracer.span(f"llm:{role}", kind="llm", model=req.model, turn=turn) as span:
                resp = ctx.backend.create(req)
                cost = resp.usage.cost_usd(pricing)
                span.set(
                    **resp.usage.model_dump(), cost_usd=round(cost, 6), backend=ctx.backend.name
                )
            ctx.budget.add(resp.usage, cost)
            messages.append(Message(role="assistant", content=list(resp.content)))

            tool_uses = [b for b in resp.content if isinstance(b, ToolUseBlock)]
            if not tool_uses:
                messages.append(
                    Message(
                        role="user",
                        content=[TextBlock(text=f"Return your result by calling {submit.name}.")],
                    )
                )
                continue

            results: list[ContentBlock] = []
            for tu in tool_uses:
                if tu.name == submit.name:
                    try:
                        out = output_model.model_validate(tu.input)
                    except ValidationError as e:
                        validation_errors += 1
                        detail = json.dumps(e.errors(include_url=False)[:6], default=str)
                        results.append(
                            ToolResultBlock(
                                tool_use_id=tu.id,
                                content=f"Schema validation failed: {detail}. Fix and resubmit.",
                                is_error=True,
                            )
                        )
                        continue
                    agent_span.set(turns=turn + 1, validation_errors=validation_errors)
                    return out
                tool_out = ctx.toolbox.execute(role, tu.name, tu.input)
                results.append(
                    ToolResultBlock(
                        tool_use_id=tu.id, content=tool_out.text, is_error=tool_out.is_error
                    )
                )
            messages.append(Message(role="user", content=results))
        agent_span.set(validation_errors=validation_errors)
        raise AgentError(f"{role} produced no valid output in {ctx.settings.max_agent_turns} turns")
