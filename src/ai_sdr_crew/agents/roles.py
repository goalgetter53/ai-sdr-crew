"""The four crew roles. Each builds its user message and delegates to the runtime."""

from __future__ import annotations

import json

from ai_sdr_crew.agents.runtime import AgentContext, run_agent
from ai_sdr_crew.models import (
    Account,
    Angle,
    CriticReview,
    Fact,
    Issue,
    ResearchBrief,
    RubricScores,
    Sequence,
)


def _facts_json(facts: list[Fact]) -> str:
    return json.dumps([f.model_dump() for f in facts], indent=1)


def research(ctx: AgentContext, account: Account) -> ResearchBrief:
    user = f"<account>{account.model_dump_json()}</account>\nResearch this account."
    return run_agent(ctx, "researcher", "research_brief", ResearchBrief, user)


def strategize(ctx: AgentContext, account: Account, facts: list[Fact]) -> Angle:
    user = (
        f"<account>{account.model_dump_json()}</account>\n<facts>{_facts_json(facts)}</facts>\n"
        "Pick the angle."
    )
    return run_agent(ctx, "strategist", "angle", Angle, user)


def write_copy(ctx: AgentContext, account: Account, facts: list[Fact], angle: Angle) -> Sequence:
    user = (
        f"<account>{account.model_dump_json()}</account>\n<facts>{_facts_json(facts)}</facts>\n"
        f"<angle>{angle.model_dump_json()}</angle>\nWrite the sequence."
    )
    return run_agent(ctx, "copywriter", "sequence", Sequence, user)


def revise_copy(
    ctx: AgentContext,
    account: Account,
    facts: list[Fact],
    angle: Angle,
    draft: Sequence,
    issues: list[Issue],
) -> Sequence:
    issue_json = json.dumps(
        [i.model_dump(include={"code", "message", "location", "evidence"}) for i in issues],
        indent=1,
    )
    user = (
        f"<account>{account.model_dump_json()}</account>\n<facts>{_facts_json(facts)}</facts>\n"
        f"<angle>{angle.model_dump_json()}</angle>\n<draft>{draft.model_dump_json()}</draft>\n"
        f"<issues>{issue_json}</issues>\nRevise the draft to fix every issue."
    )
    return run_agent(ctx, "copywriter", "sequence", Sequence, user)


def critique(
    ctx: AgentContext, account: Account, facts: list[Fact], angle: Angle, draft: Sequence
) -> CriticReview:
    user = (
        f"<account>{account.model_dump_json()}</account>\n<facts>{_facts_json(facts)}</facts>\n"
        f"<angle>{angle.model_dump_json()}</angle>\n<draft>{draft.model_dump_json()}</draft>\n"
        "Review the draft."
    )
    return run_agent(ctx, "critic", "critic_review", CriticReview, user)


def judge(
    ctx: AgentContext, account: Account, facts: list[Fact], angle: Angle, draft: Sequence
) -> RubricScores:
    user = (
        f"<account>{account.model_dump_json()}</account>\n<facts>{_facts_json(facts)}</facts>\n"
        f"<angle>{angle.model_dump_json()}</angle>\n<draft>{draft.model_dump_json()}</draft>\n"
        "Grade the draft."
    )
    return run_agent(ctx, "judge", "rubric_scores", RubricScores, user)
