"""Rubric scoring.

`heuristic_rubric` is the deterministic offline scorer used by the mock critic and mock judge. It
checks structure (signal cited in email 1, citations, sentence length, closing question, hype),
not persuasion quality. The live judge (`judge_sequence` with the Anthropic backend) is the one
that measures quality; its numbers are TBD until run with an API key.
"""

from __future__ import annotations

import re

from ai_sdr_crew.agents import roles
from ai_sdr_crew.agents.runtime import AgentContext
from ai_sdr_crew.guardrails.citations import CITE_RE, render, sentences
from ai_sdr_crew.guardrails.linter import SPAM_EXCEPTIONS, SPAM_TRIGGERS
from ai_sdr_crew.models import Account, Angle, Fact, RubricScores, Sequence

_GREETING = ("hi ", "hello ", "dear ", "hey ")


def heuristic_rubric(seq: Sequence, angle: Angle) -> tuple[RubricScores, list[str]]:
    step1 = seq.steps[0].body
    relevance = 5 if f"[{angle.signal_fact_id}]" in step1 else 2
    cited = set(CITE_RE.findall(step1))
    personalisation = 5 if len(cited) >= 2 else 4 if len(cited) == 1 else 2
    body = [s for s in sentences(render(step1)) if not s.lower().startswith(_GREETING)]
    lengths = [len(re.findall(r"\b[\w'’-]+\b", s)) for s in body] or [0]
    avg = sum(lengths) / len(lengths)
    clarity = 5 if avg <= 16 else 4 if avg <= 22 else 3 if avg <= 28 else 2
    cta = 5 if body and body[-1].endswith("?") else 2
    all_text = " ".join(render(t) for _, t in seq.texts()).lower()
    for phrase in SPAM_EXCEPTIONS:
        all_text = all_text.replace(phrase, " ")
    hype = sum(
        1 for w in SPAM_TRIGGERS if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", all_text)
    )
    hype += all_text.count("!")
    tone = 5 if hype == 0 else 3 if hype == 1 else 2
    scores = RubricScores(
        relevance=relevance, personalisation=personalisation, clarity=clarity, cta=cta, tone=tone
    )
    issues: list[str] = []
    if relevance < 3:
        issues.append("CRITIC_RELEVANCE: email 1 does not build on the chosen signal")
    if personalisation < 3:
        issues.append("CRITIC_PERSONALISATION: email 1 cites no prospect fact")
    if clarity < 3:
        issues.append("CRITIC_CLARITY: sentences in email 1 are too long")
    if cta < 3:
        issues.append("CRITIC_CTA: email 1 should end with one low-friction question")
    if tone < 3:
        issues.append("CRITIC_TONE: hype or pressure language")
    return scores, issues


def judge_sequence(
    ctx: AgentContext, account: Account, facts: list[Fact], angle: Angle, seq: Sequence
) -> RubricScores:
    return roles.judge(ctx, account, facts, angle, seq)
