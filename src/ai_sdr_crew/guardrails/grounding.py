"""Fact grounding: a research fact survives only if its quote appears verbatim in a tool output.

This is the anti-hallucination gate between the researcher and everything downstream. It also
stops injected text that the sanitiser redacted from being smuggled back in as a 'fact'.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ai_sdr_crew.guardrails.citations import numbers
from ai_sdr_crew.guardrails.injection import REDACTED
from ai_sdr_crew.models import Fact


@dataclass(frozen=True)
class ToolOutputRecord:
    tool: str
    text: str


@dataclass(frozen=True)
class DroppedFact:
    fact: Fact
    reason: str


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    t = re.sub(r"\s+", " ", t)
    return t.strip().strip("\"'“”‘’ .")


def ground_facts(
    facts: list[Fact], outputs: list[ToolOutputRecord]
) -> tuple[list[Fact], list[DroppedFact]]:
    kept: list[Fact] = []
    dropped: list[DroppedFact] = []
    for fact in facts:
        quote = _norm(fact.quote)
        if REDACTED.lower() in quote or len(quote) < 5:
            dropped.append(DroppedFact(fact, "quote is empty or quotes redacted content"))
            continue
        sources = [_norm(o.text) for o in outputs if o.tool == fact.source_tool]
        if not any(quote in src for src in sources):
            dropped.append(DroppedFact(fact, f"quote not found in any {fact.source_tool} output"))
            continue
        extra = numbers(fact.text) - numbers(fact.quote)
        if extra:
            dropped.append(DroppedFact(fact, f"text adds numbers not in quote: {sorted(extra)}"))
            continue
        kept.append(fact)
    return kept, dropped
