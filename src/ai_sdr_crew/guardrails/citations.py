"""Citation checker: every claim in the copy must cite a grounded research fact.

A sentence is treated as a claim if it contains a number, names the prospect, or uses an
event/fact verb (raised, launched, hiring, acquired, ...). Claims must carry a [F#] marker that
resolves to a known fact, and every number in the sentence must appear in a cited fact.
"""

from __future__ import annotations

import re

from ai_sdr_crew.models import CitationStats, Fact, Issue, Sequence

CITE_RE = re.compile(r"\[(F\d+)\]")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
FACT_VERBS = re.compile(
    r"\b(raised|raising|launch(?:ed|es)|announc\w*|hiring|hired|open(?:ed|ing)|"
    r"expand(?:ed|ing|s)|acquir\w*|migrat\w*|appoint\w*|partner(?:ed|ing)|grew|growing|"
    r"series [a-f]|ipo|funding|revenue|customers?|valuation)\b",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def render(text: str) -> str:
    """Strip citation markers for export / human display."""
    out = re.sub(r"[ \t]*\[F\d+\]", "", text)
    return re.sub(r"[ \t]+([.,!?;:])", r"\1", out)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def numbers(text: str) -> set[str]:
    out: set[str] = set()
    for raw in NUMBER_RE.findall(text):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def _company_names(company: str) -> list[str]:
    names = [company.lower()]
    first = company.split()[0].lower() if company.split() else ""
    if len(first) >= 5:
        names.append(first)
    return names


def is_claim(sentence: str, company: str) -> bool:
    plain = render(sentence)
    lowered = plain.lower()
    return bool(
        NUMBER_RE.search(plain)
        or any(re.search(rf"\b{re.escape(n)}\b", lowered) for n in _company_names(company))
        or FACT_VERBS.search(plain)
    )


def check_citations(
    seq: Sequence, facts: list[Fact], company: str
) -> tuple[list[Issue], CitationStats]:
    by_id = {f.id: f for f in facts}
    issues: list[Issue] = []
    stats = CitationStats()
    texts = [(f"step{s.step}", s.body) for s in seq.steps] + [("linkedin", seq.linkedin_note)]
    for loc, text in texts:
        for s in sentences(text):
            ids = CITE_RE.findall(s)
            unknown = [i for i in ids if i not in by_id]
            for i in unknown:
                issues.append(
                    Issue(
                        source="citations",
                        code="UNKNOWN_CITATION",
                        message=f"[{i}] does not match any grounded fact",
                        location=loc,
                        evidence=s,
                    )
                )
            if not is_claim(s, company):
                continue
            stats.claims += 1
            if not ids:
                issues.append(
                    Issue(
                        source="citations",
                        code="UNCITED_CLAIM",
                        message="claim has no supporting citation",
                        location=loc,
                        evidence=s,
                    )
                )
                continue
            if unknown:
                continue
            stats.cited += 1
            supported: set[str] = set()
            for i in ids:
                supported |= numbers(by_id[i].text) | numbers(by_id[i].quote)
            missing = numbers(render(s)) - supported
            if missing:
                issues.append(
                    Issue(
                        source="citations",
                        code="NUMBER_UNSUPPORTED",
                        message=f"numbers {sorted(missing)} do not appear in the cited facts",
                        location=loc,
                        evidence=s,
                    )
                )
    return issues, stats
