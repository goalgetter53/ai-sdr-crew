"""Deterministic spam / compliance linter for outbound sequences.

Error-severity findings block a sequence from reaching READY and block human approval.
"""

from __future__ import annotations

import re

from ai_sdr_crew.models import Issue, Sequence

SPAM_TRIGGERS: tuple[str, ...] = (
    "free", "guarantee", "guaranteed", "act now", "limited time", "risk-free", "risk free",
    "no obligation", "click here", "buy now", "order now", "winner", "urgent", "cash bonus",
    "earn money", "double your", "best price", "exclusive deal", "once in a lifetime",
    "100%", "no cost", "special promotion", "don't miss", "last chance", "instant access",
    "miracle", "cheap", "lowest price", "save big", "act immediately", "apply now",
)  # fmt: skip
# Legitimate phrases that contain a trigger word ("urgent care" clinics are not spam).
SPAM_EXCEPTIONS: tuple[str, ...] = ("urgent care", "hands-free", "duty-free", "free cash flow")
UNVERIFIABLE = re.compile(
    r"(#1\b|\bnumber one\b|\bbest[- ]in[- ]class\b|\bindustry[- ]leading\b|\bworld'?s (best|"
    r"leading|first)\b|\bleading provider\b|\bproven to\b|\b\d+x (faster|more|better)\b|"
    r"\bunmatched\b|\bunbeatable\b|\brevolutionary\b|\bgame[- ]chang\w*)",
    re.IGNORECASE,
)
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\[[^\]]+\]\([^)]+\)|\b[a-z0-9-]+\.(?:com|io|net|org|ai|co|example)"
    r"(?:/\S*)?\b)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(
    r"(\{\{\s*[^}]*\}\}|\{[A-Za-z_][A-Za-z0-9_ ]*\}|\[\s*(first\s*name|last\s*name|name|company"
    r"(\s*name)?|title|role|city)\s*\]|<\s*(first_?name|last_?name|company|name)\s*>|%[A-Z_]{3,}%"
    r")",
    re.IGNORECASE,
)
PLACEHOLDER_CAPS_RE = re.compile(r"\b(XXX+|TBD|INSERT [A-Z ]+|LOREM IPSUM)\b")
OPT_OUT_RE = re.compile(
    r"(reply ['\"‘’“”]?(no|stop|unsubscribe)|unsubscribe|opt[- ]out|won'?t follow up|"
    r"not (be )?follow(ing)? up|stop (emailing|reaching out)|not interested)",
    re.IGNORECASE,
)
FAKE_THREAD_RE = re.compile(r"^\s*(re|fwd?)\s*:", re.IGNORECASE)

LIMITS = {"step1_words": 120, "followup_words": 100, "subject_chars": 60, "linkedin_chars": 300}


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", re.sub(r"\[F\d+\]", "", text)))


def _sentence_with(text: str, needle: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if needle.lower() in sentence.lower():
            return sentence.strip()
    return needle


def lint_sequence(seq: Sequence) -> list[Issue]:
    issues: list[Issue] = []

    def add(code: str, msg: str, loc: str, evidence: str | None = None) -> None:
        issues.append(
            Issue(source="linter", code=code, message=msg, location=loc, evidence=evidence)
        )

    for loc, text in seq.texts():
        lowered = text.lower()
        for phrase in SPAM_EXCEPTIONS:
            lowered = lowered.replace(phrase, " ")
        for word in SPAM_TRIGGERS:
            if re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", lowered):
                add("SPAM_WORD", f"spam trigger word '{word}'", loc, word)
        for m in [*TOKEN_RE.finditer(text), *PLACEHOLDER_CAPS_RE.finditer(text)]:
            add("UNFILLED_TOKEN", f"unfilled personalisation token {m.group(0)!r}", loc, m.group(0))
        for m in UNVERIFIABLE.finditer(text):
            add(
                "UNVERIFIABLE_CLAIM",
                f"unverifiable claim '{m.group(0)}'",
                loc,
                _sentence_with(text, m.group(0)),
            )
        if text.count("!") > 1 or "!!" in text:
            add("EXCESSIVE_PUNCTUATION", "more than one exclamation mark", loc)
        caps = [w for w in re.findall(r"\b[A-Z]{4,}\b", text) if w not in {"HANA", "NASDAQ"}]
        if len(caps) >= 2:
            add("SHOUTING", f"all-caps words: {', '.join(caps[:3])}", loc)

    for step in seq.steps:
        loc = f"step{step.step}"
        if len(step.subject) > LIMITS["subject_chars"]:
            add("SUBJECT_TOO_LONG", f"subject is {len(step.subject)} chars", f"subject{step.step}")
        if FAKE_THREAD_RE.match(step.subject):
            add("DECEPTIVE_SUBJECT", "fake Re:/Fwd: subject line", f"subject{step.step}")
        limit = LIMITS["step1_words"] if step.step == 1 else LIMITS["followup_words"]
        n = words(step.body)
        if n > limit:
            add("TOO_LONG", f"{loc} has {n} words (limit {limit})", loc)
        if step.step == 1:
            for m in URL_RE.finditer(step.body):
                add(
                    "LINK_IN_FIRST_EMAIL",
                    "links in email 1 hurt deliverability",
                    loc,
                    _sentence_with(step.body, m.group(0)),
                )

    if not OPT_OUT_RE.search(seq.steps[-1].body):
        add("MISSING_OPT_OUT", "final email has no opt-out line", "step3")
    if len(seq.linkedin_note) > LIMITS["linkedin_chars"]:
        add("LINKEDIN_TOO_LONG", f"note is {len(seq.linkedin_note)} chars (limit 300)", "linkedin")
    return issues
