"""Shared builders for tests."""

from __future__ import annotations

from ai_sdr_crew.models import EmailStep, Fact, Sequence

STEP3 = "Hi Maya,\n\nClosing the loop. If this isn't relevant, reply 'no' and I won't follow up."


def make_seq(
    step1: str = "Hi Maya,\n\nSaw the news: Acme raised $5M [F1]. Worth a short call next week?",
    step2: str = "Hi Maya,\n\nFollowing up on my note. Open to it?",
    step3: str = STEP3,
    note: str = "Hi Maya, saw the news about Acme [F1]. Glad to connect.",
    subject1: str = "Scaling AP after the raise",
) -> Sequence:
    return Sequence(
        steps=[
            EmailStep(step=1, delay_days=0, subject=subject1, body=step1),
            EmailStep(step=2, delay_days=3, subject="One more thought", body=step2),
            EmailStep(step=3, delay_days=5, subject="Closing the loop", body=step3),
        ],
        linkedin_note=note,
    )


def make_fact(
    fid: str = "F1", text: str = "Acme raised $5M in seed funding.", kind: str = "funding"
) -> Fact:
    return Fact(
        id=fid,
        kind=kind,  # type: ignore[arg-type]
        text=text,
        quote=text,
        source_tool="lookup_signals",
        source_ref="https://news.example/acme",
    )
