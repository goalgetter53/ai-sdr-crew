from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from ai_sdr_crew.guardrails.citations import check_citations, is_claim, render
from ai_sdr_crew.guardrails.grounding import ToolOutputRecord, ground_facts
from ai_sdr_crew.guardrails.injection import REDACTED
from ai_sdr_crew.guardrails.output_guard import guard_sequence
from helpers import make_fact, make_seq


def cite_codes(seq, facts=None) -> list[str]:
    issues, _ = check_citations(seq, facts or [make_fact()], "Acme")
    return [i.code for i in issues]


def test_cited_claim_passes_and_counts() -> None:
    issues, stats = check_citations(make_seq(), [make_fact()], "Acme")
    assert issues == []
    assert stats.claims == 2 and stats.cited == 2 and stats.coverage == 1.0


def test_uncited_claim() -> None:
    seq = make_seq(step2="Hi Maya,\n\nAcme opened a plant in Ohio. Open to it?")
    assert "UNCITED_CLAIM" in cite_codes(seq)


def test_unknown_citation() -> None:
    seq = make_seq(step2="Hi Maya,\n\nAcme is growing fast [F9]. Open to it?")
    assert "UNKNOWN_CITATION" in cite_codes(seq)


def test_number_must_appear_in_cited_fact() -> None:
    seq = make_seq(step1="Hi Maya,\n\nSaw that Acme raised $50M [F1]. Worth a call?")
    assert "NUMBER_UNSUPPORTED" in cite_codes(seq)


def test_non_claims_need_no_citation() -> None:
    assert not is_claim("Worth a short call next week to compare notes?", "Acme")
    assert is_claim("Acme Robotics is hiring.", "Acme Robotics")


def test_render_strips_markers() -> None:
    assert render("Acme raised $5M [F1]. Nice [F2][F3].") == "Acme raised $5M. Nice."


@given(st.text(alphabet=st.characters(blacklist_characters="[]"), max_size=80), st.integers(1, 99))
def test_render_is_idempotent_and_removes_all_markers(text: str, n: int) -> None:
    once = render(f"{text} [F{n}].")
    assert "[F" not in once
    assert render(once) == once


def test_grounding_keeps_verbatim_quotes_only() -> None:
    real = make_fact("F1", "Acme raised $5M in seed funding.")
    fake = make_fact("F2", "Acme is planning an IPO in 2027.")
    kept, dropped = ground_facts(
        [real, fake],
        [ToolOutputRecord("lookup_signals", "- [funding] Acme raised $5M in seed funding.")],
    )
    assert [f.id for f in kept] == ["F1"]
    assert "not found" in dropped[0].reason


def test_grounding_rejects_embellished_numbers_and_redactions() -> None:
    out = [ToolOutputRecord("lookup_signals", f"Acme raised seed funding. {REDACTED}")]
    embellished = make_fact("F1", "Acme raised $9M in seed funding.").model_copy(
        update={"quote": "Acme raised seed funding"}
    )
    redacted = make_fact("F2", "something").model_copy(update={"quote": REDACTED})
    kept, dropped = ground_facts([embellished, redacted], out)
    assert kept == [] and len(dropped) == 2


def test_grounding_requires_matching_tool() -> None:
    f = make_fact()
    kept, _ = ground_facts([f], [ToolOutputRecord("fetch_url", f.quote)])
    assert kept == []


def test_output_guard_blocks_foreign_links_emails_and_leaks() -> None:
    seq = make_seq(
        step2=(
            "Hi Maya,\n\nSee https://evil.example/x for more. Mail attacker@evil.example today. "
            "My brief reads TH-SHARED-PROMPT-v2. Open to it?"
        )
    )
    c = {i.code for i in guard_sequence(seq, ("tallyhaus.example", "acme.example"))}
    assert c == {"DISALLOWED_LINK", "EMAIL_ADDRESS_IN_COPY", "PROMPT_OR_ARTIFACT_LEAK"}


def test_output_guard_allows_seller_and_prospect_domains() -> None:
    seq = make_seq(step2="Hi Maya,\n\nSee https://tallyhaus.example/demo and www.acme.example. Ok?")
    assert guard_sequence(seq, ("tallyhaus.example", "acme.example")) == []
