from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_sdr_crew.guardrails.linter import lint_sequence
from helpers import make_seq


def codes(seq) -> set[str]:
    return {i.code for i in lint_sequence(seq)}


def test_clean_sequence_passes() -> None:
    assert codes(make_seq()) == set()


@pytest.mark.parametrize("word", ["free", "guarantee", "act now", "risk-free", "limited time"])
def test_spam_words(word: str) -> None:
    seq = make_seq(step2=f"Hi Maya,\n\nThis is a {word} offer for you. Open to it?")
    assert "SPAM_WORD" in codes(seq)


def test_spam_exceptions_do_not_fire() -> None:
    seq = make_seq(step2="Hi Maya,\n\nYour urgent care clinics run hands-free intake. Open to it?")
    assert "SPAM_WORD" not in codes(seq)


def test_link_in_first_email_only() -> None:
    link = "See https://tallyhaus.example/demo for more."
    assert "LINK_IN_FIRST_EMAIL" in codes(make_seq(step1=f"Hi Maya,\n\n{link} Open to it?"))
    assert "LINK_IN_FIRST_EMAIL" not in codes(make_seq(step2=f"Hi Maya,\n\n{link} Open to it?"))


def test_markdown_and_bare_domain_links_count() -> None:
    assert "LINK_IN_FIRST_EMAIL" in codes(make_seq(step1="Hi,\n\nRead [this](x) now. Ok?"))
    assert "LINK_IN_FIRST_EMAIL" in codes(make_seq(step1="Hi,\n\nVisit tallyhaus.example today?"))


@pytest.mark.parametrize(
    "token", ["{{first_name}}", "{FirstName}", "[First Name]", "<company>", "%FIRSTNAME%", "TBD"]
)
def test_unfilled_tokens(token: str) -> None:
    assert "UNFILLED_TOKEN" in codes(make_seq(step2=f"Hi {token},\n\nFollowing up. Open to it?"))


@given(st.from_regex(r"[a-z_]{1,12}", fullmatch=True))
def test_any_double_brace_token_is_flagged(name: str) -> None:
    seq = make_seq(note=f"Hi {{{{{name}}}}}, glad to connect with you.")
    assert "UNFILLED_TOKEN" in codes(seq)


def test_unverifiable_claims() -> None:
    seq = make_seq(step2="Hi Maya,\n\nWe are the #1 industry-leading platform. Open to it?")
    assert "UNVERIFIABLE_CLAIM" in codes(seq)


def test_length_limits() -> None:
    long_body = "Hi Maya,\n\n" + "word " * 130 + "ok?"
    c = codes(make_seq(step1=long_body, subject1="x" * 70, note="y" * 320))
    assert {"TOO_LONG", "SUBJECT_TOO_LONG", "LINKEDIN_TOO_LONG"} <= c


def test_missing_opt_out() -> None:
    assert "MISSING_OPT_OUT" in codes(make_seq(step3="Hi Maya,\n\nThanks for reading, talk soon."))


def test_deceptive_subject_and_shouting() -> None:
    c = codes(make_seq(subject1="Re: our call", step2="Hi,\n\nTHIS IS VERY BIG NEWS!! Ok?"))
    assert {"DECEPTIVE_SUBJECT", "SHOUTING", "EXCESSIVE_PUNCTUATION"} <= c
