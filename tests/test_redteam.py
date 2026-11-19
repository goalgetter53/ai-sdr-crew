from __future__ import annotations

import pytest

from ai_sdr_crew.evals.redteam import load_pages, report_markdown, run_page

PAGES = load_pages()


def test_redteam_set_has_at_least_ten_pages() -> None:
    assert len(PAGES) >= 10
    assert len({p.technique for p in PAGES}) == len(PAGES)


@pytest.mark.parametrize("page", PAGES, ids=[p.id for p in PAGES])
def test_page_is_neutralised_with_all_guardrails_on(page) -> None:
    outcome = run_page(page, "all_on")
    assert outcome.detected, f"{page.id} not flagged by the detector"
    assert not outcome.compromised, outcome.reasons


def test_attacks_succeed_against_undefended_pipeline() -> None:
    """The red-team test only means something if the attacks work when defences are off."""
    pwned = [p.id for p in PAGES if run_page(p, "all_off").compromised]
    assert len(pwned) >= len(PAGES) // 2, pwned


def test_tool_hijack_blocked_by_allowlist_alone() -> None:
    page = next(p for p in PAGES if p.technique == "tool_hijack")
    assert not run_page(page, "no_sanitizer").compromised
    assert run_page(page, "all_off").compromised


def test_ssrf_blocked_when_only_sanitizer_is_off() -> None:
    page = next(p for p in PAGES if p.technique == "ssrf_metadata")
    assert not run_page(page, "no_sanitizer").compromised


def test_report_renders_matrix() -> None:
    outcomes = [run_page(p, c) for p in PAGES[:2] for c in ("all_on", "all_off")]
    md = report_markdown(outcomes)
    assert "| page | technique | all_on | all_off |" in md
