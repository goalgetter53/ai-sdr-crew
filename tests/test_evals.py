from __future__ import annotations

import pytest

from ai_sdr_crew.evals.regression import compare, gate_failures, regression_markdown
from ai_sdr_crew.evals.runner import report_markdown, run_eval
from ai_sdr_crew.models import RunStatus


@pytest.fixture(scope="module")
def reports():
    return run_eval("v1"), run_eval("v2")


def test_golden_set_is_large_and_fictional(golden) -> None:
    assert len(golden) >= 20
    for c in golden:
        assert c.account.domain.endswith(".example")
        assert c.account.contact.email.endswith(".example")
        assert c.account.website.startswith("https://www.") and ".example/" in c.account.website
    statuses = {c.expect.status for c in golden}
    assert statuses == {RunStatus.READY, RunStatus.NEEDS_HUMAN, RunStatus.HALTED}


def test_v2_meets_expectations_on_every_case(reports) -> None:
    _, v2 = reports
    assert v2.mode == "mock"
    assert v2.aggregates["success_rate"] == 1.0
    assert v2.aggregates["citation_coverage"] == 1.0
    assert v2.aggregates["final_lint_errors"] == 0


def test_v1_needs_more_revisions_than_v2(reports) -> None:
    v1, v2 = reports
    assert v1.aggregates["first_pass_clean_rate"] < v2.aggregates["first_pass_clean_rate"]
    assert v1.aggregates["mean_revisions"] > v2.aggregates["mean_revisions"]
    assert v1.aggregates["ungrounded_facts_dropped"] > 0


def test_regression_report_flags_worse_metrics(reports) -> None:
    v1, v2 = reports
    deltas = {d.metric: d for d in compare(v1, v2)}
    assert deltas["mean_revisions"].verdict == "better"
    assert deltas["judge_personalisation"].verdict == "WORSE"
    assert gate_failures(list(deltas.values()))
    md = regression_markdown(v1, v2)
    assert "MOCK MODE" in md and "Per-account regressions" in md and "FAIL" in md


def test_eval_is_deterministic_apart_from_latency() -> None:
    a, b = run_eval("v2"), run_eval("v2")
    skip = {"latency_p50_ms"}
    assert {k: v for k, v in a.aggregates.items() if k not in skip} == {
        k: v for k, v in b.aggregates.items() if k not in skip
    }


def test_eval_markdown_is_labelled(reports) -> None:
    assert "MOCK MODE" in report_markdown(reports[1])
