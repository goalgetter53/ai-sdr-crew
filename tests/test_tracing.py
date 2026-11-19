from __future__ import annotations

import json

import pytest

from ai_sdr_crew.tracing.tracer import Tracer
from ai_sdr_crew.tracing.viewer import load, render_tree, summarize


def test_spans_nest_and_write_jsonl(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    t = Tracer(path)
    t.start_trace("run-1")
    with t.span("run", kind="run"), t.span("agent:researcher", kind="agent"):
        with t.span("llm:researcher", kind="llm") as s:
            s.set(input_tokens=10, output_tokens=5, cost_usd=0.001)
        with t.span("tool:fetch_url", kind="tool", tool="fetch_url") as s:
            s.set(allowed=False)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    by_name = {r["name"]: r for r in lines}
    assert by_name["llm:researcher"]["parent_id"] == by_name["agent:researcher"]["span_id"]
    assert by_name["agent:researcher"]["parent_id"] == by_name["run"]["span_id"]
    assert all(r["trace_id"] == "run-1" and r["latency_ms"] >= 0 for r in lines)


def test_error_spans_are_marked() -> None:
    t = Tracer()
    with pytest.raises(ValueError, match="bad"), t.span("boom", kind="agent"):
        raise ValueError("bad")
    assert t.records[0]["status"] == "error" and "bad" in t.records[0]["attrs"]["error"]


def test_viewer_tree_and_summary(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    t = Tracer(path)
    for run in ("a", "b"):
        t.start_trace(run)
        with t.span("run", kind="run"), t.span("llm:critic", kind="llm") as s:
            s.set(input_tokens=100, output_tokens=10, cost_usd=0.002)
    records = load(path)
    tree = render_tree(records, "a")
    assert "trace a" in tree and "trace b" not in tree and "llm:critic" in tree
    summary = summarize(records)
    assert "runs=2" in summary and "llm_tokens=220" in summary and "llm_cost=$0.0040" in summary
