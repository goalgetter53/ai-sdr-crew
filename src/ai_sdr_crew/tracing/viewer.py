"""Trace viewer: render a span tree per run and an aggregate summary from a JSONL trace file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def load(path: Path | str) -> list[Record]:
    out: list[Record] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def _label(kind: str, name: str) -> str:
    return name if name.startswith(kind + ":") else f"{kind}:{name}"


def _describe(r: Record) -> str:
    a = r.get("attrs", {})
    bits = [f"{r['latency_ms']:.1f}ms"]
    if "input_tokens" in a:
        bits.append(
            f"in={a['input_tokens']} out={a.get('output_tokens', 0)} "
            f"cache_r={a.get('cache_read_input_tokens', 0)} "
            f"cache_w={a.get('cache_creation_input_tokens', 0)}"
        )
    if "cost_usd" in a:
        bits.append(f"${a['cost_usd']:.5f}")
    for key in ("tool", "allowed", "status_out", "findings", "issues", "reason", "error"):
        if key in a and a[key] not in (None, "", []):
            bits.append(f"{key}={a[key]}")
    if r.get("status") == "error":
        bits.append("ERROR")
    return " ".join(bits)


def trace_ids(records: list[Record]) -> list[str]:
    seen: dict[str, None] = {}
    for r in records:
        seen.setdefault(r["trace_id"], None)
    return list(seen)


def _walk(
    children: dict[str | None, list[Record]], parent: str | None, depth: int, lines: list[str]
) -> None:
    for r in children.get(parent, []):
        lines.append(f"{'  ' * (depth + 1)}{_label(r['kind'], r['name'])}  {_describe(r)}")
        _walk(children, r["span_id"], depth + 1, lines)


def render_tree(records: list[Record], trace_id: str | None = None) -> str:
    lines: list[str] = []
    for tid in trace_ids(records):
        if trace_id and tid != trace_id:
            continue
        spans = [r for r in records if r["trace_id"] == tid]
        children: dict[str | None, list[Record]] = defaultdict(list)
        for r in sorted(spans, key=lambda r: r["start"]):
            children[r["parent_id"]].append(r)
        lines.append(f"trace {tid}")
        _walk(children, None, 0, lines)
    return "\n".join(lines)


def summarize(records: list[Record]) -> str:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r["kind"], r["name"])].append(r)
    header = (
        f"{'kind:name':<34}{'count':>6}{'p50 ms':>10}{'p95 ms':>10}{'tokens':>10}{'cost $':>10}"
    )
    lines = [header, "-" * len(header)]
    tot_tokens = 0
    tot_cost = 0.0
    for (kind, name), rs in sorted(groups.items()):
        lat = [r["latency_ms"] for r in rs]
        tokens = sum(
            int(r["attrs"].get(k, 0))
            for r in rs
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        )
        cost = sum(float(r["attrs"].get("cost_usd", 0.0)) for r in rs)
        if kind == "llm":
            tot_tokens += tokens
            tot_cost += cost
        lines.append(
            f"{_label(kind, name):<34}{len(rs):>6}{_pct(lat, 0.5):>10.1f}{_pct(lat, 0.95):>10.1f}"
            f"{tokens:>10}{cost:>10.4f}"
        )
    blocked = sum(1 for r in records if r["kind"] == "tool" and r["attrs"].get("allowed") is False)
    errors = sum(1 for r in records if r["status"] == "error")
    lines.append("-" * len(header))
    lines.append(
        f"runs={len(trace_ids(records))} llm_tokens={tot_tokens} llm_cost=${tot_cost:.4f} "
        f"blocked_tool_calls={blocked} error_spans={errors}"
    )
    return "\n".join(lines)
