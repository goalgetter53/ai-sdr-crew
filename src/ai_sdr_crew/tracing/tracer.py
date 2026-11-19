"""Structured JSONL tracing with nested spans (run -> agent -> llm/tool/guardrail)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


class Tracer:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.trace_id = "untraced"
        self._stack: list[Span] = []

    def start_trace(self, trace_id: str) -> None:
        self.trace_id = trace_id

    @contextmanager
    def span(self, name: str, kind: str, **attrs: Any) -> Iterator[Span]:
        parent = self._stack[-1].span_id if self._stack else None
        sp = Span(self.trace_id, uuid.uuid4().hex[:12], parent, kind, name, dict(attrs))
        self._stack.append(sp)
        started_wall = time.time()
        t0 = time.perf_counter()
        try:
            yield sp
        except BaseException as e:
            sp.status = "error"
            sp.attrs["error"] = f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            self._stack.pop()
            record = {
                "trace_id": sp.trace_id,
                "span_id": sp.span_id,
                "parent_id": sp.parent_id,
                "kind": sp.kind,
                "name": sp.name,
                "start": round(started_wall, 6),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "status": sp.status,
                "attrs": sp.attrs,
            }
            self.records.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
