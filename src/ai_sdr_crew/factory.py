"""Wiring: pick the LLM backend and assemble a pipeline over the fixture environment."""

from __future__ import annotations

import os

from ai_sdr_crew.config import Settings
from ai_sdr_crew.evals.golden import FixtureEnv
from ai_sdr_crew.llm.types import LLMBackend
from ai_sdr_crew.orchestrator import Pipeline
from ai_sdr_crew.tracing.tracer import Tracer


def make_backend(settings: Settings) -> LLMBackend:
    if settings.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("--live needs ANTHROPIC_API_KEY (see .env.example)")
        from ai_sdr_crew.llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend()
    from ai_sdr_crew.llm.mock import MockBackend

    return MockBackend()


def make_pipeline(
    settings: Settings,
    env: FixtureEnv,
    tracer: Tracer | None = None,
    backend: LLMBackend | None = None,
) -> Pipeline:
    return Pipeline(
        settings=settings,
        backend=backend or make_backend(settings),
        fetcher=env.fetcher(),
        signals=env.signals,
        crm=env.crm,
        tracer=tracer,
    )
