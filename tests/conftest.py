from __future__ import annotations

from collections.abc import Callable

import pytest

from ai_sdr_crew.config import Settings
from ai_sdr_crew.evals.golden import FixtureEnv, GoldenCase, build_env, load_golden
from ai_sdr_crew.factory import make_pipeline
from ai_sdr_crew.llm.mock import MockBackend
from ai_sdr_crew.orchestrator import Pipeline
from ai_sdr_crew.tracing.tracer import Tracer


@pytest.fixture(scope="session")
def golden() -> list[GoldenCase]:
    return load_golden()


@pytest.fixture
def env(golden: list[GoldenCase]) -> FixtureEnv:
    return build_env(golden)


@pytest.fixture
def case_by_domain(golden: list[GoldenCase]) -> Callable[[str], GoldenCase]:
    def get(domain: str) -> GoldenCase:
        return next(c for c in golden if c.account.domain == domain)

    return get


@pytest.fixture
def pipeline_factory(env: FixtureEnv) -> Callable[..., Pipeline]:
    def make(
        tracer: Tracer | None = None, backend: object | None = None, **overrides: object
    ) -> Pipeline:
        settings = Settings(**overrides)  # type: ignore[arg-type]
        return make_pipeline(settings, env, tracer=tracer, backend=backend or MockBackend())  # type: ignore[arg-type]

    return make
