"""Runtime configuration. Everything that varies between environments comes from env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("SDR_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens.

    ASSUMPTION: Sonnet-class list prices (input $3, output $15, cache write 1.25x, cache read
    0.1x). Edit these to match the model and contract you actually use.
    """

    input: float = 3.0
    output: float = 15.0
    cache_write: float = 3.75
    cache_read: float = 0.30


@dataclass(frozen=True)
class Seller:
    """The (fictional) company the crew prospects for."""

    name: str = "Tallyhaus"
    domain: str = "tallyhaus.example"
    sender_name: str = "Alex Moreno"


@dataclass(frozen=True)
class Guardrails:
    """Each defence layer can be switched off individually for red-team ablations."""

    sanitize: bool = True
    tool_allowlist: bool = True
    ssrf_guard: bool = True
    output_guard: bool = True
    grounding: bool = True
    # Model-based second injection detector inside the sanitiser (no effect if sanitize is off).
    injection_classifier: bool = True

    @classmethod
    def all_off(cls) -> Guardrails:
        return cls(
            sanitize=False,
            tool_allowlist=False,
            ssrf_guard=False,
            output_guard=False,
            grounding=False,
            injection_classifier=False,
        )


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    prompt_version: str = "v2"
    max_revisions: int = 3
    max_agent_turns: int = 6
    max_tokens_per_run: int = 80_000
    max_cost_usd_per_run: float = 0.75
    live: bool = False
    pricing: Pricing = field(default_factory=Pricing)
    seller: Seller = field(default_factory=Seller)
    guardrails: Guardrails = field(default_factory=Guardrails)

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        base = cls(
            model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
            prompt_version=os.environ.get("SDR_PROMPT_VERSION", "v2"),
            max_revisions=int(os.environ.get("SDR_MAX_REVISIONS", "3")),
            max_tokens_per_run=int(os.environ.get("SDR_MAX_TOKENS_PER_RUN", "80000")),
            max_cost_usd_per_run=float(os.environ.get("SDR_MAX_COST_USD_PER_RUN", "0.75")),
        )
        return replace(base, **overrides)
