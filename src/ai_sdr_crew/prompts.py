"""Versioned prompt sets live in prompts/<version>.toml so they can be diffed and evaluated."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ai_sdr_crew.config import PROJECT_ROOT

ROLES = ("researcher", "strategist", "copywriter", "critic", "judge")


@dataclass(frozen=True)
class PromptSet:
    version: str
    shared: str
    roles: dict[str, str]

    def role(self, name: str) -> str:
        return self.roles[name]


def load_prompts(version: str, root: Path | None = None) -> PromptSet:
    path = (root or PROJECT_ROOT / "prompts") / f"{version}.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    missing = [r for r in ROLES if r not in data]
    if missing or "shared" not in data:
        raise ValueError(f"prompt set {version} is missing sections: {missing or ['shared']}")
    return PromptSet(
        version=version,
        shared=str(data["shared"]).strip(),
        roles={r: str(data[r]).strip() for r in ROLES},
    )
