"""Per-run token and cost budget. Checked before every LLM call."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_sdr_crew.models import Usage


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    max_tokens: int
    max_cost_usd: float
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    calls: int = 0

    def check(self) -> None:
        if self.usage.total_tokens >= self.max_tokens:
            raise BudgetExceeded(
                f"token budget exhausted ({self.usage.total_tokens}/{self.max_tokens})"
            )
        if self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget exhausted (${self.cost_usd:.4f}/${self.max_cost_usd:.2f})"
            )

    def add(self, usage: Usage, cost_usd: float) -> None:
        self.usage = self.usage + usage
        self.cost_usd += cost_usd
        self.calls += 1
