"""Ledger: everything a climb spent and decided, in the order it happened.

The ledger is the run's memory and its receipt. It counts model calls and
dollars against the budget, keeps the content hash of every template already
scored so the same prompt is never paid for twice, and records each candidate's
verdict so the API and UI can show *why* the climb stopped where it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BudgetExhaustedError(RuntimeError):
    """The next suite run would exceed the model-call budget."""


@dataclass(slots=True)
class Scored:
    edit: str
    template: str
    content_hash: str
    pass_rate: float
    results: list[int]
    cost_usd: float
    input_tokens: int


@dataclass(slots=True)
class Ledger:
    max_calls: int
    calls: int = 0  # search calls, the ones the budget governs
    holdout_calls: int = 0
    cost_usd: float = 0.0
    scored: dict[str, Scored] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    duplicates_skipped: int = 0

    def has_scored(self, content_hash: str) -> bool:
        return content_hash in self.scored

    def reserve(self, n_calls: int) -> None:
        """Fail *before* spending if ``n_calls`` more would breach the budget."""
        if self.calls + n_calls > self.max_calls:
            raise BudgetExhaustedError(
                f"{n_calls} more calls would exceed the budget "
                f"({self.calls}/{self.max_calls} used)"
            )

    def book(self, run: dict[str, Any], *, holdout: bool = False) -> None:
        """Count a completed suite run's calls and dollars."""
        if holdout:
            self.holdout_calls += run["n_cases"]
        else:
            self.calls += run["n_cases"]
        self.cost_usd = round(self.cost_usd + run["cost_usd"], 6)

    def charge(self, edit: str, template: str, content_hash: str, run: dict[str, Any]) -> Scored:
        """Book a dev-split suite run and remember its score under the template's hash."""
        self.book(run)
        entry = Scored(
            edit=edit,
            template=template,
            content_hash=content_hash,
            pass_rate=run["pass_rate"],
            results=list(run["results"]),
            cost_usd=run["cost_usd"],
            input_tokens=run["input_tokens"],
        )
        self.scored[content_hash] = entry
        return entry

    def record(self, **event: Any) -> None:
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.calls + self.holdout_calls,
            "search_calls": self.calls,
            "holdout_calls": self.holdout_calls,
            "max_calls": self.max_calls,
            "cost_usd": self.cost_usd,
            "duplicates_skipped": self.duplicates_skipped,
            "events": list(self.events),
        }
