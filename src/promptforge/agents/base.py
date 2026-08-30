"""Shared agent contract for optimize a prompt."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    agent: str
    output: Any
    confidence: float
    notes: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str
    role: str

    async def run(self, task: str, memory: dict[str, Any]) -> AgentResult:
        output = self.act(task, memory)
        return AgentResult(agent=self.name, output=output, confidence=self.score(output))

    @abstractmethod
    def act(self, task: str, memory: dict[str, Any]) -> Any:
        raise NotImplementedError

    def score(self, output: Any) -> float:
        if output in (None, "", [], {}):
            return 0.0
        return 0.85
