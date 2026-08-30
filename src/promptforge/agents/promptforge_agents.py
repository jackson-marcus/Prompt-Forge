"""Domain agents that collaborate to optimize a prompt."""

from __future__ import annotations

from typing import Any

from promptforge.agents.base import BaseAgent


class VariantAgent(BaseAgent):
    name = "VariantAgent"
    role = "propose prompt variants"

    def act(self, task: str, memory: dict[str, Any]) -> dict[str, str]:
        hint = str(memory.get("hint", task)).strip() or task
        return {"role": self.role, "task": task, "hint": hint[:200]}


class EvalAgent(BaseAgent):
    name = "EvalAgent"
    role = "score variants on the test suite"

    def act(self, task: str, memory: dict[str, Any]) -> dict[str, str]:
        hint = str(memory.get("hint", task)).strip() or task
        return {"role": self.role, "task": task, "hint": hint[:200]}


class GateAgent(BaseAgent):
    name = "GateAgent"
    role = "block regressions on win-rate"

    def act(self, task: str, memory: dict[str, Any]) -> dict[str, str]:
        hint = str(memory.get("hint", task)).strip() or task
        return {"role": self.role, "task": task, "hint": hint[:200]}


def build_crew() -> list[BaseAgent]:
    return [
        VariantAgent(),
        EvalAgent(),
        GateAgent(),
    ]
