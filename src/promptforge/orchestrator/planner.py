"""Decompose a user goal (optimize a prompt) into per-agent tasks."""

from __future__ import annotations

from promptforge.agents.base import BaseAgent


class Planner:
    def plan(self, goal: str, agents: list[BaseAgent]) -> dict[str, str]:
        return {agent.name: f"{agent.role}: {goal}" for agent in agents}
