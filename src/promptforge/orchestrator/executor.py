"""Run agents concurrently, persist results, and fold them into one answer."""

from __future__ import annotations

import asyncio

from promptforge.agents.base import AgentResult
from promptforge.agents.promptforge_agents import build_crew
from promptforge.orchestrator.memory import Memory
from promptforge.orchestrator.planner import Planner


class Executor:
    def __init__(self) -> None:
        self.planner = Planner()
        self.memory = Memory()
        self.agents = build_crew()

    async def solve(self, goal: str) -> dict[str, object]:
        self.memory.write("goal", goal)
        tasks = self.planner.plan(goal, self.agents)
        self.memory.write("plan", tasks)
        coroutines = [agent.run(tasks[agent.name], self.memory.as_dict()) for agent in self.agents]
        results: list[AgentResult] = list(await asyncio.gather(*coroutines))
        for result in results:
            self.memory.write(result.agent, result.output)
        answer = " | ".join(f"{item.agent}:{item.confidence:.2f}" for item in results)
        return {
            "goal": goal,
            "answer": answer,
            "n_agents": len(results),
            "memory": self.memory.as_dict(),
            "min_confidence": min(item.confidence for item in results),
        }
