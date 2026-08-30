"""Pattern #3 — User → Orchestrator → parallel agents → Memory → Answer."""

import asyncio

from promptforge.agents.promptforge_agents import build_crew
from promptforge.orchestrator.executor import Executor
from promptforge.orchestrator.memory import Memory
from promptforge.orchestrator.planner import Planner


def test_planner_assigns_every_agent():
    crew = build_crew()
    plan = Planner().plan("optimize a prompt", crew)
    assert set(plan) == {agent.name for agent in crew}
    assert len(crew) == 3


def test_memory_roundtrip():
    memory = Memory()
    memory.write("hint", "keep evidence")
    assert memory.read("hint") == "keep evidence"
    assert "hint" in memory.as_dict()


def test_executor_fans_out_and_merges():
    payload = asyncio.run(Executor().solve("optimize a prompt"))
    assert payload["n_agents"] == 3
    assert payload["min_confidence"] > 0
    assert payload["goal"] == "optimize a prompt"
