"""API routes: /health, /tasks, /leaderboard/{task}, /ab, /optimize, /variants/{task} (+history, diff, restore)."""

from __future__ import annotations

import functools
import logging
import pickle

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from promptforge.evaluation.harness import bootstrap_ab, regression_check, run_suite
from promptforge.optimizer.climb import POLICIES, HillClimb
from promptforge.registry.repository import UnknownPromptError, get_repository
from promptforge.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


class ABRequest(BaseModel):
    task: str
    template_a: str = Field(min_length=5, max_length=4000)
    template_b: str = Field(min_length=5, max_length=4000)


class OptimizeRequest(BaseModel):
    task: str
    name: str = Field(min_length=1, max_length=200, description="Variant whose head to climb from")
    policy: str | None = Field(default=None, description="greedy | safe | gated (default: config)")
    max_rounds: int | None = Field(default=None, ge=1, le=10)
    seed: int = Field(default=0, ge=0, description="Dev/test split seed")
    commit: bool = Field(
        default=True, description="Append each accepted step to the variant's history"
    )
    created_by: str = Field(default="optimizer", min_length=1, max_length=200)


class RestoreRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    version: int = Field(ge=1)
    created_by: str = Field(default="api", min_length=1, max_length=200)


@functools.lru_cache(maxsize=1)
def _report() -> dict:
    path = resolve_path(get_config()["data"]["artifacts_dir"]) / "report.pkl"
    if not path.exists():
        raise FileNotFoundError("Report missing; run make_tasks.py + promptforge.evaluation.run")
    with open(path, "rb") as f:
        return pickle.load(f)


def _cases() -> pd.DataFrame:
    return pd.read_parquet(resolve_path(get_config()["data"]["processed_dir"]) / "cases.parquet")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/tasks")
def tasks() -> dict:
    try:
        report = _report()["report"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "tasks": {
            t: {"baseline": r["baseline"], "n_variants": len(r["leaderboard"])}
            for t, r in report.items()
        }
    }


@router.get("/leaderboard/{task}")
def leaderboard(task: str) -> dict:
    try:
        report = _report()["report"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if task not in report:
        raise HTTPException(status_code=404, detail=f"unknown task {task}")
    return report[task]


@router.post("/ab")
def ab_test(request: ABRequest) -> dict:
    try:
        cases = _cases()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    task_cases = cases[cases["task"] == request.task].reset_index(drop=True)
    if task_cases.empty:
        raise HTTPException(status_code=404, detail=f"unknown task {request.task}")

    run_a = run_suite(request.template_a, task_cases)
    run_b = run_suite(request.template_b, task_cases)
    ab = bootstrap_ab(
        run_a["results"],
        run_b["results"],
        get_config()["eval"]["bootstrap_iters"],
        groups=task_cases["input"].tolist(),
    )
    reg = regression_check(run_b["pass_rate"], run_a["pass_rate"])
    return {
        "a": {"pass_rate": run_a["pass_rate"], "cost_usd": run_a["cost_usd"]},
        "b": {"pass_rate": run_b["pass_rate"], "cost_usd": run_b["cost_usd"]},
        "ab": ab,
        "regression": reg,
    }


@router.get("/variants/{task}")
def variants(task: str) -> dict:
    """Current head template of every registered variant for a task."""
    repo = get_repository()
    return {
        "baseline": repo.baseline(task),
        "variants": {name: snap.template for name, snap in repo.heads(task).items()},
    }


@router.get("/variants/{task}/history")
def variant_history(
    task: str, name: str | None = Query(default=None, description="Limit to one variant")
) -> dict:
    """Full append-only lineage of every snapshot ever saved for this task."""
    repo = get_repository()
    names = repo.names(task) if name is None else [name]
    if name is not None and not repo.history(task, name):
        raise HTTPException(status_code=404, detail=f"unknown variant {task}/{name}")
    return {
        "task": task,
        "baseline": repo.baseline(task),
        "history": {n: [s.to_dict() for s in repo.history(task, n)] for n in names},
    }


@router.get("/variants/{task}/diff")
def variant_diff(
    task: str,
    name: str = Query(description="Variant to diff"),
    a: int = Query(ge=1, description="Base version"),
    b: int | None = Query(default=None, ge=1, description="Target version (default: head)"),
) -> dict:
    """Unified line diff between two snapshots of one variant's template."""
    repo = get_repository()
    try:
        left = repo.get(task, name, a)
        right = repo.get(task, name, b)
    except UnknownPromptError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    return {
        "task": task,
        "name": name,
        "a": left.version,
        "b": right.version,
        "identical": left.has_same_content_as(right),
        "diff": repo.diff(left, right),
    }


@router.post("/variants/{task}/restore")
def variant_restore(task: str, request: RestoreRequest) -> dict:
    """Restore an old version by appending it as a new head. History is kept."""
    repo = get_repository()
    try:
        snapshot = repo.restore(task, request.name, request.version, request.created_by)
    except UnknownPromptError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    return {"task": task, "restored_from": request.version, "head": snapshot.to_dict()}


@router.post("/optimize")
def optimize(request: OptimizeRequest) -> dict:
    """Hill-climb a variant's head prompt over structural edits, gated by the A/B harness.

    Edits are chosen on a dev split of the task's cases and the result is judged
    on the held-out rest. With ``commit`` every accepted step is appended to the
    variant's history as its own version, so the climb is a lineage you can
    diff and roll back like any other change.
    """
    if request.policy is not None and request.policy not in POLICIES:
        raise HTTPException(
            status_code=422, detail=f"unknown policy {request.policy!r}; use {sorted(POLICIES)}"
        )
    try:
        cases = _cases()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    task_cases = cases[cases["task"] == request.task].reset_index(drop=True)
    if task_cases.empty:
        raise HTTPException(status_code=404, detail=f"unknown task {request.task}")

    repo = get_repository()
    try:
        head = repo.get(request.task, request.name)
    except UnknownPromptError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    climb = HillClimb(policy=request.policy, max_rounds=request.max_rounds, seed=request.seed)
    run = climb.optimize(head.template, task_cases)

    versions = []
    if request.commit:
        for step in run["steps"]:
            snapshot = repo.register(
                request.task,
                request.name,
                step["template"],
                created_by=f"{request.created_by}:{step['edit']}",
            )
            versions.append(snapshot.version)
    logger.info(
        "optimize %s/%s policy=%s steps=%s stop=%r",
        request.task,
        request.name,
        run["policy"],
        [s["edit"] for s in run["steps"]],
        run["stop_reason"],
    )
    return {
        "task": request.task,
        "name": request.name,
        "from_version": head.version,
        "committed_versions": versions,
        **run,
    }
