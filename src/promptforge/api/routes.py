"""API routes: /tasks, /leaderboard/{task}, /evaluate, /ab, /health."""

from __future__ import annotations

import functools
import logging
import pickle

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from promptforge.evaluation.harness import bootstrap_ab, regression_check, run_suite
from promptforge.registry.store import list_variants
from promptforge.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


class ABRequest(BaseModel):
    task: str
    template_a: str = Field(min_length=5, max_length=4000)
    template_b: str = Field(min_length=5, max_length=4000)


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
    ab = bootstrap_ab(run_a["results"], run_b["results"], get_config()["eval"]["bootstrap_iters"])
    reg = regression_check(run_b["pass_rate"], run_a["pass_rate"])
    return {
        "a": {"pass_rate": run_a["pass_rate"], "cost_usd": run_a["cost_usd"]},
        "b": {"pass_rate": run_b["pass_rate"], "cost_usd": run_b["cost_usd"]},
        "ab": ab,
        "regression": reg,
    }


@router.get("/variants/{task}")
def variants(task: str) -> dict:
    return list_variants(task)
