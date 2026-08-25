"""Evaluate all registered variants per task; log the A/B leaderboard to MLflow.

Usage:
    python -m promptforge.evaluation.run
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import pandas as pd

from promptforge.evaluation.harness import bootstrap_ab, regression_check, run_suite
from promptforge.registry.repository import get_repository
from promptforge.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)


def evaluate() -> dict:
    cfg = get_config()
    cases = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "cases.parquet")
    iters = cfg["eval"]["bootstrap_iters"]

    repo = get_repository()
    report = {}
    metrics = {}
    for task in sorted(cases["task"].unique()):
        task_cases = cases[cases["task"] == task].reset_index(drop=True)
        baseline = repo.baseline(task)
        # Evaluate the head snapshot of each variant: whatever restore/edit history
        # got us here, the leaderboard scores what is current.
        runs = {
            name: run_suite(snapshot.template, task_cases)
            for name, snapshot in repo.heads(task).items()
        }

        base_results = runs[baseline]["results"]
        rows = []
        for name, result in runs.items():
            ab = bootstrap_ab(base_results, result["results"], iters)
            reg = regression_check(result["pass_rate"], runs[baseline]["pass_rate"])
            rows.append(
                {
                    "variant": name,
                    "pass_rate": result["pass_rate"],
                    "cost_usd": result["cost_usd"],
                    "delta_vs_baseline": ab["delta"],
                    "ci_low": ab["ci_low"],
                    "ci_high": ab["ci_high"],
                    "beats_baseline": ab["b_wins_significant"] and name != baseline,
                    "passes_regression_gate": reg["passes_gate"],
                }
            )
            metrics[f"{task}_{name}_pass_rate"] = result["pass_rate"]
        rows.sort(key=lambda r: -r["pass_rate"])
        report[task] = {"baseline": baseline, "leaderboard": rows}

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])
    with mlflow.start_run(run_name="prompt-eval"):
        mlflow.log_params({"n_cases": len(cases), "bootstrap_iters": iters})
        mlflow.log_metrics(metrics)
    logger.info("prompt-eval %s", {k: round(v, 3) for k, v in metrics.items()})

    artifacts = resolve_path(cfg["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "report.pkl", "wb") as f:
        pickle.dump({"report": report, "metrics": metrics}, f)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate()
