"""Measure the optimiser's acceptance rules against the simulator's planted signal.

The simulator rewards exactly three prompt features (format spec, constraint,
few-shot) and is indifferent to the persona line, so for any template we know
the *true* expected pass rate. That lets us score each acceptance rule on
what actually matters: how often it accepts an edit that did nothing, how
much real gain it leaves on the table, and how far the dev-set score it
reports overshoots the held-out one.

Every run climbs from each task's `bare` variant on a fresh dev/test split.
"rows" resamples cases in the bootstrap CI; "inputs" resamples distinct
inputs (the default in the service). The seeded suites repeat each input
about four times, which is what makes that distinction bite.

Usage:
    uv run python scripts/bench_optimizer.py [--seeds 20] [--n-cases 60] [--json out.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from make_tasks import TASKS

from promptforge.llm.simulate import SimulatedModel, prompt_features
from promptforge.optimizer.climb import HillClimb

REAL_EDITS = ("format_spec", "constraint", "few_shot")


def expected_pass(template: str, model: SimulatedModel) -> float:
    """True exact-match pass rate under the simulator: zero without a format spec."""
    if not prompt_features(template)["has_format_spec"]:
        return 0.0
    return model.p_correct(template)


def one_run(task: str, cases: pd.DataFrame, policy: str, seed: int, cluster: bool) -> dict:
    model = SimulatedModel()
    template = TASKS[task]["variants"]["bare"]
    run = HillClimb(policy=policy, seed=seed, cluster_by_input=cluster, model=model).optimize(
        template, cases
    )
    inert = sum(1 for s in run["steps"] if s["edit"] not in REAL_EDITS)
    hold = run["holdout"]
    return {
        "task": task,
        "policy": policy,
        "ci_unit": "inputs" if cluster else "rows",
        "seed": seed,
        "steps": len(run["steps"]),
        "inert_accepts": inert,
        "real_missed": sum(
            1
            for e in REAL_EDITS
            if e not in {s["edit"] for s in run["steps"]}
        ),
        "true_final": expected_pass(run["final_template"], model),
        "dev_final": hold["final"]["dev_pass_rate"],
        "test_final": hold["final"]["test_pass_rate"],
        "optimism": hold["optimism"],
        "holdout_significant": hold["ab"]["b_wins_significant"],
        "search_calls": run["ledger"]["search_calls"],
        "search_cost_usd": run["ledger"]["cost_usd"],
    }


def main() -> None:
    import argparse
    import json

    from promptforge.settings import get_config, resolve_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--policies", nargs="+", default=["greedy", "safe", "gated"])
    parser.add_argument("--cluster", action="store_true", help="resample inputs, not rows")
    parser.add_argument("--json", action="store_true", help="dump per-run records")
    args = parser.parse_args()

    cfg = get_config()
    cases = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "cases.parquet")

    records = []
    for task in TASKS:
        task_cases = cases[cases["task"] == task].reset_index(drop=True)
        for policy in args.policies:
            for seed in args.seeds:
                records.append(one_run(task, task_cases, policy, seed, args.cluster))

    if args.json:
        print(json.dumps(records, indent=1, default=float))
        return

    frame = pd.DataFrame(records)
    unit = "inputs" if args.cluster else "rows"
    print(f"{len(args.seeds)} seeds x {len(TASKS)} tasks, bootstrap CI over {unit}\n")

    header = (
        f"{'policy':>8} {'inert':>7} {'missed':>7} {'true':>7} "
        f"{'dev':>7} {'test':>7} {'optimism':>9} {'calls':>7} {'usd':>7}"
    )
    print(header)
    print("-" * len(header))
    for policy in args.policies:
        rows = frame[frame["policy"] == policy]
        print(
            f"{policy:>8} {rows['inert_accepts'].mean():>7.2f} {rows['real_missed'].mean():>7.2f} "
            f"{rows['true_final'].mean():>7.3f} {rows['dev_final'].mean():>7.3f} "
            f"{rows['test_final'].mean():>7.3f} {rows['optimism'].mean():>9.3f} "
            f"{rows['search_calls'].mean():>7.0f} {rows['search_cost_usd'].mean():>7.3f}"
        )

    print(
        "\ninert  = edits accepted that the simulator is indifferent to"
        "\nmissed = real edits (format_spec, constraint, few_shot) never accepted"
        "\ntrue   = true expected pass rate of the final template"
        "\noptimism = dev score minus held-out score, i.e. how far the report overshoots"
    )


if __name__ == "__main__":
    main()
