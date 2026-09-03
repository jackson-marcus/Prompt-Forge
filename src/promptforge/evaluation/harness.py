"""Evaluation harness: run a suite, A/B compare, bootstrap CI, regression gate, cost."""

from __future__ import annotations

import numpy as np
import pandas as pd

from promptforge.evaluation.assertions import score_case
from promptforge.llm.simulate import SimulatedModel
from promptforge.settings import get_config


def run_suite(template: str, cases: pd.DataFrame, model: SimulatedModel | None = None) -> dict:
    model = model or SimulatedModel()
    labels = sorted(cases["true_label"].unique().tolist())
    results, input_tok, output_tok = [], 0, 0
    for row in cases.itertuples():
        out = model.run(template, row.input, labels, row.true_label)
        passed = score_case(out["output"], row.assertion, row.expected)
        results.append(int(passed))
        input_tok += out["input_tokens"]
        output_tok += out["output_tokens"]

    cfg = get_config()["cost"]
    cost = input_tok / 1000 * cfg["usd_per_1k_input"] + output_tok / 1000 * cfg["usd_per_1k_output"]
    return {
        "pass_rate": round(float(np.mean(results)), 4),
        "n_cases": len(results),
        "results": results,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "cost_usd": round(cost, 5),
    }


def bootstrap_ab(
    results_a: list[int],
    results_b: list[int],
    iters: int,
    seed: int = 42,
    groups: list[str] | None = None,
) -> dict:
    """Paired bootstrap of the pass-rate difference (B - A) over shared cases.

    With ``groups`` (one key per case - normally the case's input text) the
    resampling unit is the group, not the case: every case of a drawn group
    comes along. Cases that share an input are not independent evidence - a
    deterministic model answers a repeated input identically - and resampling
    them as if they were makes the interval too narrow. The seeded suites here
    draw 60 cases from ~15 seed phrases, so this is not a corner case.
    """
    rng = np.random.default_rng(seed)
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if groups is None:
        clusters = [np.array([i]) for i in range(n)]
    else:
        members: dict[str, list[int]] = {}
        for i, key in enumerate(groups[:n]):
            members.setdefault(str(key), []).append(i)
        clusters = [np.array(v) for v in members.values()]
    m = len(clusters)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = np.concatenate([clusters[j] for j in rng.integers(0, m, m)])
        diffs[i] = b[idx].mean() - a[idx].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "delta": round(float(b.mean() - a.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "b_wins_significant": bool(lo > 0),
        "a_wins_significant": bool(hi < 0),
        "n_cases": n,
        "n_groups": m,
    }


def regression_check(candidate_pass: float, baseline_pass: float) -> dict:
    tol = get_config()["eval"]["regression_tolerance"]
    drop = baseline_pass - candidate_pass
    return {
        "baseline_pass_rate": round(baseline_pass, 4),
        "candidate_pass_rate": round(candidate_pass, 4),
        "regression": round(max(drop, 0.0), 4),
        "passes_gate": bool(drop <= tol),
        "tolerance": tol,
    }
