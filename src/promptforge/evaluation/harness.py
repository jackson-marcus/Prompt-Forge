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


def bootstrap_ab(results_a: list[int], results_b: list[int], iters: int, seed: int = 42) -> dict:
    """Paired bootstrap of the pass-rate difference (B - A) over shared cases."""
    rng = np.random.default_rng(seed)
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "delta": round(float(b.mean() - a.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "b_wins_significant": bool(lo > 0),
        "a_wins_significant": bool(hi < 0),
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
