"""Prompt features, assertions, registry, A/B bootstrap, regression gate, API."""

from __future__ import annotations

import numpy as np
import pandas as pd

from promptforge.evaluation.assertions import score_case
from promptforge.evaluation.harness import bootstrap_ab, regression_check, run_suite
from promptforge.llm.simulate import SimulatedModel, prompt_features


def test_prompt_features_detected():
    engineered = "Answer with only one word. Do not explain.\nExample: 'x' -> y\nReview: {input}"
    feats = prompt_features(engineered)
    assert feats["has_format_spec"] and feats["has_fewshot"] and feats["has_constraint"]

    bare = "Classify the sentiment of: {input}"
    assert not any(prompt_features(bare).values())


def test_assertions():
    assert score_case("positive", "exact_match", "positive")
    assert not score_case("The answer is positive.", "exact_match", "positive")
    assert score_case("The answer is positive.", "contains", "positive")
    assert score_case('{"a": 1}', "is_json", "")
    assert not score_case("not json", "is_json", "")


def _cases(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["positive", "negative", "neutral"]
    seeds = {
        "positive": ["great", "loved it"],
        "negative": ["awful", "broke"],
        "neutral": ["fine", "okay"],
    }
    rows = []
    for _ in range(n):
        label = str(rng.choice(labels))
        rows.append(
            {
                "input": str(rng.choice(seeds[label])),
                "true_label": label,
                "expected": label,
                "assertion": "exact_match",
            }
        )
    return pd.DataFrame(rows)


def test_engineered_prompt_beats_bare_under_exact_match():
    cases = _cases(120, seed=1)
    bare = run_suite("Classify the sentiment of: {input}", cases, SimulatedModel())
    engineered = run_suite(
        "Answer with only one word. Do not explain.\nExample: 'x' -> positive\nReview: {input}",
        cases,
        SimulatedModel(),
    )
    # format spec lifts exact-match; few-shot + constraint lift correctness
    assert engineered["pass_rate"] > bare["pass_rate"] + 0.1
    assert engineered["input_tokens"] > bare["input_tokens"]  # longer prompt costs more


def test_bootstrap_and_regression_gate():
    a = [1, 1, 0, 0, 1, 0, 1, 0] * 20
    b = [1, 1, 1, 1, 1, 0, 1, 0] * 20  # strictly better
    ab = bootstrap_ab(a, b, iters=1000)
    assert ab["delta"] > 0 and ab["b_wins_significant"]

    same = bootstrap_ab(a, a, iters=1000)
    assert not same["b_wins_significant"] and not same["a_wins_significant"]

    assert regression_check(0.80, 0.82)["passes_gate"]  # small drop within tolerance
    assert not regression_check(0.70, 0.82)["passes_gate"]  # big drop fails


def test_evaluation_recovers_variant_ranking(evaluated):
    report = evaluated["report"]
    for task, data in report.items():
        board = {r["variant"]: r["pass_rate"] for r in data["leaderboard"]}
        assert board["engineered"] > board["bare"], task
        top = data["leaderboard"][0]
        assert top["variant"] in {"engineered", "formatted"}


def test_api_contract(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}

    tasks = api_client.get("/tasks").json()["tasks"]
    assert "sentiment" in tasks

    board = api_client.get("/leaderboard/sentiment").json()
    assert board["baseline"] == "bare"
    assert api_client.get("/leaderboard/nope").status_code == 404

    ab = api_client.post(
        "/ab",
        json={
            "task": "sentiment",
            "template_a": "Classify the sentiment of: {input}",
            "template_b": "Answer with only one word.\nExample: 'x' -> positive\nReview: {input}",
        },
    ).json()
    assert ab["b"]["pass_rate"] >= ab["a"]["pass_rate"]
    assert "ci_low" in ab["ab"] and "passes_gate" in ab["regression"]
