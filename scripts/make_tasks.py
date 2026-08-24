"""Task suites + seed prompt variants of varying engineering quality.

Two tasks (sentiment, intent) each get labeled cases with an exact-match
assertion, plus three prompt variants:
  - bare: no format spec, no examples
  - formatted: adds an explicit format instruction
  - engineered: format + few-shot + constraint

Usage:
    uv run python scripts/make_tasks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from promptforge.registry.store import register_prompt
from promptforge.settings import get_config, resolve_path

TASKS = {
    "sentiment": {
        "labels": ["positive", "negative", "neutral"],
        "seeds": {
            "positive": [
                "loved it",
                "works great",
                "highly recommend",
                "excellent value",
                "fast and smooth",
            ],
            "negative": [
                "broke immediately",
                "terrible support",
                "waste of money",
                "arrived damaged",
                "very disappointed",
            ],
            "neutral": [
                "it is fine",
                "as described",
                "average product",
                "nothing special",
                "does the job",
            ],
        },
        "variants": {
            "bare": "Classify the sentiment of this review: {input}",
            "formatted": "Classify the sentiment. Answer with only one word (positive, negative, or neutral): {input}",
            "engineered": (
                "Classify the sentiment. Do not explain. Answer with only one word.\n"
                "Example: 'awful experience' -> negative\n"
                "Example: 'pretty good' -> positive\n"
                "Review: {input}"
            ),
        },
    },
    "intent": {
        "labels": ["refund", "shipping", "account", "technical"],
        "seeds": {
            "refund": [
                "i want my money back",
                "requesting a refund",
                "charge was wrong",
                "cancel and refund",
            ],
            "shipping": [
                "where is my package",
                "delivery is late",
                "track my order",
                "wrong address",
            ],
            "account": [
                "reset my password",
                "change my email",
                "locked out",
                "update billing info",
            ],
            "technical": [
                "app keeps crashing",
                "cannot log in",
                "error on checkout",
                "page wont load",
            ],
        },
        "variants": {
            "bare": "What is the user's intent? {input}",
            "formatted": "Identify the intent. Reply with just the category name: {input}",
            "engineered": (
                "Identify the intent. Only output one of: refund, shipping, account, technical.\n"
                "Example: 'give me a refund' -> refund\n"
                "Message: {input}"
            ),
        },
    },
}


def build_cases(task: str, spec: dict, n: int, rng) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        label = str(rng.choice(spec["labels"]))
        seed = str(rng.choice(spec["seeds"][label]))
        rows.append(
            {
                "task": task,
                "input": seed,
                "true_label": label,
                "expected": label,
                "assertion": "exact_match",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = get_config()
    rng = np.random.default_rng(cfg["data"]["seed"])
    out = resolve_path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # reset registry
    (out / "registry.json").write_text(json.dumps({"prompts": {}}), encoding="utf-8")

    frames = []
    for task, spec in TASKS.items():
        frames.append(build_cases(task, spec, cfg["data"]["n_cases_per_task"], rng))
        for i, (name, template) in enumerate(spec["variants"].items()):
            register_prompt(task, name, template, set_baseline=(i == 0))
    pd.concat(frames, ignore_index=True).to_parquet(out / "cases.parquet", index=False)
    print(json.dumps({"tasks": list(TASKS), "cases": sum(len(f) for f in frames)}))


if __name__ == "__main__":
    main()
