"""Deterministic, prompt-quality-aware simulated model.

Fully offline: a real LLM would introduce nondeterminism and network cost the
workbench doesn't need to demonstrate. Output quality depends on measurable
prompt-engineering features (format spec, few-shot examples, explicit
constraints) — so A/B evaluation recovers a real, planted signal, and the
harness that scores it is production code.
"""

from __future__ import annotations

import hashlib
import re


def prompt_features(template: str) -> dict:
    lowered = template.lower()
    return {
        "has_format_spec": bool(
            re.search(
                r"(respond with|answer with only|output only|only output|output one of|"
                r"reply with just|as json|one word|one of:|category name|just the)",
                lowered,
            )
        ),
        "has_fewshot": lowered.count("example:") >= 1 or "->" in template,
        "has_constraint": bool(re.search(r"(do not|don't|only|must not|never)\b", lowered)),
    }


def _seeded_unit(*parts: str) -> float:
    digest = hashlib.md5("||".join(parts).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class SimulatedModel:
    """Renders {input} into the template and produces a labeled answer whose
    correctness + formatting depend on the prompt's engineering features."""

    name = "simulated"

    def __init__(self, base_accuracy: float = 0.55) -> None:
        self.base_accuracy = base_accuracy
        self.calls = 0

    def run(self, template: str, task_input: str, labels: list[str], true_label: str) -> dict:
        self.calls += 1
        feats = prompt_features(template)
        p_correct = self.base_accuracy + 0.2 * feats["has_fewshot"] + 0.12 * feats["has_constraint"]
        p_correct = min(p_correct, 0.97)

        roll = _seeded_unit(template, task_input, "correct")
        if roll < p_correct:
            label = true_label
        else:
            others = [x for x in labels if x != true_label] or [true_label]
            idx = int(_seeded_unit(template, task_input, "wrong") * len(others))
            label = others[min(idx, len(others) - 1)]

        # formatting: without a format instruction the model wraps the label in prose
        if feats["has_format_spec"]:
            text = label
        else:
            text = f"The answer is {label}."

        prompt_text = template.replace("{input}", task_input)
        return {
            "output": text,
            "input_tokens": max(1, len(prompt_text.split())),
            "output_tokens": max(1, len(text.split())),
        }
