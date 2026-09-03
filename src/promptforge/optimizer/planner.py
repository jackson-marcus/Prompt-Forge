"""Planner: which edits are worth trying on this head, and what they may look at.

The planner owns the information boundary of the search. It builds a
:class:`TaskContext` from the *dev* split only - the label set and one
exemplar per label - and asks every operator whether it still applies to the
current template. An operator whose feature is already present is not offered
again, so the candidate set shrinks as the climb progresses and the loop
terminates on its own once every edit has been either accepted or rejected.
"""

from __future__ import annotations

import pandas as pd

from promptforge.optimizer.edits import (
    Candidate,
    EditOperator,
    TaskContext,
    default_operators,
)


def mine_examples(dev_cases: pd.DataFrame, per_label: int = 1) -> tuple[tuple[str, str], ...]:
    """The first ``per_label`` (input, label) pairs of each label, in dev-split order."""
    picked: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for row in dev_cases.itertuples():
        label = str(row.true_label)
        if seen.get(label, 0) >= per_label:
            continue
        seen[label] = seen.get(label, 0) + 1
        picked.append((str(row.input), label))
    return tuple(picked)


class Planner:
    def __init__(self, operators: list[EditOperator] | None = None) -> None:
        self.operators = operators or default_operators()

    def context(self, dev_cases: pd.DataFrame) -> TaskContext:
        labels = tuple(sorted(dev_cases["true_label"].unique().tolist()))
        return TaskContext(labels=labels, examples=mine_examples(dev_cases))

    def propose(self, template: str, ctx: TaskContext) -> list[Candidate]:
        """Every applicable single edit of ``template``, one candidate per operator."""
        candidates = []
        for op in self.operators:
            candidate = op.propose(template, ctx)
            if candidate is not None:
                candidates.append(candidate)
        return candidates
