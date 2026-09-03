"""The climb: propose, score on dev, gate, accept the best, repeat; then judge on test.

One round scores every applicable single edit of the current head on the dev
split and asks the acceptance rule whether each one beat the head. The best
accepted candidate becomes the new head; if nothing is accepted, the search
stops. The three rules trade false accepts against starvation:

* ``greedy`` - accept any positive dev delta. Cheap, and happily accepts an
  edit that did nothing except re-roll the noise.
* ``safe``   - positive delta, and the CI's lower bound must not dip below the
  regression tolerance. A middle ground.
* ``gated``  - the paired-bootstrap CI must clear zero, the same bar the
  leaderboard uses for "beats baseline". On a small dev split this refuses
  genuine gains because the interval is too wide to see them.

The final prompt is then scored on the held-out test split against the
starting prompt, so the reported gain was not chosen by the data it is
reported on. ``scripts/bench_optimizer.py`` measures all three rules against
the simulator's planted signal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from promptforge.evaluation.harness import bootstrap_ab, regression_check, run_suite
from promptforge.llm.simulate import SimulatedModel
from promptforge.optimizer.edits import Candidate
from promptforge.optimizer.ledger import BudgetExhaustedError, Ledger, Scored
from promptforge.optimizer.planner import Planner
from promptforge.settings import get_config

AcceptanceRule = Callable[[dict[str, Any], float], bool]

POLICIES: dict[str, AcceptanceRule] = {
    "greedy": lambda ab, tol: ab["delta"] > 0,
    "safe": lambda ab, tol: ab["delta"] > 0 and ab["ci_low"] >= -tol,
    "gated": lambda ab, tol: bool(ab["b_wins_significant"]),
}


def split_cases(
    cases: pd.DataFrame, dev_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle once and cut: the first ``dev_fraction`` chooses edits, the rest judges them."""
    if len(cases) < 2:
        raise ValueError("need at least two cases to hold anything out")
    order = np.random.default_rng(seed).permutation(len(cases))
    n_dev = min(max(round(len(cases) * dev_fraction), 1), len(cases) - 1)
    dev = cases.iloc[order[:n_dev]].reset_index(drop=True)
    test = cases.iloc[order[n_dev:]].reset_index(drop=True)
    return dev, test


class HillClimb:
    def __init__(
        self,
        *,
        policy: str | None = None,
        max_rounds: int | None = None,
        max_calls: int | None = None,
        dev_fraction: float | None = None,
        bootstrap_iters: int | None = None,
        seed: int = 0,
        cluster_by_input: bool = True,
        planner: Planner | None = None,
        model: SimulatedModel | None = None,
    ) -> None:
        cfg = get_config()
        opt = cfg["optimizer"]
        self.policy = policy or opt["policy"]
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; choose from {sorted(POLICIES)}")
        self.rule = POLICIES[self.policy]
        self.max_rounds = max_rounds or opt["max_rounds"]
        self.max_calls = max_calls or opt["max_model_calls"]
        self.dev_fraction = dev_fraction or opt["dev_fraction"]
        self.bootstrap_iters = bootstrap_iters or cfg["eval"]["bootstrap_iters"]
        self.tolerance = cfg["eval"]["regression_tolerance"]
        self.seed = seed
        # resample distinct inputs, not rows: repeated inputs are one unit of evidence
        self.cluster_by_input = cluster_by_input
        self.planner = planner or Planner()
        self.model = model or SimulatedModel()

    def _ab(self, base: list[int], cand: list[int], cases: pd.DataFrame) -> dict[str, Any]:
        groups = cases["input"].tolist() if self.cluster_by_input else None
        return bootstrap_ab(base, cand, self.bootstrap_iters, groups=groups)

    def _score(self, ledger: Ledger, candidate: Candidate, cases: pd.DataFrame) -> Scored:
        if ledger.has_scored(candidate.content_hash):
            ledger.duplicates_skipped += 1
            return ledger.scored[candidate.content_hash]
        ledger.reserve(len(cases))
        run = run_suite(candidate.template, cases, self.model)
        return ledger.charge(candidate.edit, candidate.template, candidate.content_hash, run)

    def optimize(self, template: str, cases: pd.DataFrame) -> dict[str, Any]:
        dev, test = split_cases(cases, self.dev_fraction, self.seed)
        ledger = Ledger(max_calls=self.max_calls)
        ctx = self.planner.context(dev)
        start = head = self._score(ledger, Candidate.create("start", template), dev)
        steps: list[dict[str, Any]] = []
        rounds = 0
        stop_reason = "max rounds reached"
        for round_no in range(1, self.max_rounds + 1):
            candidates = self.planner.propose(head.template, ctx)
            if not candidates:
                stop_reason = "no applicable edits left"
                break
            rounds = round_no
            accepted: list[Scored] = []
            try:
                for cand in candidates:
                    scored = self._score(ledger, cand, dev)
                    ab = self._ab(head.results, scored.results, dev)
                    ok = self.rule(ab, self.tolerance)
                    ledger.record(
                        round=round_no,
                        edit=cand.edit,
                        dev_pass_rate=scored.pass_rate,
                        delta=ab["delta"],
                        ci_low=ab["ci_low"],
                        ci_high=ab["ci_high"],
                        cost_usd=scored.cost_usd,
                        accepted=ok,
                    )
                    if ok:
                        accepted.append(scored)
            except BudgetExhaustedError as exc:
                stop_reason = f"budget exhausted: {exc}"
                break
            if not accepted:
                stop_reason = f"no candidate passed the {self.policy} rule"
                break
            # ties on pass rate go to the cheaper prompt
            best = max(accepted, key=lambda s: (s.pass_rate, -s.cost_usd))
            steps.append(
                {
                    "round": round_no,
                    "edit": best.edit,
                    "dev_pass_rate": best.pass_rate,
                    "cost_usd": best.cost_usd,
                    "content_hash": best.content_hash,
                    "template": best.template,
                }
            )
            head = best

        return {
            "policy": self.policy,
            "seed": self.seed,
            "start_template": start.template,
            "final_template": head.template,
            "steps": steps,
            "rounds": rounds,
            "stop_reason": stop_reason,
            "holdout": self._holdout(ledger, start, head, test),
            "ledger": ledger.to_dict(),
        }

    def _holdout(
        self, ledger: Ledger, start: Scored, head: Scored, test: pd.DataFrame
    ) -> dict[str, Any]:
        """Judge start vs final on cases the climb never saw. Not subject to the budget."""
        start_run = run_suite(start.template, test, self.model)
        ledger.book(start_run, holdout=True)
        if head.content_hash == start.content_hash:
            final_run = start_run
        else:
            final_run = run_suite(head.template, test, self.model)
            ledger.book(final_run, holdout=True)
        ab = self._ab(start_run["results"], final_run["results"], test)
        return {
            "n_dev": len(start.results),
            "n_test": len(test),
            "start": {"dev_pass_rate": start.pass_rate, "test_pass_rate": start_run["pass_rate"]},
            "final": {"dev_pass_rate": head.pass_rate, "test_pass_rate": final_run["pass_rate"]},
            "ab": ab,
            "regression": regression_check(final_run["pass_rate"], start_run["pass_rate"]),
            "optimism": round(head.pass_rate - final_run["pass_rate"], 4),
        }
