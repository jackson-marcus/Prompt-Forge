"""Automated prompt optimisation: hill-climb over prompt edits, gated by the A/B harness.

The registry versions prompts and the harness scores them; this package closes
the loop. Starting from a variant's head it proposes structural edits (a format
instruction, a constraint, few-shot examples, a persona line), scores each one
on a *dev* split of the task's cases, accepts a step only when the configured
acceptance rule says the gain is real, and reports the final prompt on the
held-out *test* split so the number you read is not the number that chose it.
"""
