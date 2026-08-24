"""Assertion-based scorers for model outputs."""

from __future__ import annotations

import json
import re


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def exact_match(output: str, expected: str) -> bool:
    return _normalize(output) == _normalize(expected)


def contains(output: str, expected: str) -> bool:
    return _normalize(expected) in _normalize(output)


def is_json(output: str, _expected: str) -> bool:
    try:
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def regex_match(output: str, pattern: str) -> bool:
    return re.search(pattern, output) is not None


ASSERTIONS = {
    "exact_match": exact_match,
    "contains": contains,
    "is_json": is_json,
    "regex": regex_match,
}


def score_case(output: str, assertion: str, expected: str) -> bool:
    fn = ASSERTIONS.get(assertion)
    if fn is None:
        raise ValueError(f"unknown assertion {assertion}")
    return bool(fn(output, expected))
