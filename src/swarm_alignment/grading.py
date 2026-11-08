"""Pluggable answer grading with a dependency-free deterministic default."""

from __future__ import annotations

from typing import Literal

GraderName = Literal["exact", "math_verify"]


def grade_answer(reference: str, candidate: str, grader: GraderName) -> bool:
    if grader == "exact":
        return candidate.strip() == reference.strip()
    if grader != "math_verify":
        raise ValueError(f"unknown answer grader: {grader}")

    try:
        from math_verify import parse, verify
    except ImportError as error:
        raise RuntimeError(
            "math_verify grading requires `uv sync --extra math` or the Kaggle runtime install"
        ) from error

    gold = parse(reference)
    answer = parse(candidate)
    if not gold or not answer:
        return False
    return bool(verify(gold, answer))
