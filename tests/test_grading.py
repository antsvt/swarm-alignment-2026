import pytest

from swarm_alignment.grading import grade_answer


def test_exact_grader_preserves_deterministic_string_semantics():
    assert grade_answer("16", "16", "exact") is True
    assert grade_answer("16", "x = 16", "exact") is False


def test_math_verify_accepts_equivalent_assignment_and_expression():
    pytest.importorskip("math_verify")
    assert grade_answer("16", "x = 16", "math_verify") is True
    assert grade_answer("3*sqrt(13)", r"3\sqrt{13}", "math_verify") is True
