from swarm_alignment.benchmark import AgentSpec, BenchmarkConfig, TaskSpec
from swarm_alignment.diagnostics import DIAGNOSTIC_CONDITIONS, diagnostic_config, diagnostic_prompt


def test_diagnostic_conditions_are_explicit_and_costed():
    assert [condition.label for condition in DIAGNOSTIC_CONDITIONS] == [
        "A-original",
        "B-explicit",
        "C-free",
    ]
    assert [condition.verification_cost for condition in DIAGNOSTIC_CONDITIONS] == [1, 1, 0]
    assert "internally does not count" not in diagnostic_prompt(DIAGNOSTIC_CONDITIONS[0])
    assert "internally does not count" in diagnostic_prompt(DIAGNOSTIC_CONDITIONS[1])


def test_free_condition_changes_cost_not_base_protocol():
    base = BenchmarkConfig(
        "B0", 1, 3, (AgentSpec("A", "mock"),), (TaskSpec("t", "1+1", "2"),)
    )
    free = diagnostic_config(base, DIAGNOSTIC_CONDITIONS[2])
    assert free.verification_cost == 0
    assert free.agents == base.agents and free.tasks == base.tasks
