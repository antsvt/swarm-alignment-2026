import json
from pathlib import Path

import pytest

from swarm_alignment.benchmark import Action, BenchmarkConfig, BenchmarkEnvironment, TaskSpec
from swarm_alignment.population import (
    PopulationConfig,
    RoundRobinScheduler,
    backend_groups,
    build_agents,
    load_population_config,
)
from swarm_alignment.tasks import KAGGLE_VALIDATION_TASK_IDS, load_selected_tasks


@pytest.mark.parametrize("count", [5, 10, 20])
def test_population_sizes_execute_complete_scripted_episode(count):
    qwen = (count + 1) // 2
    population = PopulationConfig(count, {"qwen": qwen, "gemma": count - qwen})
    agents = build_agents(population)
    assert len(agents) == count == len({agent.agent_id for agent in agents})
    assert len({agent.team for agent in agents}) == 2
    env = BenchmarkEnvironment(
        BenchmarkConfig("B1", 3, 3, agents, (TaskSpec("t", "2+2", "4"),))
    )
    for agent in agents:
        env.apply(agent.agent_id, Action("solve", "t", "4"))
        env.apply(agent.agent_id, Action("verify", "t"))
        env.apply(agent.agent_id, Action("submit", "t"))
    summary = env.finish()
    assert summary["agent_count"] == count
    assert summary["completed_agents"] == count
    assert sum(summary["scoreboard"].values()) == count


def test_round_robin_is_stable_and_complete():
    turns = RoundRobinScheduler(["A", "B", "C"], 3).turns()
    assert [turn.agent_id for turn in turns] == ["A", "B", "C"] * 3
    assert [turn.round_number for turn in turns] == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_mixed_and_model_aligned_teams_are_distinct():
    mixed = build_agents(PopulationConfig(6, {"qwen": 3, "gemma": 3}, team_assignment="mixed"))
    aligned = build_agents(
        PopulationConfig(6, {"qwen": 3, "gemma": 3}, team_assignment="model_aligned")
    )
    assert any(a.model == b.model and a.team != b.team for a in mixed for b in mixed)
    groups = backend_groups(aligned)
    assert sorted(map(len, groups.values())) == [3, 3]
    assert all(
        len({agent.team for agent in aligned if agent.model == model}) == 1 for model in groups
    )


def test_population_config_files_are_loadable(tmp_path):
    path = tmp_path / "population.json"
    path.write_text(
        json.dumps(
            {
                "agents": {"count": 5, "models": {"qwen": 3, "gemma": 2}},
                "teams": {"enabled": True, "assignment": "mixed", "count": 2},
                "scheduler": {"mode": "round_robin", "max_concurrent_inference": 1},
                "environment": {"max_rounds": 3, "action_budget": 8},
            }
        )
    )
    config = load_population_config(path)
    assert config.count == 5 and config.max_rounds == 3


def test_frozen_math_manifest_has_eight_unique_level_3_to_5_tasks():
    data = json.loads(open("configs/math500-subset.json", encoding="utf-8").read())
    tasks = data["tasks"]
    assert len(tasks) == 8 == len({task["task_id"] for task in tasks})
    assert {task["level"] for task in tasks}.issubset({3, 4, 5})


def test_kaggle_runner_tasks_are_loaded_exactly_from_frozen_manifest():
    data = json.loads(open("configs/math500-subset.json", encoding="utf-8").read())
    by_id = {item["task_id"]: item for item in data["tasks"]}
    loaded = load_selected_tasks(
        Path("configs/math500-subset.json"), KAGGLE_VALIDATION_TASK_IDS
    )
    assert [task.task_id for task in loaded] == list(KAGGLE_VALIDATION_TASK_IDS)
    assert [
        (task.problem, task.answer) for task in loaded
    ] == [
        (by_id[task_id]["problem"], by_id[task_id]["answer"])
        for task_id in KAGGLE_VALIDATION_TASK_IDS
    ]

