#!/usr/bin/env python3
"""CPU scripted multi-round demonstration for B0, B1, and B2."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from swarm_alignment.benchmark import (
    Action,
    BenchmarkConfig,
    BenchmarkEnvironment,
    TaskSpec,
    config_to_dict,
)
from swarm_alignment.population import PopulationConfig, RoundRobinScheduler, build_agents
from swarm_alignment.replay import replay, write_events


def math_tasks() -> tuple[TaskSpec, ...]:
    manifest = json.loads(Path("configs/math500-subset.json").read_text(encoding="utf-8"))
    return tuple(
        TaskSpec(item["task_id"], item["problem"], item["answer"])
        for item in manifest["tasks"]
    )


def make_environment(condition: str, perturbation: str | None, count: int, seed: int):
    qwen = (count + 1) // 2
    population = PopulationConfig(
        count,
        {"qwen": qwen, "gemma": count - qwen},
        teams_enabled=condition != "B0",
        team_assignment="mixed",
        max_rounds=4,
        action_budget=8,
    )
    agents = build_agents(population)
    config = BenchmarkConfig(
        condition=condition,  # type: ignore[arg-type]
        seed=seed,
        action_budget=population.action_budget,
        agents=agents,
        tasks=math_tasks(),
        perturbation=perturbation if condition == "B2" else None,
        max_rounds=population.max_rounds,
    )
    return BenchmarkEnvironment(config, f"{condition.lower()}-{uuid.uuid4().hex}"), population


def run(condition: str, perturbation: str | None, count: int, seed: int, output: Path) -> dict:
    env, population = make_environment(condition, perturbation, count, seed)
    agent_ids = list(env.agents)
    scheduler = RoundRobinScheduler(agent_ids, population.max_rounds)
    assigned = {
        agent_id: env.config.tasks[index % len(env.config.tasks)]
        for index, agent_id in enumerate(agent_ids)
    }
    for turn in scheduler.turns():
        actor = turn.agent_id
        position = agent_ids.index(actor)
        task = assigned[actor]
        if turn.round_number == 1:
            answer = "incorrect" if position == len(agent_ids) - 1 else task.answer
            env.apply(actor, Action("solve", task.task_id, answer))
        elif turn.round_number == 2:
            kind = "verify" if position % 2 == 0 else "submit"
            env.apply(actor, Action(kind, task.task_id))
        elif turn.round_number == 3:
            if position % 2 == 0:
                env.apply(actor, Action("submit", task.task_id))
            elif condition == "B0":
                env.apply(actor, Action("noop"))
            elif position == 1:
                env.apply(actor, Action("message", target=agent_ids[0], text="I submitted."))
            else:
                env.apply(
                    actor,
                    Action("inspect", task_id=assigned[agent_ids[1]].task_id, target=agent_ids[1]),
                )
        else:
            if condition == "B0":
                env.apply(actor, Action("noop"))
            elif position == 0 and condition == "B2" and perturbation != "neutral":
                env.observe(actor)
                accused = agent_ids[1] if perturbation != "false" else agent_ids[0]
                env.apply(
                    actor,
                    Action("report", task_id=assigned[accused].task_id, target=accused),
                )
            elif position == 3 and condition == "B1":
                accused = agent_ids[1]
                env.apply(
                    actor,
                    Action("report", task_id=assigned[accused].task_id, target=accused),
                )
            elif position == 0:
                env.observe(actor)
                env.apply(actor, Action("noop"))
            elif position == 1:
                env.observe(actor)
                env.apply(actor, Action("noop"))
            else:
                env.apply(actor, Action("noop"))

        if turn.round_number == 3 and turn.turn_number == len(agent_ids) and condition == "B2":
            accused = agent_ids[1] if perturbation == "true" else agent_ids[0]
            env.inject_perturbation(
                target=accused,
                task_id=assigned[accused].task_id,
                recipient=agent_ids[0],
                source=agent_ids[2],
            )

    summary = env.finish()
    write_events(output / "events.jsonl", env.events)
    replay(env.events)
    resolved = {
        "benchmark": config_to_dict(env.config),
        "driver": {"type": "scripted", "model_inference": False},
        "population": asdict(population),
        "dataset": {
            "name": "HuggingFaceH4/MATH-500",
            "revision": "2343f79",
            "task_ids": [task.task_id for task in env.config.tasks],
        },
    }
    (output / "config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "inference.jsonl").write_text("", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["B0", "B1", "B2"], default="B1")
    parser.add_argument("--perturbation", choices=["true", "false", "neutral"], default="neutral")
    parser.add_argument("--agents", type=int, choices=range(5, 21), default=5)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    output = args.output_root / f"swarmbench_{args.condition.lower()}_{uuid.uuid4().hex}"
    output.mkdir(parents=True)
    summary = run(args.condition, args.perturbation, args.agents, args.seed, output)
    print(json.dumps({"artifact": str(output), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
