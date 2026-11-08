"""Configurable logical-agent populations and deterministic scheduling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from swarm_alignment.benchmark import AgentSpec

TeamAssignment = Literal["mixed", "model_aligned"]


@dataclass(frozen=True, slots=True)
class PopulationConfig:
    count: int
    models: dict[str, int]
    teams_enabled: bool = True
    team_assignment: TeamAssignment = "mixed"
    team_count: int = 2
    scheduler_mode: str = "round_robin"
    max_concurrent_inference: int = 1
    max_rounds: int = 3
    action_budget: int = 8

    def __post_init__(self) -> None:
        if not 5 <= self.count <= 20:
            raise ValueError("logical-agent count must be between 5 and 20")
        wrong_total = sum(self.models.values()) != self.count
        negative_count = any(value < 0 for value in self.models.values())
        if wrong_total or negative_count:
            raise ValueError("model counts must be non-negative and sum to count")
        if self.teams_enabled and self.team_count < 2:
            raise ValueError("team_count must be at least two")
        if self.team_assignment not in {"mixed", "model_aligned"}:
            raise ValueError("unknown team assignment")
        if self.scheduler_mode != "round_robin" or self.max_concurrent_inference < 1:
            raise ValueError("only deterministic round_robin scheduling is supported")
        if self.max_rounds < 1 or self.action_budget < 1:
            raise ValueError("round and action budgets must be positive")


@dataclass(frozen=True, slots=True)
class ScheduledTurn:
    round_number: int
    turn_number: int
    agent_id: str


class RoundRobinScheduler:
    def __init__(self, agent_ids: list[str], max_rounds: int) -> None:
        if len(agent_ids) != len(set(agent_ids)) or not agent_ids or max_rounds < 1:
            raise ValueError("scheduler requires unique agents and positive rounds")
        self.agent_ids = tuple(agent_ids)
        self.max_rounds = max_rounds

    def turns(self) -> tuple[ScheduledTurn, ...]:
        return tuple(
            ScheduledTurn(round_number, turn, agent_id)
            for round_number in range(1, self.max_rounds + 1)
            for turn, agent_id in enumerate(self.agent_ids, 1)
        )


def build_agents(config: PopulationConfig) -> tuple[AgentSpec, ...]:
    """Create stable logical IDs; repeated agents share the same model backend name."""

    families = sorted(config.models)
    prefixes = _unique_prefixes(families)
    per_family: dict[str, list[tuple[str, str]]] = {}
    for family in families:
        model_name = _model_name(family)
        per_family[family] = [
            (f"{prefixes[family]}{index}", model_name)
            for index in range(1, config.models[family] + 1)
        ]

    ordered: list[tuple[str, str, str]] = []
    if config.team_assignment == "mixed":
        largest = max(config.models.values(), default=0)
        for index in range(largest):
            for family in families:
                if index < len(per_family[family]):
                    agent_id, model = per_family[family][index]
                    ordered.append((agent_id, model, family))
    else:
        for family in families:
            ordered.extend((agent_id, model, family) for agent_id, model in per_family[family])

    agents: list[AgentSpec] = []
    family_positions = {family: 0 for family in families}
    for agent_id, model, family in ordered:
        team: str | None = None
        if config.teams_enabled:
            if config.team_assignment == "mixed":
                team_index = family_positions[family] % config.team_count
                family_positions[family] += 1
            else:
                team_index = families.index(family) % config.team_count
            team = f"team-{team_index + 1}"
        agents.append(AgentSpec(agent_id, model, team, role="solver"))
    return tuple(agents)


def backend_groups(agents: tuple[AgentSpec, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for agent in agents:
        groups.setdefault(agent.model, []).append(agent.agent_id)
    return {model: tuple(agent_ids) for model, agent_ids in groups.items()}


def load_population_config(path: Path) -> PopulationConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PopulationConfig(
        count=int(data["agents"]["count"]),
        models={key: int(value) for key, value in data["agents"]["models"].items()},
        teams_enabled=bool(data["teams"]["enabled"]),
        team_assignment=data["teams"]["assignment"],
        team_count=int(data["teams"].get("count", 2)),
        scheduler_mode=data["scheduler"]["mode"],
        max_concurrent_inference=int(data["scheduler"]["max_concurrent_inference"]),
        max_rounds=int(data["environment"]["max_rounds"]),
        action_budget=int(data["environment"]["action_budget"]),
    )


def _model_name(family: str) -> str:
    names = {
        "qwen": "Qwen/Qwen3-4B-GGUF",
        "gemma": "ggml-org/gemma-3-4b-it-GGUF",
    }
    return names.get(family, family)


def _unique_prefixes(families: list[str]) -> dict[str, str]:
    used: set[str] = set()
    result: dict[str, str] = {}
    for family in families:
        length = 1
        while family[:length].upper() in used and length <= len(family):
            length += 1
        prefix = family[:length].upper()
        if prefix in used:
            raise ValueError("model family names cannot produce unique agent prefixes")
        used.add(prefix)
        result[family] = prefix
    return result
