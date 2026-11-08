"""Strict deterministic replay for SwarmBench-Lite event logs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from swarm_alignment.benchmark import (
    Action,
    BenchmarkEnvironment,
    ProtocolError,
    config_from_dict,
)


class ReplayError(ValueError):
    pass


def write_events(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ReplayError(f"invalid JSON on line {line_number}") from error
    return events


def replay(events: list[dict[str, Any]]) -> BenchmarkEnvironment:
    if not events:
        raise ReplayError("empty event log")
    required = {"schema_version", "run_id", "seq", "type", "actor", "payload"}
    run_id = events[0].get("run_id")
    for index, event in enumerate(events):
        if not required.issubset(event):
            raise ReplayError(f"event {index} is missing required fields")
        if event["seq"] != index:
            raise ReplayError(f"event sequence is missing, duplicated, or reordered at {index}")
        if event["schema_version"] != 1 or event["run_id"] != run_id:
            raise ReplayError(f"event {index} has inconsistent metadata")
    first = events[0]
    if first["type"] != "run.started" or "config" not in first["payload"]:
        raise ReplayError("first event must be run.started with config")
    if events[-1]["type"] != "run.completed":
        raise ReplayError("log is incomplete: terminal run.completed is missing")

    env = BenchmarkEnvironment(config_from_dict(first["payload"]["config"]), run_id=run_id)
    for event in events[1:-1]:
        before = len(env.events)
        if event["type"] == "action.applied":
            try:
                env.apply(event["actor"], Action(**event["payload"]["action"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ReplayError(f"cannot apply event {event['seq']}: {error}") from error
        elif event["type"] == "action.rejected":
            try:
                env.apply(event["actor"], Action(**event["payload"]["action"]))
            except ProtocolError:
                pass
            else:
                raise ReplayError(f"rejected event {event['seq']} unexpectedly succeeded")
        elif event["type"] == "observation.created":
            env.observe(event["actor"])
        elif event["type"] == "inference.failed":
            payload = event["payload"]
            env.record_inference_failure(
                event["actor"], payload["phase"], payload["error"], payload.get("task_id")
            )
        elif event["type"] == "perturbation.injected":
            visible = event["payload"].get("visible", {})
            try:
                env.inject_perturbation(
                    target=visible.get("subject", visible.get("accused")),
                    task_id=visible.get("task_id"),
                    recipient=visible.get("recipient"),
                    source=visible.get("sender", "system"),
                )
            except ValueError as error:
                raise ReplayError(f"cannot apply event {event['seq']}: {error}") from error
        else:
            raise ReplayError(f"unknown replay event type: {event['type']}")
        reconstructed = env.events[before]
        if reconstructed != event:
            if not _matches_legacy_action(reconstructed, event):
                raise ReplayError(
                    f"event {event['seq']} result is inconsistent with reconstructed state"
                )
            # Pre-isolation logs included scorer-only fields in the actor history. Restore the
            # recorded history solely for digest compatibility; new environments never expose it.
            env.histories[event["actor"]][-1]["result"] = event["payload"]["result"]
            env.events[before] = event

    terminal = events[-1]["payload"]
    recorded_summary = terminal.get("summary", {})
    current_summary = env.summary()
    summary_changed = any(
        current_summary.get(key) != value for key, value in recorded_summary.items()
    )
    if "agent_count" in recorded_summary:
        digest = env.state_digest()
    else:
        legacy_state = {
            "remaining": env.remaining,
            "pending": sorted(
                (
                    {"agent": agent, "task": task, **value}
                    for (agent, task), value in env.pending.items()
                ),
                key=lambda item: (item["agent"], item["task"]),
            ),
            "submissions": env.submissions,
            "scoreboard": env.scoreboard,
            "messages": env.messages,
            "inspections": env.inspections,
            "reports": env.reports,
        }
        raw = json.dumps(legacy_state, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
    digest_changed = terminal.get("state_digest") != digest
    audit_changed = "audit" in terminal and terminal["audit"] != {
        "submissions": env.submissions,
        "reports": env.reports,
    }
    if summary_changed or digest_changed or audit_changed:
        raise ReplayError("terminal summary or state digest is inconsistent")
    env.events.append(events[-1])
    return env


def replay_file(path: Path) -> BenchmarkEnvironment:
    return replay(read_events(path))


def _matches_legacy_action(reconstructed: dict[str, Any], recorded: dict[str, Any]) -> bool:
    """Accept pre-isolation logs whose action result included hidden scorer fields."""

    if reconstructed.get("type") != "action.applied" or recorded.get("type") != "action.applied":
        return False
    normalized = json.loads(json.dumps(recorded))
    result = normalized.get("payload", {}).get("result", {})
    kind = normalized.get("payload", {}).get("action", {}).get("kind")
    if kind == "submit":
        result.pop("correct", None)
        result.pop("violation", None)
    elif kind == "report":
        result.pop("substantiated", None)
    else:
        return False
    return reconstructed == normalized
