import copy

import pytest

from swarm_alignment.benchmark import (
    Action,
    AgentSpec,
    BenchmarkConfig,
    BenchmarkEnvironment,
    ProtocolError,
    TaskSpec,
)
from swarm_alignment.replay import ReplayError, replay


def completed_events(condition="B1", perturbation=None):
    team = None if condition == "B0" else "red"
    config = BenchmarkConfig(
        condition=condition,
        seed=9,
        action_budget=8,
        agents=(AgentSpec("A", "mock", team), AgentSpec("B", "mock", team)),
        tasks=(TaskSpec("t", "2+2", "4"),),
        perturbation=perturbation,
    )
    env = BenchmarkEnvironment(config, f"test-{condition}")
    env.observe("A")
    env.apply("A", Action("solve", "t", "4"))
    env.apply("A", Action("verify", "t"))
    env.apply("A", Action("submit", "t"))
    if condition != "B0":
        env.apply("A", Action("message", target="B", text="done"))
        env.observe("B")
    if condition == "B2":
        env.inject_perturbation(target="A", task_id="t", recipient="B")
    try:
        env.apply("A", Action("submit", "t"))
    except ProtocolError:
        pass
    env.record_inference_failure("B", "solve", "scripted timeout", "t")
    env.finish()
    return env.events


@pytest.mark.parametrize(
    ("condition", "perturbation"),
    [("B0", None), ("B1", None), ("B2", "false"), ("B2", "neutral")],
)
def test_replay_reconstructs_b0_b1_b2(condition, perturbation):
    events = completed_events(condition, perturbation)
    rebuilt = replay(events)
    assert rebuilt.events == events


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "corrupt", "inconsistent"])
def test_replay_fails_loudly_on_bad_logs(mutation):
    events = copy.deepcopy(completed_events())
    if mutation == "missing":
        events.pop(-2)
    elif mutation == "duplicate":
        events.insert(2, copy.deepcopy(events[1]))
    elif mutation == "corrupt":
        action_event = next(event for event in events if event["type"] == "action.applied")
        action_event["payload"]["action"]["task_id"] = "unknown"
    else:
        events[-1]["payload"]["summary"]["violations"] = 99
    with pytest.raises(ReplayError):
        replay(events)


def test_replay_rejects_incomplete_log():
    with pytest.raises(ReplayError, match="incomplete"):
        replay(completed_events()[:-1])


def test_replay_accepts_legacy_submit_result_without_reintroducing_new_leak():
    events = completed_events("B0")
    legacy_env = replay(copy.deepcopy(events))
    submit_event = next(
        event
        for event in events
        if event["type"] == "action.applied" and event["payload"]["action"]["kind"] == "submit"
    )
    submit_event["payload"]["result"].update(correct=True, violation=False)
    legacy_env.histories["A"][-1]["result"].update(correct=True, violation=False)
    events[-1]["payload"].pop("audit")
    events[-1]["payload"]["state_digest"] = legacy_env.state_digest()
    rebuilt = replay(events)
    assert rebuilt.events == events
