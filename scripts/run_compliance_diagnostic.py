#!/usr/bin/env python3
"""CPU scripted validation for the A/B/C verification diagnostic."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from swarm_alignment.benchmark import (
    Action,
    AgentSpec,
    BenchmarkConfig,
    BenchmarkEnvironment,
    TaskSpec,
    config_to_dict,
)
from swarm_alignment.diagnostics import DIAGNOSTIC_CONDITIONS, diagnostic_config
from swarm_alignment.replay import replay, write_events


def main() -> int:
    root = Path("artifacts") / f"diagnostic_scripted_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    base = BenchmarkConfig(
        "B0",
        20261108,
        3,
        (AgentSpec("scripted-1", "scripted"),),
        (TaskSpec("diagnostic-1", "Compute 8 * 9.", "72"),),
    )
    results: dict[str, dict] = {}
    for condition in DIAGNOSTIC_CONDITIONS:
        env = BenchmarkEnvironment(
            diagnostic_config(base, condition), f"diagnostic-{condition.label}-{uuid.uuid4().hex}"
        )
        env.apply("scripted-1", Action("solve", "diagnostic-1", "72"))
        if condition.label != "A-original":
            env.apply("scripted-1", Action("verify", "diagnostic-1", "72"))
        env.apply("scripted-1", Action("submit", "diagnostic-1"))
        summary = env.finish()
        condition_dir = root / condition.label
        write_events(condition_dir / "events.jsonl", env.events)
        replay(env.events)
        resolved = {
            "benchmark": config_to_dict(env.config),
            "diagnostic_label": condition.label,
            "instruction": condition.instruction,
            "driver": {"type": "scripted", "model_inference": False},
            "dataset": {"name": "synthetic-compliance-diagnostic", "revision": "1"},
        }
        (condition_dir / "config.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (condition_dir / "inference.jsonl").write_text("", encoding="utf-8")
        (condition_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        results[condition.label] = summary
    (root / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(root), "conditions": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
