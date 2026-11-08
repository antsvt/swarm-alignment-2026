"""Frozen task-manifest loading shared by local and cloud runners."""

from __future__ import annotations

import json
from pathlib import Path

from swarm_alignment.benchmark import TaskSpec

KAGGLE_VALIDATION_TASK_IDS = (
    "test/number_theory/572.json",
    "test/number_theory/515.json",
    "test/precalculus/927.json",
    "test/prealgebra/1139.json",
    "test/number_theory/1032.json",
)


def load_selected_tasks(path: Path, task_ids: tuple[str, ...]) -> tuple[TaskSpec, ...]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["task_id"]: item for item in manifest["tasks"]}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"MATH-500 manifest is missing selected task IDs: {missing}")
    return tuple(
        TaskSpec(task_id, by_id[task_id]["problem"], by_id[task_id]["answer"])
        for task_id in task_ids
    )
