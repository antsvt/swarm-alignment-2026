#!/usr/bin/env python3
"""Build a readable Kaggle notebook bundle from the current repository sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COMMON_FILES = (
    "src/swarm_alignment/__init__.py",
    "src/swarm_alignment/grading.py",
    "src/swarm_alignment/benchmark.py",
    "src/swarm_alignment/diagnostics.py",
    "src/swarm_alignment/population.py",
    "src/swarm_alignment/replay.py",
    "src/swarm_alignment/tasks.py",
    "configs/math500-subset.json",
    "scripts/run_kaggle_b0.py",
)


def notebook_cell(kind: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def remote_path(local_path: str) -> str:
    path = Path(local_path)
    if path.parts[0] == "src":
        relative = Path(*path.parts[1:])
    elif path.parts[0] == "scripts":
        relative = Path(path.name)
    else:
        relative = path
    return f"/kaggle/working/{relative}"


def build_notebook(repo: Path, runner: str) -> dict[str, Any]:
    files = list(COMMON_FILES)
    if runner == "b1":
        files.append("scripts/run_kaggle_b1.py")
    target = f"/kaggle/working/run_kaggle_{runner}.py"
    cells = [
        notebook_cell(
            "markdown",
            f"# SwarmBench-Lite — real {runner.upper()} validation\n\n"
            "Readable source bundle with no embedded credentials.\n",
        ),
        notebook_cell(
            "code",
            "from pathlib import Path\n"
            "import sys\n"
            "Path('/kaggle/working/swarm_alignment').mkdir(parents=True, exist_ok=True)\n"
            "Path('/kaggle/working/configs').mkdir(parents=True, exist_ok=True)\n"
            "sys.path.insert(0, '/kaggle/working')\n",
        ),
    ]
    for local in files:
        source = (repo / local).read_text(encoding="utf-8")
        cells.append(notebook_cell("code", f"%%writefile {remote_path(local)}\n{source}"))
    cells.append(notebook_cell("code", f"%run {target}\n"))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=("b0", "b1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-id", help="Optional Kaggle owner/slug; writes kernel metadata")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    notebook_name = f"swarmbench_{args.runner}.ipynb"
    notebook = build_notebook(repo, args.runner)
    (args.output / notebook_name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if args.kernel_id:
        if args.kernel_id.count("/") != 1:
            raise ValueError("kernel ID must have the form owner/slug")
        metadata = {
            "id": args.kernel_id,
            "title": args.kernel_id.split("/", 1)[1],
            "code_file": notebook_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (args.output / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    print(args.output / notebook_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
