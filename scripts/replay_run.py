#!/usr/bin/env python3
"""Replay one SwarmBench-Lite event stream without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_alignment.replay import replay_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        help="Path to events.jsonl or to a run directory containing it",
    )
    args = parser.parse_args()
    path = args.path / "events.jsonl" if args.path.is_dir() else args.path
    env = replay_file(path)
    print(
        json.dumps(
            {
                "events": len(env.events),
                "path": str(path),
                "state_digest": env.events[-1]["payload"]["state_digest"],
                "summary": env.summary(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
