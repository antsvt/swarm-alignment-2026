.PHONY: install test lint demo diagnostic replay-smoke

install:
	uv sync --extra dev

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests scripts

demo:
	.venv/bin/python scripts/run_swarmbench_lite.py --condition B1

diagnostic:
	.venv/bin/python scripts/run_compliance_diagnostic.py

replay-smoke:
	.venv/bin/python scripts/replay_run.py artifacts/b0-kaggle-v5

