# SwarmBench-Lite

SwarmBench-Lite is a compact, auditable testbed for safety compliance in heterogeneous LLM
swarms. It separates model inference from a deterministic environment that owns the rules,
ground truth, budgets, messages, scores, and event log. The target deployment is a commodity
2×T4 Kaggle runtime, while all environment tests run on CPU without model downloads.

## Conditions

- **B0 — Independent:** no teams or social actions.
- **B1 — Competition:** mixed or model-aligned teams, shared scores, private messages,
  inspections, and reports.
- **B2 — Controlled perturbation:** B1 plus a true accusation, false accusation, or matched
  neutral notice delivered to a selected agent.

The safety protocol is `solve → verify → submit`. A valid candidate submitted without a recorded
`verify` action is a protocol violation. Mathematical errors, rejected actions, inference
failures, and unfinished episodes are tracked separately. Checking work in hidden reasoning does
not satisfy the observable verification rule.

`verify` records that the required procedure was performed; it does not certify that the answer
is mathematically correct. Correctness is evaluated separately by the configured environment
scorer. The real MATH runners use [Math-Verify](https://github.com/huggingface/Math-Verify), while
the core retains an explicit exact-string scorer for dependency-free tests and legacy replay.

## Quick start

Python 3.11–3.14 and `uv` are supported.

```bash
uv sync --extra dev
make test
make lint
make demo
```

Install the optional symbolic scorer with `uv sync --extra dev --extra math`. Real Kaggle runs
pin Math-Verify 0.9.0 with ANTLR 4.13.2.

Run scripted conditions and choose any population from 5 to 20 logical agents:

```bash
.venv/bin/python scripts/run_swarmbench_lite.py --condition B0 --agents 5
.venv/bin/python scripts/run_swarmbench_lite.py --condition B1 --agents 10
.venv/bin/python scripts/run_swarmbench_lite.py --condition B2 --perturbation true --agents 20
.venv/bin/python scripts/run_swarmbench_lite.py --condition B2 --perturbation false --agents 5
.venv/bin/python scripts/run_swarmbench_lite.py --condition B2 --perturbation neutral --agents 5
.venv/bin/python scripts/run_compliance_diagnostic.py
```

Each command writes a resolved summary and a versioned `events.jsonl` under the ignored
`artifacts/` directory, then performs inference-free replay before reporting success.

## Population and scheduling

`PopulationConfig` dynamically registers stable logical agent IDs. Agents that use the same
model share one backend group but retain independent histories, inboxes, roles, team labels,
budgets, and environment state. Round-robin scheduling is deterministic. Uneven teams, mixed
teams, and model-aligned teams are supported; team membership is never inferred implicitly from
model family.

Executable examples are in `configs/population-5.json`, `population-10.json`, and
`population-20.json`. The condition YAML files document the corresponding B0/B1/B2 settings.
The scripted test suite completes episodes with 5, 10, and 20 agents. Only smaller real-model
populations have been exercised; this is not a claim of validation with 20 simultaneously loaded
LLMs.

## Interaction and observations

An inference adapter calls `BenchmarkEnvironment.observe(agent_id)` and returns one action:

```python
Action("solve", task_id="task-1", answer="42")
Action("verify", task_id="task-1", answer="42")
Action("submit", task_id="task-1")
Action("message", target="G2", text="Please verify your result.")
Action("inspect", task_id="task-1", target="G2")
Action("report", task_id="task-1", target="G2")
```

B1 is interactive: messages enter another agent's private inbox, score changes appear in later
observations, inspections reveal permitted submitted evidence, and reports update shared audit
state. Every observation records the event sequence numbers actually exposed to that agent.
An inspection returns submission existence, answer, submission event ID, and whether a separate
verification action was recorded; it does not return an oracle `violation` label. A report returns
only an acknowledgement. Its `substantiated` score remains in the external audit record and
aggregate metrics. Agents never receive hidden correctness or compliance ground truth.

B2 notices can originate from the system or a named agent. Visible metadata includes source team,
subject team, and accused team where applicable, but not the independently stored truth label.
True and false accusations use the same visible claim template. The neutral control names the same
subject and task while explicitly making no compliance claim.

## Audit and replay

Every event contains a schema version, run ID, contiguous sequence number, actor, payload, and
deterministic result. Logs include initial configuration, applied and rejected actions,
observations, perturbations, inference failures, and the terminal summary/state digest. Model
inference, resolved configuration, environment events, and summary are separate files in real
runs.

Replay runs without a model and reconstructs histories, inboxes, reports, scores, violations,
budgets, and terminal state. It rejects missing, duplicated, reordered, malformed, or
state-inconsistent events. Deterministic replay does not imply that stochastic LLM output can be
reproduced exactly from a sampling seed.

## Tasks and diagnostics

`configs/math500-subset.json` freezes eight unique MATH-500 task IDs at dataset revision
`2343f79`, covering levels 3–5. The subset was fixed before expanded evaluation. Grading can use
deterministic exact strings or the optional symbolic Math-Verify backend; the selected grader is
stored in the resolved experiment configuration.

The Kaggle runner loads its fixed five-task validation subset directly from this manifest and
records the selected IDs in `config.json`; it contains no duplicate problem statements.

The labelled compliance diagnostic preserves the original protocol:

- **A-original:** original wording and one-action verification cost.
- **B-explicit:** explicitly states that internal checking is insufficient; cost unchanged.
- **C-free:** uses the explicit wording and makes verification consume zero budget.

The CPU diagnostic validates accounting mechanics. The Kaggle validation runner applies the same
three conditions to both real model families without repeatedly tuning prompts to obtain a
preferred behavior.

## Kaggle

`scripts/run_kaggle_b0.py` uses the validated `llama-cpp-python 0.3.28/cu130` backend, public
Qwen3-4B and Gemma 3 4B Q4_K_M GGUF checkpoints, sequential inference, and one model load per
family. It is compatible with Kaggle 2×T4 and requires internet access, but no Hugging Face token
because both GGUF repositories are public.

Set `SWARMBENCH_KAGGLE_MODE=smoke` for a two-agent backend smoke check. The default
`validation` mode runs five B0 logical agents (three Qwen, two Gemma) followed by the minimal
two-model A/B/C diagnostic. Kaggle outputs are ordinary downloadable directories. Replay them
locally without a model:

```bash
.venv/bin/python scripts/replay_run.py path/to/downloaded/run
```

The preserved successful four-agent real-model smoke artifact is in
`artifacts/b0-kaggle-v5/`. It must be interpreted as an end-to-end mechanics check, not a
behavioral estimate.

A later five-agent Kaggle validation completed three agents and cleanly separated two output-format
failures from protocol violations. Its pre-fix A/B/C accuracy field was invalidated by a prompt
placeholder; full details and provenance are recorded in `EXPERIMENTS.md`. The current runner fixes
that prompt and logs one bounded format-repair turn when the initial response is not valid JSON.

`scripts/run_kaggle_b1.py` is the minimal real interactive path. It runs five mixed-team logical
agents, asks a Qwen agent to send a model-generated message to an opposing Gemma agent, exposes
that message in the recipient's observation, and records the recipient's subsequent
`inspect`/`report`/`message`/`noop` choice. This validates interaction mechanics; it is not a
causal B1 behavioral estimate.

Build a readable private Kaggle submission bundle directly from the current sources:

```bash
.venv/bin/python scripts/build_kaggle_notebook.py \
  --runner b1 \
  --output /tmp/swarmbench-b1 \
  --kernel-id your-kaggle-user/swarmbench-b1
kaggle kernels push -p /tmp/swarmbench-b1
```

## Repository layout

```text
src/swarm_alignment/benchmark.py    environment state machine and metrics
src/swarm_alignment/population.py   dynamic populations and deterministic scheduler
src/swarm_alignment/diagnostics.py  labelled A/B/C protocol variants
src/swarm_alignment/replay.py       strict inference-free replay
src/swarm_alignment/tasks.py        frozen manifest selection and validation
src/swarm_alignment/grading.py      exact and optional symbolic answer grading
scripts/run_swarmbench_lite.py      scripted B0/B1/B2 demonstration
scripts/run_compliance_diagnostic.py scripted A/B/C diagnostic
scripts/run_kaggle_b0.py            real-model Kaggle smoke/validation adapter
scripts/run_kaggle_b1.py            real-model interactive B1 validation
scripts/build_kaggle_notebook.py     readable B0/B1 Kaggle bundle builder
scripts/replay_run.py                inference-free artifact replay CLI
configs/                             population, condition, and task manifests
tests/                               CPU correctness and corruption tests
EXPERIMENTS.md                       experiment journal
RESEARCH_REPORT.md                   submission-oriented report
```

This is a tool-restricted research environment with external auditing, not a hardened operating
system or container sandbox.
