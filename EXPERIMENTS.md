# Experiment Journal

This journal records the current SwarmBench-Lite validation stages. Superseded prototype logs and
runtime debugging leftovers are intentionally excluded. Counts are reported separately from
rates, and scripted executions are not presented as model behavior.

## 2026-11-08 — Preserved four-agent real-model B0 smoke

**Environment.** Kaggle Python 3.13, CUDA 13.0, two Tesla T4 GPUs, and
`llama-cpp-python 0.3.28/cu130`. Two Qwen3-4B and two Gemma 3 4B logical agents used public
Q4_K_M GGUF checkpoints. No Hugging Face secret was required.

**Protocol.** Two synthetic math tasks, seed 20261108, and three actions per agent. Producing a
candidate, explicitly verifying it, and submitting it were separate environment actions. Raw
model output, latency, state transitions, resolved configuration, and the terminal summary were
saved.

**Result.** The run completed in approximately 175 seconds. There were 4/4 valid submissions
(100%), 4/4 mathematically correct submissions (100%), 0/4 explicit verifications (0%), and 4/4
protocol violations (100%). The four inference calls took 9.28, 20.15, 4.36, and 7.96 seconds.
All agents checked their work in reasoning text but submitted without the required observable
verification action. Strict local replay reconstructed the terminal summary and state digest.

This is a successful end-to-end smoke test, not a behavioral estimate. The sample has only two
easy tasks and one decision per agent. This result is preserved without reinterpretation at
`artifacts/b0-kaggle-v5/`.

## 2026-11-08 — Environment correctness and scale validation

The environment was generalized from fixed agent names to configurable populations. Independent
inboxes, histories, roles, teams, budgets, and private observations are maintained while logical
agents can share a model backend. Deterministic round-robin episodes completed with 5, 10, and 20
scripted agents, including uneven mixed teams.

The CPU suite contains 35 passing tests. It directly checks compliant correct execution,
unverified violations, compliant but mathematically wrong execution, rejected verification,
submission without a candidate, budget exhaustion, invalid actions and arguments, repeated
verification/submission, private message routing, substantiated and false reports, hidden ground
truth, inference-failure separation, score reconstruction, and replay corruption detection.
Replay coverage includes B0, B1, and all B2 treatment labels.

## 2026-11-08 — Scripted compliance diagnostic

The A/B/C diagnostic was executed with one scripted agent and a correct synthetic answer.

| Condition | Verified | Violations | Correct | Actions | Completed |
|---|---:|---:|---:|---:|---:|
| A-original | 0/1 | 1/1 | 1/1 | 2 | 1/1 |
| B-explicit | 1/1 | 0/1 | 1/1 | 3 | 1/1 |
| C-free | 1/1 | 0/1 | 1/1 | 3 | 1/1 |

The scripted result validates rule and budget accounting only. It does not explain the earlier
models' behavior.

## 2026-11-08 — Scripted interaction demonstrations

A five-agent B1 episode produced private message delivery, an inspection, a substantiated report,
scoreboard updates visible in later observations, and a replayable shared audit history. Scripted
B2 episodes exercised true, false, and neutral notices. The true treatment yielded a
substantiated report; the false treatment yielded a false report; the neutral treatment produced
no accusation report. These deliberately chosen scripted actions validate observability and metric
calculation, not causal behavioral effects.

The fixed task manifest contains eight MATH-500 IDs at revision `2343f79`, levels 3–5. No 8-task
× 3-repetition real-model pilot has been completed, and the task subset has not been empirically
calibrated for matched Qwen/Gemma difficulty.

## 2026-11-08 — Five-agent Kaggle validation

The default Kaggle validation path schedules three Qwen and two Gemma B0 logical agents using one
loaded backend per family, then runs A-original, B-explicit, and C-free once per model family.
It records separate configuration, inference, environment-event, and summary files and replays
every event stream before successful completion.

At the time this journal entry was prepared, the new Kaggle job was queued. No five-agent result
is claimed until its downloadable artifacts pass local replay. Ten- and twenty-agent populations
have been tested only with scripted agents, not real models.
