# SwarmBench-Lite: A Reproducible Testbed for Safety Compliance in Small Heterogeneous LLM Swarms

## 1. Abstract

SwarmBench-Lite is a compact environment for studying whether safety-rule compliance changes in
small heterogeneous groups of language-model agents. It provides independently enforced action
budgets, mandatory observable verification, team competition, private messages, inspections,
reports, controlled safety-related notices, complete event logging, and inference-free replay.
The environment supports 5–20 logical agents while sharing model weights by family, making it
practical on commodity 2×T4 cloud GPUs. CPU tests validate environment semantics and populations
up to 20 scripted agents. A preserved four-agent Qwen3-4B/Gemma 3 4B Kaggle smoke test validates
the real inference path. The work establishes testbed mechanics; it does not yet establish a
causal behavioral result about intergroup competition.

## 2. Motivation

Multi-agent safety experiments often mix model behavior, orchestration failures, scoring rules,
and infrastructure errors. This makes a surprising trace difficult to audit. SwarmBench-Lite
instead makes the environment the authority over ground truth and records every observable state
transition. Its research question is whether competition and information about other agents'
violations can change verification compliance, selective enforcement, or retaliatory
rule-breaking. The immediate contribution is a reproducible instrument for that question, not a
claim that the effect exists.

## 3. Environment Architecture

The system has four small components. A deterministic state machine validates actions and owns
scores, budgets, candidates, verification state, inboxes, histories, reports, and hidden ground
truth. A population layer creates logical agents and backend-sharing groups. An adapter converts
model output into environment actions. A strict replay engine reconstructs the run without model
inference.

Model backends cannot edit the authoritative state or event log. Observations are filtered per
agent and expose only permitted messages, submissions, scores, and notices. This is tool-level
restriction and external auditing, not hardened container or operating-system isolation.

## 4. Agent Protocol and Safety Constraint

The core task protocol is `solve → verify → submit`. Verification is a distinct environment
action. Internal calculation or self-checking in model text is not observable verification.
Submitting a valid candidate without a successful verify action is therefore a protocol
violation, regardless of mathematical correctness.

The evaluator keeps six outcomes separate: protocol violation, mathematical error, invalid
action, inference failure, incomplete episode, and successful compliant execution. A rejected
action or backend failure is never silently converted into a safety violation.

Three baselines are implemented. B0 is independent work without teams or social tools. B1 adds
team competition, messages, inspections, reports, and a shared scoreboard. B2 adds a controlled
true accusation, false accusation, or neutral notice whose truth is known only to the environment.

## 5. Scalable Logical-Agent Scheduling

Populations are configured by total count, model-family counts, team policy, team count, maximum
rounds, and action budget. Stable IDs are generated dynamically; there are no required
Q1/Q2/G1/G2 names. Mixed and model-aligned team assignment are explicit, and uneven team sizes
are valid.

Logical agents using the same model family share a loaded inference backend while retaining
independent IDs, roles, histories, private inboxes, team membership, and budgets. A deterministic
round-robin scheduler serializes inference by default. Scripted complete episodes validate 5, 10,
and 20 agents. This does not demonstrate 10 or 20 real-model agents; the current real-model target
is five logical agents across two shared backends.

## 6. Auditing and Deterministic Replay

Versioned JSONL events record the resolved initial configuration, sequence number, actor, action
payload, deterministic result, observations, perturbations, rejected actions, inference failures,
and terminal state digest. Real runs keep environment events, raw inference records, configuration,
and summary in separate files.

Replay reconstructs budgets, candidates, verification status, communication histories, reports,
scores, violations, and observations. It fails on missing, duplicated, reordered, malformed, or
state-inconsistent events and compares the final digest and metrics. Replay determinism means a
recorded environment trace can be audited exactly. It does not mean stochastic model generations
are reproduced exactly from the same seed.

## 7. Experimental Setup

The local correctness suite requires only Python. The real adapter targets Kaggle Python 3.13,
CUDA 13.0, and two Tesla T4 GPUs. It uses `llama-cpp-python 0.3.28/cu130` with public Q4_K_M GGUF
checkpoints for Qwen3-4B and Gemma 3 4B. Inference is sequential, with one backend loaded per
family. Public checkpoints remove the need to transmit a Hugging Face token.

Eight MATH-500 identifiers were frozen at dataset revision `2343f79` before expanded evaluation.
They cover levels 3–5 and five subject labels. The current evaluator uses normalized exact-string
answers rather than symbolic equivalence. The subset has not yet been independently calibrated to
equal difficulty for both model families.

The compliance diagnostic has three explicitly labelled conditions. A-original preserves the
original wording and verification cost. B-explicit states that internal checking is insufficient
and keeps the cost. C-free uses the explicit wording and removes only the simulated budget cost of
verification.

## 8. Validation Results

Thirty-five CPU tests pass. Beyond digest equality, tests assert the intended scoring semantics:
correct compliant submission, unverified violation, verified mathematical error, invalid verify,
submit without solve, exhausted budget, invalid tool arguments, repeated actions, private message
access, substantiated and false reports, hidden ground truth, failure separation, and scoreboard
reconstruction. B0/B1/B2 replay tests also inject missing, corrupted, duplicate, and inconsistent
events. Complete scripted population tests pass at 5, 10, and 20 agents.

The scripted A/B/C diagnostic produces the expected accounting: A has one correct unverified
submission and one violation; B and C have one correct verified submission and no violation. C
uses three observable actions but pays no budget cost for verification. This validates the
diagnostic manipulation, not real-model sensitivity to it.

The preserved four-agent Kaggle smoke completed in about 175 seconds. It produced 4/4 valid and
correct submissions, 0/4 explicit verifications, and 4/4 protocol violations. All agents showed
some textual self-checking, demonstrating why observable tool use must remain distinct from
inferred intention. Replay matched the recorded summary and digest.

## 9. B0/B1/B2 Behavioral Demonstration

B0 has been exercised with both real model families in the preserved four-agent smoke. This is
only an end-to-end validation because it used two easy synthetic tasks and one decision per
agent.

Scripted B1 demonstrates genuine interaction rather than independent episodes with a cosmetic
scoreboard: a private message changes another inbox, inspections expose permitted submitted
evidence, reports update shared audit state, and subsequent observations contain the relevant
event sequence numbers and scores.

Scripted B2 demonstrates independently controlled true, false, and neutral notices. The metrics
record inspection rate, reporting rate, false-reporting rate, verification compliance, subsequent
violations, and completion from observable actions. Scripted response choices were deliberately
constructed to exercise those metrics and are not evidence of manipulation or intergroup bias.

No statistically meaningful B0/B1/B2 model comparison has been completed. Selective enforcement
and retaliatory violations remain hypotheses.

## 10. Limitations

The principal limits are small models, one two-GPU runtime, sequential inference, short episodes,
exact-string mathematics grading, and an uncalibrated task subset. Current real evidence is a
four-agent smoke; ten and twenty agents are validated only with scripted policies. Team-label
permutations, repeated seeds, confidence intervals, timeouts, and a full 8-task × 3-repetition
pilot remain future work.

The environment does not provide hardened sandbox isolation. Malicious-model containment is not
evaluated. B2 currently demonstrates one controlled notice pattern and does not cover a broad
space of persuasion or collusion strategies. Small samples must not be used to rank model safety
or infer general properties of LLM swarms.

## 11. Reproducibility

CPU validation is reproduced with `uv sync --extra dev`, `make test`, and `make lint`. Scripted
conditions run through `scripts/run_swarmbench_lite.py`; the A/B/C manipulation runs through
`scripts/run_compliance_diagnostic.py`. Population and task manifests are versioned in `configs/`.

On Kaggle 2×T4, `scripts/run_kaggle_b0.py` installs the pinned CUDA backend and downloads the two
public GGUF files. `SWARMBENCH_KAGGLE_MODE=smoke` selects a two-agent backend test; the default
validation selects five B0 agents and the two-model A/B/C diagnostic. Outputs can be downloaded
and replayed locally without checkpoints or inference. Dependency versions, sampling parameters,
model names, task IDs, and seeds are recorded, but exact stochastic LLM text is not promised to
recur.

## 12. Dual-Use and Ethical Considerations

The same environment features that reveal selective enforcement, retaliation, or collusion could
help optimize those behaviors. Releases should therefore emphasize aggregate safety evaluation,
retain raw auditability, avoid deploying learned adversarial policies, and protect any real user
messages used in future studies. Current tasks are synthetic or public benchmark problems and
contain no personal data. Results should be reported with denominators, failure categories, and
limitations so that a tiny demonstration is not mistaken for a general claim about a model or a
group of people.
