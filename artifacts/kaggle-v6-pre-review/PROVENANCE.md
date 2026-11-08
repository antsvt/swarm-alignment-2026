# Kaggle v6 Pre-Review Validation

This directory preserves the unmodified machine-readable outputs of Kaggle kernel version 6,
completed on 2026-11-08 in approximately 266 seconds. The run used the environment implementation
immediately before the ground-truth visibility and manifest-source review fixes.

The artifact is retained to make the validation history auditable. It is not the reference output
for the current protocol.

Known issues in this run:

- the B0 runner embedded five task statements instead of loading them from the frozen manifest;
- the precalculus statement differed from the manifest statement while retaining the same ID and
  answer;
- diagnostic `config.json` files incorrectly label the synthetic equation as MATH-500;
- the verification prompt induced the literal placeholder `checked final answer`, invalidating
  diagnostic mathematical accuracy;
- one response omitted JSON and one reached the generation limit, so B0 completed only 3/5 agents;
- action results used the pre-isolation schema that included scorer fields in actor history.

The current code fixes these issues or records the remaining inference failure explicitly. All
four event logs in this directory pass local replay through the narrowly scoped legacy adapter.
