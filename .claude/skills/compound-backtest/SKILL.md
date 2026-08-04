---
name: compound-backtest
description: Backtest a model or provider switch on the user's own traffic with Compound. Use when the user asks which model/provider to use, whether a cheaper model is good enough, or wants a regression gate before switching. Runs the full loop - import traces, curate cases, dry-run, paid gate - with hard budget caps.
---

# Compound backtest

Compound turns production traces into a switch verdict: it replays candidate
models and providers against graded history and answers "can I switch without
losing quality?" with a statistical gate, not a vibe. You are driving a
measurement instrument; your job is to keep it honest and keep it cheap.

## Ground rules (read first)

1. **Money**: nothing spends without `--paid`. NEVER add `--paid` to any
   command unless the user has explicitly approved that specific run and its
   cap in this conversation. Always dry-run first and show the user the cost
   estimate. Every paid run needs `--cap <usd>`; pick a cap just above the
   estimator's ceiling, never a round guess.
2. **Read the surface, don't guess flags**: before the first command of a
   session, run `bunx compound --help` and the subcommand's `--help`. Compound
   evolves; help output wins over this file.
3. **The sealed partition is sealed**: `decision_test` cases exist so the final
   verdict is out-of-sample. Never run exploratory work against them; the gate
   opens them itself and requires a `--reason`.
4. **Report verdicts verbatim**: `MEETS GATE`, `FAILS`, `INSUFFICIENT_DATA`,
   judge abstentions. If the verdict is `INSUFFICIENT_DATA`, say so - do not
   round it up to a recommendation.

## The loop

Work in the user's repo; Compound state lives in a local SQLite db (see
`compound.yaml`, or `compound init` if absent).

### 1. Get traces in
- Langfuse export or portable JSON: `bunx compound import <file>`
- Show the import report (accepted / rejected / duplicate counts). Redaction
  runs before persistence; if the user worries about secrets, point at the
  redaction report rather than reassuring blindly.

### 2. Curate cases
- `bunx compound curate <task-key>` turns traces into provenance-typed cases
  and assigns immutable partitions (train / validation / calibration /
  decision_test).
- Surface how many cases landed per partition. If decision_test is thin
  (< ~20), warn that the gate may return INSUFFICIENT_DATA - that is the
  honest outcome, not a failure to avoid.

### 3. Dry-run the candidate (free)
- `bunx compound experiment <task-key> --candidate <model>` without `--paid`
  makes zero provider calls and prints the cost estimate.
- Present the estimate to the user and ask whether to proceed paid, and with
  what cap.

### 4. Paid run (only after explicit user approval)
- Same command plus `--paid --cap <usd>`. Provider/host pinning is first-class:
  `--openrouter-provider <slug>` isolates a specific serving host (fallbacks
  disabled, served host recorded).
- Completions are content-addressed and cached: re-running a comparison is $0.
  Tell the user this - re-deciding later is free.

### 5. Gate the switch
- `bunx compound gate <task-key> --candidate <model> --reference <model>
  --reason "<why>"` decides over cached runs on the sealed set with a
  pre-declared non-inferiority rule. `--max` is refused on paid gates by
  design; do not try to work around it.

### 6. Compare routes
- `bunx compound view compare <task-key>` renders quality / latency / TPS /
  cost per (model x provider) route from recorded telemetry - the table a
  switch decision is actually made from.

## When things look wrong

- Estimator ceiling seems high: check case counts and prompt sizes before
  blaming the tool; show the math to the user.
- A judge-graded task abstains: run `bunx compound judge calibrate <task-key>`
  and explain that an uncalibrated judge refusing to emit a verdict is the
  feature working.
- Budget errors: the hard limit in `compound.yaml` is a wall, not a suggestion.
  Ask the user before raising it; never edit it silently.
