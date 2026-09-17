# Claude execution handoff

Updated 2026-09-17. This document prepares execution; it does not launch a job.
Claude owns setup and execution. Codex reviews the frozen design, exceptions and
result artifacts. Preserve the user's existing work and all historical evidence.

## Start here

1. Read this file, `flex-agentic/budget-curve-v1.md` and its matching JSON.
2. Inspect `git status`. The repaired implementation is currently local work;
   do not assume a fresh remote clone contains it. Transfer a reviewed commit or
   explicit source bundle to GCP. Record its commit or SHA-256 inventory. Include
   the new `agentic_safety.py`, spec and qualification scripts. Do not transfer
   unrelated working-tree changes or credentials in the bundle.
3. Run the relevant offline checks below. Do not spend inference credit to debug
   imports, broken graders, or missing datasets.
4. Use the isolated runtime and commands in the canonical protocol. Keep the
   Python environment, source files, spec, execution mode and output directory
   unchanged after sealing. Keys are loaded from a local `keys.json` by the
   runner; provision it securely on the VM, restrict access, never print it or
   include it in source/evidence archives.

## Flex pilot: exact scope

- Model: `deepseek-ai/DeepSeek-V4.1-Flash`, Doubleword realtime versus Flex.
- Workload: ten frozen retail tasks, two trials, three agent allowances
  ($0.01/$0.03/$0.08), paired within task/trial/budget. 120 planned episodes.
- One fresh continuous episode per allowance. No repeated independent attempts,
  hidden-grader feedback, or oracle selection of a winning answer. A retry/search
  policy would be a separate protocol amendment before new paid data.
- Keep the standard-tier simulator fixed. Agent allowance excludes simulator
  spend; total experiment spend and total cost per success include it.
- $10 is the inference ceiling for probes, smoke and continuation together,
  not $10 per stage. Agent and auxiliary role caps are $6 and $4. Do not raise
  caps or create a fresh spend ledger to get around an exhausted reserve.
- GCP compute is additional. Account for it against the user's remaining total
  authorization; this document grants no new compute allowance. Record VM type,
  region, uptime and actual/estimated compute charges separately. Shut down the
  experiment VM when work is finished, after copying evidence safely.

## Required order and stop conditions

1. **Offline preparation.** Install the pinned benchmark checkout and dependencies
   in the isolated runtime. Verify data hashes and clean harness revision. Run
   reference qualification with the selected spec and save
   `reference-qualification.json` in the intended run directory. It must cover
   exactly the selected tasks and match the target Python environment. Reference
   tool replay alone is not a successful end-to-end agent/grader smoke test.
2. **Paid API qualification.** Use the `probe` command. It sends two requests per
   agent tier and two for the simulator. Re-running `probe` spends again. Inspect
   tool correctness, cache-read usage, costs and actual tier evidence separately
   for each arm. Never remove markers or switch to Responses to obtain a tier
   echo at the expense of prompt caching.
3. **Resolve the known blocker.** Cached Doubleword Chat Completions previously
   omitted tier echoes. The current runner accepts response echoes only; there
   is no implemented billing-receipt verifier. A price-derived cost, requested
   tier, dashboard aggregate or hand-edited `tier_confirmed` field is not a fix.
   If echoes remain absent, stop and return the probe evidence for a design/code
   decision. Do not repeatedly probe or bypass the gate. A provider receipt
   verifier would need implementation and tests before another scaled run.
4. **Paired smoke.** Run `--count 2` only after qualification passes. Inspect both
   official grader artifacts, traces, errors, budget stops and costs. Record
   observed wall time and projected remaining runtime. Missing checkpoints,
   unexplained zero-call episodes or missing charges are not successful smoke.
5. **Continuation.** Continue in the same directory only after the smoke is
   sound. Keep one lane and the frozen pair ordering. Missing episodes stay
   pending. An interrupted paid episode without an outcome requires evidence
   reconciliation, not an automatic retry or deletion of its spend entry.

A 402, infrastructure failure, contradictory tier, invalid provider evidence,
or failed cache gate means stop and diagnose. Report the exact error and artifact
path. Do not adapt task selection, scoring, model, budget or output caps after
seeing results and silently pool the changed run with the original.

## Offline checks and report

Run from the repository root, before freezing the target benchmark runtime:

```sh
uv run pytest tests/test_agentic_study.py tests/test_agentic_gateway.py tests/test_agentic_run.py tests/test_agentic_worker.py tests/test_agentic_safety.py tests/test_cache_policy.py tests/test_qualify_agentic_study.py tests/test_tier_equivalence.py -q
```

Use the canonical runbook for environment preparation, reference replay, probes
and paid commands. Afterwards, generate the descriptive budget report offline:

```sh
.compound/venvs/flex-retail/bin/python -m compound.agentic_study report --spec benchmarks/flex-agentic/budget-curve-v1.json --outcomes artifacts/budget-curve-v1/outcomes.jsonl --out artifacts/budget-curve-v1/report-final.json
```

The report writer refuses to overwrite an existing output. Use a new report
filename when recomputing; preserve earlier artifacts. It separates each budget
and reports agent and auxiliary costs separately. For total inference cost per
success, combine those costs only when both have complete coverage; include
failed attempts. Account for qualification overhead separately in the complete
experiment bill. Do not present the cost subtotals as a reconciled total when
charges are unresolved. Repeated-trial intervals need task clustering; this
pilot does not establish equivalence or non-inferiority.

## Evidence package for Codex review

Return one directory or archive, with a short `REVIEW.md` index containing:

- Exact code identity, launch commands, GCP region/machine, start/end times,
  Python version and frozen packages; any deviations and their timing.
- `spec.json`, `plan.json`, `run-manifest.json`, rate snapshot and reference
  qualification. Include the dataset/harness hashes and revisions, not secrets.
- Complete `calls.jsonl`, `spend.json`, `probes.json`, `outcomes.jsonl`,
  `budget-stops.jsonl` if present, and execution amendments if present.
- All `attempts/` artifacts (official grader outputs, trajectories, outcomes,
  `failure.json`) and `worker-logs/`. Include interrupted attempts too.
- Descriptive report, completion counts by tier/budget, unresolved charges,
  tier-unverified calls, cache-hit evidence, provider failures and infrastructure
  failures. Distinguish reported from token/rate-derived cost.
- Total inference spend including probes, held/unsettled reservations and GCP
  cost. List which conclusions are supported, inconclusive or blocked.

Do not send thousands of log lines in chat. Send the artifact location, a small
results table, total spend, and a concise list of anomalies. Codex will review
pairing/coverage, graders, cache/tier provenance, budget accounting, censoring
and whether the claims follow from the evidence. No automatic publication.

## Fireworks experiment: preparation only

Goal: compare the same DeepSeek V4.1 Flash workload across Fireworks, Doubleword
realtime and Doubleword Flex, measuring quality, cost and time together.
The [published article](https://fireworks.ai/blog/DeepSeek-V4.1-Flash-Astra)
is context, not an executable configuration supplied by this repository.

Before execution, document task IDs, harness commit, prompts/tools, reasoning
settings, sampling controls, retries/termination, grader, runtime images, cache
policy and cost boundaries. If the original setup is unavailable, propose an
independent comparison with our pinned harness and label it accordingly. Do not
call SWE-bench/mini-SWE-agent a DeepSWE replication. Do not silently replace the
quality comparison with token-price arithmetic.

There is currently no frozen runnable Fireworks spec or approved launch budget
in this handoff. Return a concrete proposed spec, adapter gaps and a conservative
smoke-first budget for review. Do not add Astra or a batch arm by assumption.

## Prompt to start a Claude session

> Read benchmarks/claude-experiment-handoff.md and the linked canonical Flex
> protocol. Prepare the GCP execution setup and complete the offline checks.
> Use my existing spending authorization only; do not increase it. Execute the
> paid qualification and paired smoke only when their documented prerequisites
> are satisfied. Stop on missing tier evidence or infrastructure problems,
> preserve the ledger and return the evidence. Continue the pilot only after a
> sound smoke, without changing its frozen design. Keep Fireworks at preparation
> stage pending a reviewed spec. Return the evidence package and REVIEW.md for
> Codex to review; do not publish findings.
