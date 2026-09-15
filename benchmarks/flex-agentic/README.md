# Flex agentic pilot

Four models, five provider/model routes, two service tiers, and 15 fixed tasks:
150 planned episodes, one trial per task and route/tier. This is an integration
pilot, not a sufficiently powered model ranking. Paid execution runs on an isolated
GCP VM; inspect the saved outcomes for completion rather than treating the plan
as a completed result.

`pilot.json` freezes source revisions, dataset hashes, task IDs, selection rules,
and execution controls. `catalog-2026-09-09.json` is a read-only endpoint
snapshot, not evidence that a request was served by a particular tier. OpenRouter
prices are dollars per token; do not apply the `discount` field a second time.

## Workloads and success

- Retail uses Amazon's corrected tau2-bench checkout. Selected reference actions
  must execute without errors and change the database. Task 54 failed this check
  and was replaced by 16 before any inference. Preserve the official reward and
  its components; task success means reward equals 1, not HTTP 200. Keep simulator
  and judge inference separate from agent cost and latency.
- Coding uses SWE-bench Verified with the pinned mini-SWE-agent harness and
  official SWE-bench grader. A submitted patch is not a success until the grader
  reports it resolved. Container startup, reference patch validation, and grader
  execution are qualified on GCP before paid coding episodes. The current grader
  uses additional official dataset metadata, fetched with
  `scripts/prepare_swe_grader.py`. It verifies identical task text, patches, base
  commits and test sets against the frozen selection before adding image and
  grading-script fields.
- Finance uses FinBalance's existing `tool_native_full_tool_agent` mode and
  original component scorer. The declared strict success composite requires a
  parsed answer, exact cited entries and balance sheet for clean cases, or the
  correct inconsistency flag/code and empty reconciliation for inconsistent
  cases. Preserve component metrics as well; this composite is our study's task
  success definition, not FinBalance's headline balance-sheet metric.

## Offline preparation and checks

Source checkouts live under `.compound/sources/`; they are not bundled into the
public repository. Preserve the original tau checkout. The Verified fork uses
`RunConfig`, while Compound's existing tau adapter uses `TextRunConfig`; they
cannot be swapped implicitly. Verified also needs its own compatible runtime
dependencies before launching the user simulator.

```sh
python -m compound.agentic_study plan --out artifacts/flex-pilot-plan.json
PYTHONPATH=src:.compound/sources/finbalance python scripts/qualify_agentic_study.py finance
PYTHONPATH=.compound/sources/tau2-bench-verified/src TAU2_DATA_DIR=.compound/sources/tau2-bench-verified/data python scripts/qualify_agentic_study.py retail
```

Run from the repository root in an environment with that suite's dependencies.
The qualification script makes no inference calls. Finance replays reference
answers through the parser/scorer, rejects an incorrect answer, checks hidden
answer invariance for prompts and the ledger tool, and exercises the native
tool loop using a fake client. Retail replays reference tools without an LLM.
These are fixture checks, not measured model successes.

## Outcome contract

`python -m compound.agentic_study report --outcomes outcomes.jsonl --out summary.json`
is offline. Every outcome carries `episode_id` and `spec_sha256` from the plan,
`status`, `success`, `duration_s`, and grader provenance in `grader` for graded
episodes. Status is `graded`, `provider_error`, `timeout`, or `infrastructure_error`.
Only graded episodes have a boolean success; other statuses use null. Record
`agent_cost_usd`, `auxiliary_cost_usd`, and `sandbox_cost_usd` separately; unknown
cost is null, not zero. Store raw grader output and trajectories locally.

For a complete group, success rate is successful episodes / all planned episodes,
including provider errors and timeouts. Also show success among graded episodes
and grading coverage so service failures and task mistakes remain distinguishable.
Pending episodes and infrastructure errors make the group incomplete; they do
not silently become model failures. Deadline success uses the same denominator.
Cost per success includes failed-episode costs and is null when cost coverage is
incomplete or no tasks succeeded. Wilson intervals are descriptive for this
single-trial pilot; a repeated study needs task-clustered uncertainty estimates.

## Paid execution on GCP

The pilot uses one `e2-standard-8` VM in `us-central1-a`, with a 200 GB standard
persistent disk and a four-hour automatic deletion limit. The $15 authorization
is split into a $12 inference guard (including simulators and route checks) and
$3 reserved for GCP. Results must be copied locally before deleting the VM.

The VM needs Python 3.12, Docker, the pinned source checkouts and dependencies,
and a mode-600 `keys.json` containing the OpenRouter and Doubleword credentials.
Only the parent process reads real keys. Workers call a loopback gateway; coding
containers receive neither host keys nor a GCP service account.

```sh
python scripts/prepare_swe_grader.py
python -m compound.agentic_run probe --go
python -m compound.agentic_run oracle --count 5
python -m compound.agentic_run run --suite finance --task COV_PRO_M1_0075 --count 10 --go
python -m compound.agentic_run run --count 150 --parallel-routes --go
```

Without `--go`, probe/run makes no inference calls. A process lock prevents two
controllers from independently spending the same allowance. The optional
`--parallel-routes` mode overlaps independent routes, with one active agent episode
per route and one coding episode globally. Simulator calls may overlap. It records
an execution amendment before dispatch and labels each resulting outcome; the
initial qualification episodes were sequential. Each request reserves
conservative maximum cost before dispatch; ambiguous failures retain their full
reservation. There are no automatic retries or fallback routes. A budget stop
leaves the remaining episodes pending. Infrastructure failures stop execution for
repair and are not silently counted as model mistakes.

OpenRouter requests pin the exact provider endpoint and service tier. Doubleword
uses the blocking Responses API. Its original requests included five-minute
cache markers, but those are ineffective: Doubleword's Responses endpoint does
not support prompt caching. The corrected gateway omits those markers. Actual
service-tier echoes are recorded and checked. Provider reasoning signatures are
preserved across tool turns, including harnesses whose message schemas discard
those opaque fields. Reasoning effort is `medium` within every pair; this does not
imply equal reasoning compute across model families. Temperature is left at each
provider's default, consistently within pairs.

`duration_s` measures the finance tool loop, retail simulation (including the user
simulator), or coding agent loop. Coding setup and post-submission grading have
separate durations. This blocking experiment does not measure time to first
token. API-call durations, usage, cache tokens, tier evidence, request hashes,
and costs are recorded in `calls.jsonl`; raw benchmark results live beside each
outcome. OpenRouter costs are provider-reported. Doubleword costs use observed
tokens and declared input/output rates. The original write-price estimate is
corrected offline, removing the inapplicable surcharge while preserving the
original ledgers. These estimates are not reconciled invoices. Caching support
is documented at https://docs.doubleword.ai/inference-api/prompt-caching.

The upstream mini-SWE-agent trajectory's own `model_stats` cost is not used:
that harness sees a generic wire-model alias. Cost analysis uses the central
gateway ledger, which knows the actual route, tier, and usage.

The first ten finance runs and first retail run predate the timing-field fix:
their `outcome.json` values include worker overhead. Their upstream `official.json`
artifacts retain the task-loop durations, which should be used in analysis.

`spend.json` is the restart-persistent charged-or-reserved ledger. Probe costs and
failed-call reservations belong in the total even though probes are excluded from
task-success results. GCP costs are tracked separately at study level; don't
invent per-task compute bills. A single trial on five tasks per suite is a pilot,
with descriptive uncertainty only, not a robust model or provider ranking.

After copying results back, run `python scripts/summarize_flex_pilot.py`. It
normalizes the early timing fields from upstream artifacts and writes
`analysis.json`. Tier comparisons use only task IDs attempted in both tiers,
exclude infrastructure errors, and retain provider errors and timeouts. Each
comparison lists its task IDs and execution modes; different routes may have
different coverage in a budget-limited run. Unmatched tasks stay in the full
coverage report rather than being silently dropped from the study.

Historical accounting corrections are explicit in `analysis.json`: an OpenRouter
HTTP-200 envelope with a nested upstream error is an API failure; an official
SWE-bench `empty_patch_ids` result is an unsuccessful task. Raw outcomes remain
untouched, and these corrections do not require repeating paid attempts.

For this authorized GCP pilot, `python scripts/supervise_flex_gcp.py --go`
checkpoints completed results locally every ten minutes, permits one VM restart
to resume pending episodes, and deletes the dedicated VM and boot disk after
the final checkpoint. It targets only the original pilot instance. The $15
allocation is split into a $12 inference guard and $3 for infrastructure; two
boots are limited to four hours each. Completed episodes are never rerun.

Confirmed OpenRouter input-context validation rejections are reconciled to zero
under its documented failed-request billing policy, with the original request
hash and policy source retained in `reconciliations.jsonl`. This reconciliation
runs only while the controller is stopped. Ambiguous failures keep their
reservations, and the rejected episode remains unsuccessful in the analysis.

The first allocation stopped at 132 recorded episodes. On September 9 the user
authorized completing the remainder on GCP. The continuation has a separate
$12 inference ceiling and a fresh four-hour auto-deleting VM. It carries forward
131 completed outcomes, runs the 18 unstarted episodes, and reruns one episode
interrupted by the harness's own budget guard. That interruption is not a model
failure. Its original files and inference charges remain in the first checkpoint.

`scripts/supervise_flex_finish.py --go` checkpoints the continuation and removes
its VM and disk after the controller stops. `scripts/report_flex_complete.py`
merges the two preserved checkpoints into `artifacts/flex-agentic-complete/`,
retains the interrupted attempt as study overhead, and generates `report.md`,
`metrics.json`, and `analysis.json`. Provider failures and incorrect model outputs
are not retried to improve the scores.

The original paid processes continue with their conservative spend guards;
offline Doubleword corrections do not mutate a live guard or trigger additional
calls. The merged `cost-corrections.json` matches each adjustment to its request
hash, phase, original estimate, and pricing-policy source. Cache results for
Doubleword are marked unsupported, not interpreted as a zero-percent hit rate.

## Fixed-budget engineering trial, September 10

`budget-ten-dollar-token-informed.json` defines a separate 40-attempt retail
trial: two previously seen tasks, two repetitions, five routes and two tiers.
GPT-6 Astra replaces Sol as an evaluated model; GLM-5.3-Flash replaces GLM.
The fixed user simulator remains Sol on standard service. Astra receives a
$0.40 agent allowance per attempt; the other routes receive $0.10. Allowances
are equal within a model's tier pair, not across models. The simulator has its
own episode and global limits. One active attempt per route, randomized adjacent
tier pairs, no retries, and a 900-second attempt ceiling are held fixed.

The $10 authorization includes smoke checks and cloud costs: all inference
phases share $8, with $2 reserved for GCP. Ten tool-call smoke checks passed.
The initial `budget-ten-dollar.json` diagnostic used full request bytes as the
input-token bound. It stopped some attempts far below their dollar allowance;
one simulator transport failure then stopped dispatch after 29 recorded attempts.
Those outcomes stay in `main/` as diagnostics, and their costs remain included.
They are not pooled with the corrected experiment in `token-informed/`.

The corrected input bound uses prior reported input tokens plus newly changed
request bytes and a 4096-token framing allowance, with a full-byte fallback.
Output caps shrink to fit the remaining allowance. This is conservative budget
admission, not an exact provider-token budget frontier. The corrected phase's
$6.49 ceiling is the remaining shared allowance, not a new authorization.
Its spec, plan, source hashes and executed source are saved before paid calls.

`scripts/supervise_flex_budget.py` checkpoints this experiment and deletes its
dedicated VM and boot disk on completion. `scripts/report_flex_budget.py` rebuilds
the deadline table, repeated-success counts, cost accounting and reservation
audit from local artifacts. Success at 60, 300 and 900 seconds uses all planned
attempts as its denominator; pending results and harness failures remain visible.
This small trial tests the mechanism and cannot establish a reliable ranking.
