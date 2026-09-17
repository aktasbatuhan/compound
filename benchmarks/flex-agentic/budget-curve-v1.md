# Budget-first Flex pilot

Status: repaired offline setup, awaiting target-runtime and paid qualification.
This supersedes the old tier-equivalence configuration for the next pilot.
Historical configurations and results remain unchanged.

## Question and frozen policy

For the same model, tasks and harness, does Flex solve more tasks at a given
agent-dollar allowance? Report the time required alongside the budget curve.
`budget-curve-v1.json` uses DeepSeek V4.1 Flash on Doubleword realtime and Flex,
10 seeded tasks from the previously reference-qualified retail pool, two trials,
and $0.01, $0.03 and $0.08 agent allowances: 120 episodes total.

Each budget gets a fresh environment and one continuous agent episode. Tool use
and self-correction are allowed within it. The hidden grader is used only after
the episode; it never chooses a retry or tells the agent when to stop. This is
not pass-at-k or best-of-k. Budget exhaustion without a graded result counts as
no delivered success. Output caps stay fixed; a request that cannot be reserved
in full is refused. Actual spend can therefore be below the allowance. Record
that unspent amount rather than claiming every arm consumed its full budget.

The simulator stays on its pinned standard route. Its spend is outside the
agent allowance but inside the total experiment cap and cost-per-success.
Report both agent-only and agent-plus-simulator costs. This measures the current
harness's budget policy, not an optimal allocation of inference compute.

## Output cap and what it costs the small budgets

`max_output_tokens` is 8192, which is exactly what the retail harness requests,
so the study control no longer clamps the agent's own request. The previous 4096
cut every reply in half inside the gateway, and a truncated reply is graded the
same as a wrong one, so the cap was buying a measurement error rather than a
saving.

The cap is not free. Output is reserved in full before each call at the fixed
policy, so the reservation per call doubles:

| Output cap | Reserved per call, standard | Reserved per call, flex |
|---|---:|---:|
| 4096 | $0.00246 | $0.00197 |
| 8192 | $0.00492 | $0.00393 |

Against a $0.01 allowance the output side alone now reserves 49% (standard) or
39% (flex) before any input is counted, and input is reserved at the 1h
cache-write multiplier of 2.0. The $0.01 cell can therefore afford very few
calls, possibly one. Read a low-budget cell as a statement about this harness's
reservation policy at that allowance, not as evidence that the model cannot do
the task. Report the budget stops for those cells rather than folding them into
a success rate.

## Money and time bounds

Planned agent ceilings sum to $4.80; simulator ceilings sum to $3.60. A $10
inference guard leaves at most $1.60 for probes and unused reserves. These are
ceilings, not predicted bills. Role caps remain $6 agent / $4 auxiliary; use the
same output directory and spend ledger for qualification and execution.
GCP compute is separate and must fit the user's remaining overall allowance.
No VM or paid calls are launched by this change.

The one-hour episode ceiling is an operational stop, not the primary endpoint.
At one lane the theoretical 120-episode worst case is 120 hours; do not promise
a completion time from token-price arithmetic. Use the smoke measurements to
estimate actual runtime before launching the pilot. Report success by 60, 300,
900 and 3600 seconds, plus completion times and provider failures. Timeouts
must not be reported as completed episodes with a one-hour latency.

## Qualification and launch sequence

1. On GCP, freeze the Python environment, harness revisions and task data; replay
   reference solutions for the selected tasks and verify failure cases too.
   The historical reference qualification does not prove the new runtime works.
2. Recheck the rate snapshot. Doubleword V4.1 rates are explicitly tied to its
   model ID. The simulator's reservation rates use the highest listed scheduled
   OpenRouter rate; its actual cost comes from the usage receipt.
3. Run the cache-enabled tool probes using the same spec and output directory.
   Both requests carry markers on opt-in APIs. Qualify standard, Flex and the
   simulator independently. Missing usage or tier evidence is not a pass.
4. Run one complete task pair as the smoke test, inspect official grader output,
   per-call receipts, logs and costs, then continue the same sealed run.

Offline plan:

```sh
uv run python -m compound.agentic_study plan --spec benchmarks/flex-agentic/budget-curve-v1.json --out /tmp/budget-curve-plan.json
```

Prepare an isolated target runtime first. The base Compound environment does
not include tau2's dependencies (the local check currently stops at missing
`toml`); do not run this with a bare `uv run` environment. Install the pinned
checkout, freeze the resolved packages, and keep this environment unchanged:

```sh
uv venv --python 3.12 .compound/venvs/flex-retail
uv pip install --python .compound/venvs/flex-retail/bin/python -e . -e .compound/sources/tau2-bench-verified
mkdir -p artifacts/budget-curve-v1
uv pip freeze --python .compound/venvs/flex-retail/bin/python > artifacts/budget-curve-v1/runtime-requirements.txt
```

Reference qualification on the target runtime (no inference):

```sh
PYTHONPATH=src:.compound/sources/tau2-bench-verified/src TAU2_DATA_DIR=.compound/sources/tau2-bench-verified/data .compound/venvs/flex-retail/bin/python scripts/qualify_agentic_study.py retail --spec benchmarks/flex-agentic/budget-curve-v1.json --out artifacts/budget-curve-v1/reference-qualification.json
```

Paid commands, for the qualification stage after GCP preparation:

```sh
.compound/venvs/flex-retail/bin/python -m compound.agentic_run probe --spec benchmarks/flex-agentic/budget-curve-v1.json --out artifacts/budget-curve-v1 --inference-limit 10 --go
.compound/venvs/flex-retail/bin/python -m compound.agentic_run run --spec benchmarks/flex-agentic/budget-curve-v1.json --out artifacts/budget-curve-v1 --inference-limit 10 --count 2 --go
.compound/venvs/flex-retail/bin/python -m compound.agentic_run run --spec benchmarks/flex-agentic/budget-curve-v1.json --out artifacts/budget-curve-v1 --inference-limit 10 --count 120 --go
```

**Current live blocker:** Doubleword Chat Completions has previously omitted the
tier echo. Such calls are now `unverified`, not `billing_meter`. The scaled runner
will stop at qualification unless actual tier echoes are present. If still
absent, a tested verifier for provider-side, request-linked tier receipts is
needed. Do not bypass this gate using a price-derived tier or a hand-set flag.

## Reporting

Report each suite, route, tier and budget separately. The headline curve is
delivered task success versus allowance, with actual spend, sample counts and
incomplete cells visible. Compare tiers within a budget; do not pool budgets.
Cost per success includes unsuccessful attempts and simulator calls; unresolved
charges prevent a complete cost estimate. Infrastructure errors invalidate a
cell rather than making the model look worse. Preserve them for audit.

This small pilot is descriptive. Repeated trials share tasks, so naive
episode-level binomial intervals overstate precision. Use task-clustered
uncertainty for any later inferential report; do not make equivalence claims
from this budget curve or from a degenerate bootstrap.

## Separate Fireworks replication

The [Fireworks article](https://fireworks.ai/blog/DeepSeek-V4.1-Flash-Astra)
is a reference result, not a complete executable protocol. Before calling an
experiment a replication, obtain the task IDs, harness commit and configuration,
prompts/tools, reasoning settings, stopping and retry policies, grader and
runtime images, cache policy, and token/cost accounting boundaries.

Keep a same-model Fireworks / Doubleword realtime / Doubleword Flex comparison
separate from this retail budget pilot. If those details remain unavailable,
label the eventual run an independent comparison under our published harness.
Do not substitute our retail setup and call it a replication of their DeepSWE
result, or extrapolate a paid-run budget from their aggregate token count.
