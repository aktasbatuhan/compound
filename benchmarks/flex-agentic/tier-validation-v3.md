# Held-out Realtime versus Async validation

Frozen before new paid calls, 2026-09-21. Supersedes v2 for new performance claims;
retain v2 evidence, including its reservation-dominated $0.01 cells.

Question: on identical model/harness/task settings, what changes in delivered
success, cost per successful task and time when requesting Async instead of
Realtime? This is not a budget scaling experiment. No guaranteed headline or
statistical-significance stopping rule.

## Sample and controls

30 seeded tasks selected from 78 held-out eligible tasks, excluding all ten v2
pilot tasks. Eligibility uses only reference actions and the official DB and
communication graders: gold passes, empty DB trajectory fails. It does not use
model outcomes. `scripts/prepare_tier_validation.py` reproduces the selection and
writes pool-audit.json. This restricts inference to these mutating retail tasks,
not the full benchmark or all agentic applications.

Three fresh trials per task and tier: 90 adjacent randomized pairs, 180 episodes.
Frozen seed controls pair order and harness trials. Temperature zero is explicitly
sent to both providers; this does not guarantee deterministic model responses.
Same DeepSeek V4.1 Flash, reasoning medium, tools, 8192 output cap, 30 steps,
fixed simulator, explicit 1h cache markers and one lane. Provider cache reuse is
permitted as in production; this is not an isolated cold-cache comparison.
Record timestamps and analyze pair order/time drift as limitations; this is one
campaign, not a balanced multi-day load study. Served tier remains unverified
when no provider echo is returned.

Agent $0.10 and auxiliary $0.06 per episode are safety ceilings, not target spend.
Retain conservative reservations. Halt the study on any budget-exhausted outcome,
infrastructure error, contradictory tier, account error or failed cache gate.
No automatic retries. Preserve all attempts, errors and held reservations.

## Spending

New run has a hard $9.40 inference cap, agent role $7 and auxiliary role $2.40.
All qualification/smoke/continuation share one directory and ledger. Prior runs
spent/reserved approximately $0.509103 including pricing and failed qualification;
$9.40 + prior allowance use remains below the original $10 authorization.
Per-episode ceilings are not an assurance all episodes fit the global guard.
Stop with pending work rather than increase spending. GCP is separate within the
existing overall authorization; use a bounded VM session and automatic shutdown.

## Gates before execution

1. Pin code/harness and freeze packages on GCP. Reproduce pool selection and require
   byte-identical spec. Re-run reference qualification in the target environment.
2. `qualify_retail_admission.py` exercises real harness opening messages/tools
   through the actual gateway and reservation guard for every selected task and
   both tiers, using a deliberately long synthetic simulator opening. Network is
   intercepted before transport; no credential or paid call is needed. It must
   record all 120 task/tier/role admissions. This proves opening affordability,
   not whole-episode affordability. The runtime gate rejects missing evidence.
3. Paid cache/tool probes for both tiers and simulator, then one full paired
   task smoke. Review official outputs and call/charge coverage. Outcome success
   is not a qualification requirement; correct grading and accounting are.
4. Continue remaining episodes with no design adaptation. Stop starting episodes
   before the VM cutoff, allowing a full episode to finish and save evidence.

## Frozen analysis

Primary descriptive estimates: paired success difference (Async minus Realtime),
and ratio of total agent-plus-simulator inference cost per successful task.
Include unsuccessful attempts in costs. Missing charges make a reconciled cost
ratio unavailable, never zero. Separate agent-only estimates and qualification
cost; report compute separately.

Report counts of planned, graded, budget-stopped and infrastructure-failed
attempts. A budget stop invalidates the claim that allowances were nonbinding.
Do not silently exclude it, replace it or label it a model-quality failure.
For infrastructure-missing results, report observed successes over planned plus
complete-pair sensitivity; no complete-primary-result claim.

Use 10,000 bootstrap replicates with fixed seed 20260921, resampling task IDs
and retaining both tiers and all three trials together. Report percentile 95%
intervals on success difference and cost-per-success ratio where identifiable.
Zero-success resamples yield undefined cost ratios and must be disclosed.
Repeated trials are not 90 independent tasks. No equivalence/non-inferiority or
statistical-significance claim based on an interval including zero/one.

Report success by 60, 300, 900, 3600 seconds over planned attempts, and graded-only
median/p90 task time with its denominator. Do not treat budget stops or timeouts
as fast completed tasks. Do not pool v2 outcomes into this held-out run. A second
study chosen after seeing results must be separately labeled.

## Commands on the pinned runtime

```sh
export PYTHONPATH=src:.compound/sources/tau2-bench-verified/src
export TAU2_DATA_DIR=.compound/sources/tau2-bench-verified/data
export LITELLM_LOCAL_MODEL_COST_MAP=True
mkdir -p artifacts/tier-validation-v3
.compound/venvs/flex-retail/bin/python scripts/prepare_tier_validation.py
.compound/venvs/flex-retail/bin/python scripts/qualify_agentic_study.py retail --spec benchmarks/flex-agentic/tier-validation-v3.json --out artifacts/tier-validation-v3/reference-qualification.json
.compound/venvs/flex-retail/bin/python scripts/qualify_retail_admission.py --spec benchmarks/flex-agentic/tier-validation-v3.json --out artifacts/tier-validation-v3/admission-qualification.json
.compound/venvs/flex-retail/bin/python -m compound.agentic_run probe --spec benchmarks/flex-agentic/tier-validation-v3.json --out artifacts/tier-validation-v3 --inference-limit 9.40 --go
.compound/venvs/flex-retail/bin/python -m compound.agentic_run run --spec benchmarks/flex-agentic/tier-validation-v3.json --out artifacts/tier-validation-v3 --inference-limit 9.40 --count 2 --go
```

Only after smoke review, use the same run command with `--count 180` and an explicit
`--stop-at` UTC deadline. Controller skips recorded episodes. Preserve the
finished directory, receipts and resume notes for independent review.
