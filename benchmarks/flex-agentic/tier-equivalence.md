# Does the async tier change answer quality?

The pilot and its replication measured five routes on three tasks. They settled
the cost and latency questions and left the quality question open. This protocol
is the scaled run that closes it, for one provider's tier pair.

Revision 2, 2026-09-15, after an independent methods review. The changes that
review forced are listed at the end, including two that would have invalidated
the run and one that would have crashed it.

## The claim, stated exactly

Moving DeepSeek V4 Flash from Doubleword's realtime tier to its flex tier
changes task success by no more than 7.5 percentage points, on the qualified
tau2-bench retail pool, under this simulator and these workload limits.

That is an equivalence claim, so a nonsignificant difference test does not
support it. The analysis is two one-sided tests against a margin fixed before
collection, and it can return equivalent, different, or inconclusive. It is not
a claim that flex is free: 7.5 points against a base rate near 0.30 is a 25%
relative reduction, which is a tolerance, not "no change". Anyone quoting this
result must quote the margin with it.

## Why the existing runs cannot answer it

The same frozen spec was executed twice with matching episode ids. Of the 35
episodes graded cleanly in both runs, 8 flipped, 4 in each direction. Within a
single run, repeated trials of one task disagreed at a similar rate, 7 of 39.
Per-episode disagreement is therefore around 20%, which is a disagreement rate
and not a variance estimate, but it is enough to show that a one or two pass gap
in a six attempt cell carries no information. The pilot's per-route pass counts
are not reproducible and must not be published as a ranking.

Cost and median duration did reproduce. Those findings stand and are not
re-measured here.

## Task pool

Offline gold replay over all 114 retail tasks, no inference, recorded in
`retail-qualified-2026-09-14.json`:

| Outcome | Tasks |
|---|---:|
| Every reference action executes and the database changes | 88 |
| Reference action raises (missing product, missing user, tool signature) | 18 |
| Replay is clean but the database never changes | 8 |

All 88 are used, which removes any per-task choice on our part. It does not make
the pool unselected: this is the population of reference-replay-qualified,
database-mutating retail tasks, and the result generalises to that population
and no wider. The 8 non-mutating tasks are excluded because their reward rests
entirely on the communication check, a different construct from the database
outcome the other 88 turn on. They are not excluded for being noisy: in this
pinned checkout `evaluator_communicate.py` is a lowercased substring match with
commas stripped, and makes no model call.

## Design

One route, DeepSeek V4 Flash on Doubleword, realtime against flex. Every task
runs 7 trials on each tier, 1232 episodes, 616 per arm, balanced by
construction rather than by truncation. The two tiers of a task and trial are
adjacent in the shuffled plan and execute back to back in one lane.

The user simulator is `deepseek/deepseek-v4.1-flash`, fixed and identical across
both arms. It replaces the pilot's GPT-5.6 Sol because the simulator was 63% of
per-attempt cost and buying power was the binding constraint. Two consequences
are declared here rather than discovered later: absolute pass rates are not
comparable with the pilot, and the simulator now shares a model family with the
agent under test, which is a generalisation caveat. Neither threatens the tier
contrast, because the simulator is held constant across arms.

## Sample size, from simulation rather than a formula

An earlier version of this protocol sized the study with an unpaired normal
approximation. That was wrong twice: it used the coefficient for 80% power at
each TOST boundary separately rather than for the joint test, and it never
checked whether the interval was calibrated at all. Sizing now comes from
simulating the actual analysis on paired binary outcomes with heterogeneous
task difficulty, 500 replications:

| Per arm | Trials | Power at a true difference of 0 | False equivalence at a true difference of exactly the margin |
|---:|---:|---:|---:|
| 440 | 5 | 58% | 5.0% |
| 616 | 7 | 81% | 4.2% |

Type-I error at the boundary must be at or below 5% for the test to mean
anything, and it is. Power is reported at a true difference of zero and falls as
the truth approaches either margin, which is a property of equivalence testing
and not a defect.

## Calibration, measured 2026-09-15

Eight episodes were run under the exact configuration before committing to the
study, to replace an extrapolated cost with a metered one:

| Quantity | Pilot with GPT-5.6 Sol | Calibrated with DeepSeek V4.1 Flash |
|---|---:|---:|
| Cost per attempt, all in | $0.0153 | **$0.00744** |
| Simulator share of that cost | 63% | 19% |
| Projected cost of 1232 episodes | $18.85 | **$9.17** |

All eight episodes graded, with no infrastructure failures, so the cheaper
simulator drives a working environment rather than a degenerate one. Observed
success was 7 of 8, far above the 0.30 used for sizing, which is expected: the
pilot's three tasks were chosen for difficulty and these 88 are not.

Power was re-simulated at the calibrated base rate. At a task success level near
0.85 the design gives 98% power with 4.0% error at the boundary; at the
conservative 0.30 assumption it gives 81% with 5.2%. Trials stay at 7.

Two harness defects surfaced during calibration and were fixed before the run:

- The rate table had no entry for the new simulator, which failed as an opaque
  gateway error four episodes into the run rather than at startup. Prices are
  now validated against the spec before any episode is dispatched.
- The gateway requires a provider to echo back the requested service tier. The
  DeepSeek endpoint never sends that field, so the simulator could not run. The
  check is now enforced strictly for the agent, whose tier is the independent
  variable and where an unconfirmed tier would invalidate the study, and a
  missing echo is tolerated only for the simulator, where tier is not an
  experimental factor. A contradicting echo still fails for either role.

## Stopping rule: a fixed sample, funded to completion

The run stops at a pre-committed episode count. The dollar ceiling is a safety
rail set above expected spend, and if it ever binds, the run is a failed
experiment to be reported as such, not a completed one.

This is not fussiness. `Gateway.call()` sizes each request against the remaining
shared allowance and applies `output = min(output, affordable)`, so episodes run
near the ceiling receive smaller output caps than episodes run early. A budget
that binds does not merely stop the study, it degrades the agent as the money
runs out, and the degradation lands on whichever episodes happen to run last.
The earlier design, which stopped when the money ran out and called the result
an unbiased random subset, was wrong on both counts.

The harness enforces a hard maximum of $12 per study (`SpendGuard.__init__`).
Expected spend is set well inside it, with the figure confirmed by a metered
calibration before the confirmatory run rather than extrapolated from the pilot.

An interrupted pair is possible: with N lanes, up to N pairs are in flight, so
up to N pairs can lose their second arm if the run halts. The earlier claim that
at most one pair could split was false. Tasks missing an arm are reported by id
and excluded from the paired estimate, never silently dropped.

## Analysis, frozen in code before collection

`scripts/analyze_tier_equivalence.py` is written, tested and frozen. It is
validated on synthetic boundary cases in `tests/test_tier_equivalence.py`,
including the case this study exists to avoid: a small sample must read as
inconclusive, never as equivalent.

- **Estimand**: the task-weighted mean of the per-task difference in success
  rate, flex minus standard. Each task carries equal weight because the claim is
  about the task population, not about the episodes we happened to run.
- **Primary outcome, intention to treat**: every recorded attempt counts, and a
  timeout, provider error or budget stop counts as not successful. A tier that
  does not return did not do the task.
- **Secondary outcome, conditional on grading**: restricted to attempts that
  reached the grader, which separates answer quality from service delivery.
  Both are always reported. Which is primary is fixed here.
- **Interval**: percentile bootstrap over tasks, 10000 draws, fixed seed, 90%
  two-sided, which is the interval matching a 5% TOST.
- **Decision**: equivalent if the interval lies entirely inside the margin;
  different if it lies entirely outside; inconclusive otherwise. Failing to
  establish equivalence does not establish a difference.
- **One analysis, once.** No top-up after seeing the result, because repeated
  testing on accumulating data voids the error rate. If the run is extended, the
  extension is a new study with its own preregistration.

Per-task outcomes and coverage are published so weighting and heterogeneity can
be checked. Per-task pass rates are not reported as findings.

## Execution regime, which is part of the result

Lanes run concurrently within the route, so this measures the tiers under
concurrent load rather than in isolation. Duration figures from this run
describe this regime and do not supersede the pilot's single-lane numbers. Both
tiers face the same lanes and the same adjacent-pair ordering, so the contrast
holds, but the estimand is end-to-end success under this deployment regime, not
isolated model quality. Per-tier active load and request timing are recorded.

The 900 second episode ceiling and the 30 step limit are part of that regime. A
slower tier can fail the wall-clock deadline, and under intention to treat that
counts against it, which is the intended behaviour.

## Known limits

One model, one provider, one domain. The claim is about Doubleword's two tiers
for this pinned DeepSeek route on this retail pool, not about async tiers in
general.

`reward == 1` is a reproducible operational endpoint, not a perfect measure of
task success. Task 41 is documented to mark a policy-permitted Visa refund wrong
against a PayPal reference. It stays in the pool, because removing a task after
seeing results is the bias this design exists to avoid, and at equal task weights
it can move the aggregate by at most 1/88, about 1.1 points. Equal scoring rules
do not guarantee equal scoring error, so a blinded audit of disagreements is
reported as a sensitivity check.

`verify_sources()` checks dataset bytes, not the grader or runtime revision, so
the checkout revision and dependency set are recorded with the run.

## What the review changed

1. The documented command could not run: the ceiling exceeded the harness's hard
   $12 cap and would have raised before the first episode.
2. Budget-triggered stopping was treated as benign. It is not, because the guard
   shrinks output caps as the allowance depletes.
3. The power calculation used the wrong coefficient and was never calibrated.
   Simulation replaced it, and the correction went in our favour: pairing buys
   more than the unpaired formula credited.
4. The communication grader was described as an LLM judge. It is a substring
   match, so the stated reason for excluding 8 tasks was wrong and has been
   replaced with a construct argument.
5. "At most one tier pair can be split" was false at more than one lane.
6. The analysis existed only as prose. It is now code with tests.
7. The margin was affordable rather than justified, and is now stated as a
   tolerance with its relative size spelled out.
8. Episode ids do not depend on the total trial count; only the spec fingerprint
   does. The rationale for pinning trials was corrected.
