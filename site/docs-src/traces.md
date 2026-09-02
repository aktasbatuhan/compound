title: Your own traces
order: 70

# Backtest on your own traces

The benchmark sweep compares hosts on a public task set. The TypeScript half of
Compound compares them on your own traffic instead: it imports the traces you
already produce, turns them into graded cases, replays candidates against them,
and decides a switch under a rule fixed before anyone looks.

This half is earlier stage than the sweep. See [Money controls](#money-controls)
for what the budget guarantees.

```bash
bun install
bun run compound init                                       # writes compound.yaml
bun run compound import export.jsonl --importer langfuse    # or json, otel
bun run compound curate support                             # traces -> cases, partitions
bun run compound experiment support kimi-k3                 # dry run, cost estimate
bun run compound experiment support kimi-k3 --paid --cap 2.00
bun run compound gate support --candidate kimi-k3 --reference opus-5 \
    --reason "quarterly cost review"
```

## Pipeline

1. **Import.** Langfuse export, portable JSON, or OTel GenAI spans. Redaction rules from `compound.yaml` run before anything is stored.
2. **Curate.** Traces become provenance-typed cases split into train, validation, calibration, and a sealed decision partition. `suggest-assertions` proposes deterministic checks from the traces.
3. **Experiment.** A model or host replays the cases. Every completion is content-addressed and cached, so a re-run of the same fingerprint is free.
4. **Judge.** Assertions grade first, at no cost. An LLM judge is used only after `judge calibrate` shows it agrees with human labels (Cohen's kappa with a bootstrap interval). Until then it abstains.
5. **Gate.** Candidate and reference are compared on the sealed set with paired-bootstrap intervals under a non-inferiority rule declared up front. Opening the sealed set requires a `--reason`.
6. **Optimize.** Optionally, GEPA evolves the candidate's prompt on train and validation cases only. The result is re-gated as a new declaration.

The gate returns exactly one of five verdicts: meets gate, fails, insufficient
data, judge abstained, no reliable improvement. `compound eval` is the CI form
and exits 0, 1, or 2.

`compound view compare <task>` prints cost, latency, throughput, and quality per
route, with an optional priority vector for a weighted ranking and Pareto
frontier. `compound serve` exposes the same data on a local API, and
`packages/dashboard` renders it.

## Data handling

Storage is local SQLite. Keys stay in `.env`. Redacted case inputs are sent to
the providers you explicitly run, and nowhere else.

## Money controls

Every paid call goes through one primitive, in both languages:

1. **Reserve.** The call's estimated cost (prompt size plus the full output budget at the model's price) is written to `spend_reservations` inside an IMMEDIATE transaction, after checking the committed ledger plus every other live reservation against the per-run cap and the global hard limit. Two processes on one database serialize here.
2. **Call** the provider.
3. **Settle.** The reservation is deleted and the actual charge appended to `spend_records` in one transaction. A call that fails releases its reservation. A call that returns no usage is charged at the estimate, never zero.

`compound optimize` requires `--paid --cap` and hands the Python side the database path, the cap, the limit, and the prices; every candidate rollout and reflection call reserves and settles the same way, and the run's measured cost is stored with the optimization. A reservation left by a crashed process expires after 15 minutes.

A paid gate claims its sealed cohort atomically with the prior-decision check before running either experiment, so two gates on the same cohort cannot both pass the preflight. The claim is released when the gate ends; the recorded verdict is what blocks the next attempt.
