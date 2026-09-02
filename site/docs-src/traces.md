title: Your own traces
order: 70

# Backtest on your own traces

The benchmark sweep compares hosts on a public task set. The TypeScript half of
Compound compares them on your own traffic instead: it imports the traces you
already produce, turns them into graded cases, replays candidates against them,
and decides a switch under a rule fixed before anyone looks.

This half is earlier stage than the sweep. Read [Known gaps](#known-gaps)
before relying on its money controls.

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

## Known gaps

These were found in review and confirmed in the code. Until they are closed,
treat the budget controls as guard rails for one sequential run, not a hard
wall.

- [#51](https://github.com/aktasbatuhan/compound/issues/51) The hard USD limit is checked before a call and recorded after it, with no reservation. Concurrent runs can overshoot it.
- [#52](https://github.com/aktasbatuhan/compound/issues/52) `compound optimize` calls providers from Python outside the completion cache and spend ledger, and records its cost as $0.
- [#54](https://github.com/aktasbatuhan/compound/issues/54) The sealed-set repeat guard now blocks by default, but its preflight is a read, not a claim, so two gates started at the same moment can both pass it.

Closed: a `compound.yaml` that fails to load now stops the import instead of persisting raw traces; `--unsafe-no-redaction` is the explicit override (#53).
