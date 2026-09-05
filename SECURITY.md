# Security

## Reporting

Email aktasbatuhann@gmail.com with "compound security" in the subject. Please
do not open a public issue for anything that could expose keys, traces, or
spend. You will get a reply within a few days.

## What Compound handles

- **Provider API keys.** Read from `.env` or the environment at run time, never
  written to disk by Compound, and never sent anywhere except the host they
  belong to. The pinning proxy holds the real key in-process and hands the
  harness a placeholder.
- **Trace content.** The trace pipeline stores imported traces in a local
  SQLite database. Redaction rules from `compound.yaml` run before persistence.
  Redacted case inputs are sent only to the providers you explicitly run.
- **Money.** Benchmark runs are dry runs without `--go`; trace experiments need
  `--paid` and a cap.

## Money controls

Paid calls in the TypeScript trace pipeline and its Python optimizer reserve
their estimated cost against the shared SQLite ledger
inside an IMMEDIATE transaction before the provider is called, and settle the
reservation at the actual charge afterwards, so two runs on one database cannot
both pass a check only one could afford. The Python optimizer uses the same
tables under the same rules and refuses to run without a budget. A reservation
older than 15 minutes expires without checking whether its process is still alive.
These controls serialize estimates; they cannot guarantee that a provider's actual
charge stays below its estimate.

Benchmark telemetry uses separate JSONL files, not this reservation ledger.
`run`, `harbor`, `serving`, and `providers --probe` require `--go` to spend.
BFCL and DS-1000 enforce their Python budget controls; tau2, MMLU, terminal-bench,
Harbor, serving measurements, and endpoint probes have no shared dollar cap.
Limit the workload before opting in. Report violations of the documented
controls using the contact above.
