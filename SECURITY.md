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

Every paid call reserves its estimated cost against the shared SQLite ledger
inside an IMMEDIATE transaction before the provider is called, and settles the
reservation at the actual charge afterwards, so two runs on one database cannot
both pass a check only one could afford. The Python optimizer uses the same
tables under the same rules and refuses to run without a budget. A reservation
left by a crashed process expires after 15 minutes. If you find a way to spend
outside the ledger, report it as above.
