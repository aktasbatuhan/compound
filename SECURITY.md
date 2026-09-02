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

## Known limitations

These are tracked publicly and are listed here so nobody relies on a guarantee
the code does not yet give:

- [#51](https://github.com/aktasbatuhan/compound/issues/51): the hard USD limit
  in the trace pipeline is checked before a call and recorded after it, so
  concurrent runs can overshoot it.
- [#52](https://github.com/aktasbatuhan/compound/issues/52): `compound optimize`
  spends outside the completion cache and spend ledger.

If you find something in the same family, report it the same way.
