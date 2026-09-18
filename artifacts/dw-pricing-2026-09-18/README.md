# Doubleword pricing diagnostic, 2026-09-18

Four authorized local diagnostic calls, no benchmark or VM launched. Explicit
1h cache markers on every call. One unique prefix per requested tier, repeated
once. Model: deepseek-ai/DeepSeek-V4.1-Flash. Max output 64, medium reasoning.
Reservation cap $0.15; actual aggregate-meter spend **$0.00622429**.

| Requested policy | Phase | Meter USD | Corrected estimate USD |
|---|---|---:|---:|
| priority | Cache write | 0.00341430 | 0.00341430 |
| priority | Cache read | 0.00005297 | 0.000052965 |
| flex | Cache write | 0.00271128 | 0.00271128 |
| flex | Cache read | 0.00004574 | 0.0000457368 |

Each interval booked exactly one request and matched its input/output token
counts. A quiet baseline preceded the test. Both repeat calls reported all input
tokens cached. The corrected rates match to less than $0.00000001 per request.

The website displayed cache reads as $0.01/M but also described a 0.02 multiplier.
Our original estimates used the displayed dollar figure and overestimated read
charges. Meter charges are consistent with 0.02 times each tier's input rate:
$0.003/M priority and $0.0024/M flex. One-hour cache writes match 2x tier input
rates ($0.30/M and $0.24/M). Output rates used were $0.60/M and $0.48/M.
These calls validate those combined charge calculations, not every rate in
isolation. In particular, inferred read rates condition on published output
rates. Raw response token counts include reasoning output.

Sources: [model pricing](https://docs.doubleword.ai/inference-api/models/deepseek-ai-deepseek-v4-1-flash),
[cache pricing](https://docs.doubleword.ai/inference-api/prompt-caching).

`report.json` preserves original displayed-rate estimates, responses and billing
snapshots. `request-*.json` contains exact synthetic request bodies.
`executed-probe.py` is the original executed source; the reusable script was
subsequently corrected. `reconciliation.json` holds revised calculations.
`model-catalog.json` is a free metadata lookup; it contains no pricing fields.

The meter is aggregated by model, not tier. Exact count matching supports
isolation but is not request-linked proof of served tier. The account lacks
Read Requests permission, and responses still omit service_tier. Accordingly,
this is reconciliation of charges during isolated requested-policy intervals,
not independent per-request served-tier confirmation. No quality or latency
comparison should be inferred from these four calls.

Deduct this spend from the shared $10 pilot allowance. The v2 runbook retains a
conservative $0.15 diagnostic reserve and uses a $9.85 inference cap.
