title: Serving metrics
order: 60

# Serving metrics

Task success is one axis. `compound-bench serving` measures the others
directly: time to first token, decode speed, and cost per route, repeated on a
schedule so you can see how a host moves through the day.

```bash
compound-bench serving \
    --providers openrouter/deepinfra,doubleword/realtime,doubleword/flex,openrouter/auto \
    --model-or deepseek/deepseek-v4-flash-0731 \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --shapes shapes.json --rounds 8 --interval 3600 --reps 3 \
    --out artifacts/serving --go
```

Omit `--go` to preview the call count without credentials, output writes, or
provider calls. This command has no dollar cap: the preview counts calls, not
maximum charges. Existing automation must now pass `--go` explicitly.

- `--model-or` is the OpenRouter slug; `--model` is the id Doubleword or a direct host uses for the same weights. Hosts name models differently.
- `--shapes` is a JSON file mapping a name to `{messages, response_format}`, so you measure the request shapes your workload actually sends, including structured output.
- `--rounds` and `--interval` schedule repeated measurements. `--reps` repeats each (route, mode, shape) cell within a round.

Every call is streamed so first-token and decode time are measured, not
inferred. Results land in `results.jsonl` under `--out`, one row per call, and
feed the per-host profile in the sweep report.

## Make an interactive report

Turn a serving run into a single HTML file you can open locally or publish:

```bash
compound-bench serving-report artifacts/serving/results.jsonl \
    --out artifacts/serving/report.html --title "Our support workload"
```

This command runs offline and makes no provider calls. The HTML includes charts,
provider highlighting on hover, keyboard focus and tap, plus downloads of sanitized
measurements, aggregates and a source-hash manifest. No server or JavaScript libraries
are needed. Without JavaScript, all conditions remain readable.

Select a workload condition to compare cost and first-token latency, first-token p50/p90,
generation speed, cache token share, cold/warm cost and failures. A routing table shows
reported upstreams. Every metric includes its definition and measurement coverage.

You can pass several JSONL or `.jsonl.gz` files. Matching shape names and settings are
pooled across files and rounds. Use distinct shape names for different prompts or
workloads. Reasoning mode, temperature and output budget select separate conditions;
model IDs and cache-marker settings remain separate provider entries. Older records
without these fields are labeled “not recorded”. New serving runs record output budgets.

All supplied calls count, including HTTP 402 credit errors and failed calls. Latency,
decode, cache and cost summaries use successful calls with the necessary measurements.
Cost is reported total request charge per million input tokens, including output charges;
it excludes failed-call charges. Missing costs stay unknown. Cold/warm comparisons require
both groups for the same provider, model and marker setting. Warm includes the initial
cache-populating call. This is serving measurement, not an answer-quality evaluation.

Request/response text, raw error messages and credentials are excluded from downloads.
Review route, model and shape labels before publishing; those labels are included.
Existing output files are preserved unless you pass `--force`.
