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

## Run a controlled cache experiment

`cache-study` compares OpenRouter automatic routing, pinned routes and Doubleword
realtime/flex. Each independent trial primes a new prefix, waits for the selected idle delay,
then submits a burst of probes. Priming completes before any probe starts. No retries
are added. Trial order is shuffled with a recorded seed.

```bash
compound-bench cache-study \
    --providers openrouter/auto,openrouter/relace \
    --model-or YOUR_MODEL \
    --shapes benchmarks/cache-study/shapes.json \
    --reuse exact,prefix,none --delays 0,30 --concurrency 1,4 \
    --trials 3 --max-calls 252 --out artifacts/cache-study
```

To include Doubleword, append `doubleword/realtime,doubleword/flex` to `--providers`
and supply `--model deepseek-ai/DeepSeek-V4-Flash-0731` alongside the matching
`--model-or deepseek/deepseek-v4-flash-0731`. Four routes double the plan to 504 calls;
raise `--max-calls` explicitly. Doubleword runs require cache markers and refuse to
start if `COMPOUND_DW_CACHE` disables them. Paid runs load keys from `.env`.

This is a dry run: **72 priming calls + 180 probes = 252 calls**, with 1,080 seconds
of scheduled idle time plus request time if executed. Confirm the model and pinned
host are available before adding `--go`. The call limit rejects larger plans before
execution; it is not a dollar cap. Execution requires a new output directory.
For a smaller wiring check, use `--reuse exact --delays 0 --concurrency 1 --trials 1`:
two routes, four calls total.

The bundled synthetic reference has 32,768 characters before instructions and trial
identifiers, with a 16-token output budget. It is not a quality benchmark or a claim
about an exact token count. Replace it with your own shapes to test representative
prompt sizes; every shape must declare `max_tokens`.

The reuse conditions are:

- `exact`: probes repeat the priming messages exactly.
- `prefix`: probes retain the document prefix and change a request ID in the final message.
- `none`: each probe gets a new identifier at the start of the prompt as a fresh-prefix control.

Every route and trial gets an independent fixed-length prefix identifier, preventing
one experiment arm from priming another. Consequently, prompts across routes differ
in this identifier; they contain the same reference content. Prefix isolation avoids
cross-arm cache reuse but does not control shared capacity or provider load.

Concurrency is the requested probe burst size, not a sustained requests-per-second
load. Idle time is measured from priming completion; `actual_idle_s` records each
probe's start gap. A failed prime remains in the evidence and probes still run, so
inspect priming failures before interpreting a low cache share. Provider cache hits
are measured, never assumed from the phase name.

```bash
compound-bench serving-report artifacts/cache-study/results.jsonl \
    --out artifacts/cache-study/report.html --title "Routing and prefix reuse"
```

The report separates reuse policy, delay, burst size and prime/probe phases. Its cost
comparison pairs priming and probe groups within matching conditions. The output
`experiment.json` records planned calls, trial order, routes and a hash of the shapes;
per-call records include prompt hashes, settings, timings and reported upstreams.

Start by asking whether the cost gap changes with reuse, delay or burst size. Routing
distributions and cache shares are observations; they do not by themselves establish
why a provider cached or evicted a prefix. Three trials are a pilot, not enough to
support a general provider ranking. Repeat a larger balanced experiment across time
windows before publishing broader conclusions.
