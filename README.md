<div align="center">

# compound

**Which host should serve your open model? Measure it on your workload.**

[docs](https://compound-1js.pages.dev/docs/) · [examples](https://compound-1js.pages.dev/examples/)

<br>

![Compound CLI: point it at a model and see every host serving it, with quantization, price, and live status](assets/cli-demo.gif)

</div>

---

## The problem

An open model like DeepSeek V4 Flash or GLM 5.3 Flash is served by ten or more hosts
within weeks of release. Same weights, but each host picks its own quantization, price,
hardware, prompt-cache policy, rate limits, and defaults such as whether reasoning is on.
Those choices change month to month. Leaderboards score the model, not the host, so the
question that decides your bill goes unanswered: **for my traffic, which host is cheapest at
the quality and speed I need?** Compound answers it by running your workload against each
host with the host pinned and verified, and reporting success, cost, latency, and cache-hit
rate per host with confidence intervals.

## Run it on your model

```bash
uv sync --extra dev

# 1. which hosts serve this model, with quant, price, and live status
compound-bench providers z-ai/glm-5.3-flash

# 2. run a benchmark subset across the hosts you care about (drop --go for a free dry run)
compound-bench run terminal_bench \
    --model z-ai/glm-5.3-flash \
    --providers openrouter/deepinfra/fp8,openrouter/parasail/fp8,openrouter/z-ai/fp8 \
    --tasks hello-world,fix-permissions --trials 3 --go

#    or Terminal-Bench 4.0 through Harbor, same provider tokens
compound-bench harbor --model z-ai/glm-5.3-flash \
    --providers openrouter/auto,openrouter/deepinfra/fp8 \
    --dataset terminal-bench@4.0.0 --n-tasks 5 --attempts 2 --go

# 3. one report: success, cost per task, latency, tokens per second, cache hits,
#    and which upstream actually served each call
PYTHONPATH=src python -m compound.bench_report artifacts/<run>
```

You need an OpenRouter key in `.env` for OpenRouter routes. A `--go` run checks every
credential the chosen hosts need before it spends anything.

What a run gives you:

- **Verified pinning.** Each OpenRouter route runs with fallbacks disabled, and the served
  upstream is recorded on every call, so a pinned run is checked rather than trusted.
  Third-party agent harnesses (terminal-bench, Harbor) go through a localhost proxy that
  stamps the pin into every request.
- **A per-call ledger.** Status, latency, prompt and cached tokens, reported cost, and the
  upstream that answered, for every call. Cost is OpenRouter's own accounting where the host
  reports it; hosts that do not report cost get the `--prices` you declare, labeled as such.
- **Intervals, not bare means.** Wilson intervals on rates, two-proportion tests with
  Holm correction when arms are compared.
- **Any OpenAI-compatible host.** `--provider myhost --api-base http://localhost:8000/v1`
  puts your own vLLM box in the same table as the OpenRouter routes.

Provider tokens: `openrouter/<upstream>[/<quant>]`, `doubleword/<realtime|flex>`, or
`direct/<name>` for a host defined in `compound.yaml`. `compound-bench providers <model>`
prints paste-ready tokens.

### Benchmarks

| Benchmark | What it measures | How it grades |
|---|---|---|
| `terminal_bench` | agentic terminal tasks (Terminal-Bench 1.x and 4.0 via Harbor) | official harness in Docker |
| `tau2` | interactive tool-calling support (airline, retail, telecom) | live user simulator + official reward |
| `bfcl` | single-turn function-call generation | official AST checker |
| `ds1000` | data-science code generation | official tests in a pinned container |
| `mmlu` | multiple-choice knowledge, 57 subjects | exact letter match, no judge |

`compound-bench prepare <name>` does any one-time setup. Adding a benchmark is one
registry entry backed by a partitioned manifest; see
[`src/compound/adapters/`](src/compound/adapters).

## Examples

Two runs made with Compound, published with every per-call record. They are small
(one model each, a handful of hosts) and most within-mode host differences do not reach
significance, so read them as worked examples of a report, not a leaderboard.

- [Same model, eight hosts](https://compound-1js.pages.dev/report/): `deepseek-v4-flash` on
  terminal-bench, 588 episodes. Quality tied across hosts; cost per resolved task did not.
- [Pinned host vs router](https://compound-1js.pages.dev/report/tb4/): `glm-5.3-flash` on
  Terminal-Bench 4.0, five routes run concurrently, 1,801 traced calls. Stall rate and cost
  per prompt token by route, and where the router actually sent traffic.

## Backtest on your own traces (TypeScript, earlier stage)

The benchmark sweep tells you how hosts compare on a public task set. The second half of
Compound replays your own production traffic instead:

```bash
bun install
bun run compound import export.jsonl --importer langfuse   # or --importer json
bun run compound curate support                            # traces -> graded cases, sealed decision split
bun run compound experiment support kimi-k3 --paid --cap 2.00
bun run compound gate support --candidate kimi-k3 --reference opus-5 --reason "quarterly cost review"
```

The gate compares candidate and reference on a sealed decision set under a rule fixed
before anyone looks, and returns one of: meets gate, fails, insufficient data, judge
abstained, no reliable improvement. Judges are only trusted after they agree with human
labels (Cohen's kappa with a bootstrap CI). An optional step evolves the candidate's prompt
with [GEPA](https://github.com/gepa-ai/gepa) on train and validation cases only.

Storage is local SQLite and keys live in a git-ignored `.env`. Redacted case inputs are
sent to the providers you explicitly run, and nowhere else.

### Known gaps in the money and safety guarantees

Reviewers found these and they are real. Until they are closed, treat the budget controls
as guard rails for a single sequential run, not as a hard wall:

- [#51](https://github.com/aktasbatuhan/compound/issues/51) The hard USD limit is checked
  before a call and recorded after it, with no reservation, so concurrent runs can overshoot it.
- [#52](https://github.com/aktasbatuhan/compound/issues/52) `compound optimize` calls
  providers from Python outside the cache and spend ledger and records its cost as $0.
- [#53](https://github.com/aktasbatuhan/compound/issues/53) A config file that fails to
  load imports traces with no redaction after only a warning.
- [#54](https://github.com/aktasbatuhan/compound/issues/54) The sealed-set repeat guard is
  warn-only by default and not atomic.

## Development

```bash
uv run pytest -q          # Python: benchmark engine, proxy, ledger, adapters
bun test                  # TypeScript: ingest, curation, experiments, gate, dashboard
```

CI, a security policy, and a contributing guide are tracked in
[#55](https://github.com/aktasbatuhan/compound/issues/55).

## License

[Apache-2.0](LICENSE)
