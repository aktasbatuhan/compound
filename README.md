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
    --tasks hello-world,fix-permissions --trials 3 \
    --output artifacts/example --call-ledger artifacts/example/calls.jsonl --go

#    or Terminal-Bench 4.0 through Harbor, same provider tokens
compound-bench harbor --model z-ai/glm-5.3-flash \
    --providers openrouter/auto,openrouter/deepinfra/fp8 \
    --dataset terminal-bench@4.0.0 --n-tasks 5 --attempts 2 \
    --ledger-dir artifacts/harbor/calls --go

# 3. episode report, then per-call cache and routing evidence
PYTHONPATH=src python -m compound.bench_report artifacts/example
compound-bench ledger artifacts/example/calls.jsonl --hosts
```

You need an OpenRouter key in `.env` for OpenRouter routes. A `--go` run checks every
credential the chosen hosts need before it spends anything.

What a run gives you:

- **Host pinning.** Pinned OpenRouter routes disable fallbacks. Third-party agent harnesses
  (terminal-bench, Harbor) use a localhost proxy to inject the pin. Enable the call ledger
  to compare the requested host with the provider echo; a missing echo stays unverified.
  Quantization suffixes label discovered endpoints but do not constrain quantization.
- **A per-call ledger.** With recording enabled, status, latency, token usage, reported cost,
  and provider echo are captured when available. Missing values stay unknown; partial cost
  totals are labeled. Episode reports can derive cost from declared `--prices`. The JSONL
  call ledger records evidence; it does not enforce a spend limit.
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

### Money controls

Paid calls in the TypeScript trace pipeline and its Python optimizer reserve their estimates
against the shared SQLite ledger inside a write transaction before the provider is
called, and settle the reservation at the actual charge afterwards. Concurrent runs
therefore serialize on the same limit instead of each checking a stale total. A
gate claims its sealed cohort the same way before spending. Runs stay dry without
`--paid` and a per-run `--cap`; a config that fails to load stops an import instead
of persisting raw traces.

Benchmark `run`, `harbor`, `serving`, and `providers --probe` require `--go` to spend.
BFCL and DS-1000 runs use their Python budget controls. The other benchmark paths
do not enforce the shared SQLite limit or a dollar cap; bound their tasks, trials,
and serving repetitions before opting in. Reservations in the trace pipeline are
estimates, so actual charges can exceed reserved amounts.

## Development

```bash
uv run pytest -q          # Python: benchmark engine, proxy, ledger, adapters
bun test                  # TypeScript: ingest, curation, experiments, gate, dashboard
```

CI runs both suites plus lint and a docs-build check on every pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[current implementation map](site/docs-src/development.md).

## License

[Apache-2.0](LICENSE)
