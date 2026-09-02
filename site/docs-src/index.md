title: Overview
order: 10

# Compound documentation

Compound measures how the hosts that serve an open model differ on your workload.
You name a model and a list of hosts, pick a benchmark or bring your own traces,
and get one table per host: task success, cost, latency, tokens per second, and
cache-hit rate, with confidence intervals and a record of every model call.

Two halves, two languages:

| Half | Language | Entry point | What it does |
|---|---|---|---|
| Benchmark sweeps | Python | `compound-bench` | Run public benchmarks across pinned hosts and report per host |
| Trace backtesting | TypeScript | `bun run compound` | Turn your production traces into graded cases and gate a model or host switch on them |

Most people start with the sweep. The trace pipeline is earlier stage and has
[known gaps](traces/#known-gaps) in its money controls.

## Install

```bash
git clone https://github.com/aktasbatuhan/compound
cd compound
uv sync --extra dev          # Python half, installs compound-bench
bun install                  # TypeScript half, optional
```

Keys go in a git-ignored `.env` at the repo root:

```env
OPENROUTER_API_KEY=
DOUBLEWORD_API_KEY=          # only if you use doubleword/* tokens
```

`compound-bench` reads `.env` itself. Do not `source` it into your shell.

## First run

```bash
# who serves this model, with quantization, price, and live status
compound-bench providers z-ai/glm-5.3-flash

# a dry run prints the plan and spends nothing
compound-bench run terminal_bench --model z-ai/glm-5.3-flash \
    --providers openrouter/deepinfra/fp8,openrouter/z-ai/fp8 \
    --tasks hello-world --trials 1

# add --go to execute
```

Every `run` and `harbor` invocation is a dry run until you add `--go`. A `--go`
run checks that every credential the chosen hosts need is present before it
sends a request.

## Where things land

| Path | Contents |
|---|---|
| `artifacts/<run>/<host>/` | per-host episode output from the harness |
| `artifacts/<run>/report/` | `summary.json`, `episodes.csv`, `per_task.csv`, `charts.html` |
| `<ledger-dir>/<host>.jsonl` | one row per model call when a ledger is enabled |

## Pages

- [Providers](providers/): provider tokens, pinning, verification, the proxy
- [Benchmarks](benchmarks/): the five shipped benchmarks and how to run subsets
- [Terminal-Bench 4.0](harbor/): running TB4 through Harbor, and on a cloud VM
- [Reports and ledgers](reports/): what a run writes and how to compare arms
- [Serving metrics](serving/): first-token and decode speed per host over time
- [Your own traces](traces/): the TypeScript pipeline
- [Configuration](config/): `compound.yaml` and direct hosts
- [CLI reference](reference/): every flag, generated from `--help`
