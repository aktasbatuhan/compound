title: Benchmarks
order: 30

# Benchmarks

One front door runs any task subset from any shipped benchmark. Every benchmark
ships with its official grader; Compound adds the provider sweep, the report,
and the ledger around it.

| Benchmark | What it measures | How it grades | Needs |
|---|---|---|---|
| `terminal_bench` | agentic terminal tasks | official harness in Docker, task test suites | Docker, `prepare terminal_bench` |
| `tau2` | interactive tool-calling support (airline, retail, telecom) | live user simulator + official reward | `prepare tau2` |
| `bfcl` | single-turn function-call generation | official AST checker | nothing |
| `ds1000` | data-science code generation | official tests in a pinned container | Docker image |
| `mmlu` | multiple-choice knowledge, 57 subjects | exact letter match, no judge | `prepare mmlu` |

Terminal-Bench 4.0 runs through a separate `harbor` subcommand; see
[Terminal-Bench 4.0](../harbor/).

## Commands

```bash
compound-bench list                                # benchmarks, case counts, how each runs
compound-bench prepare tau2                        # one-time engine or dataset setup
compound-bench tasks tau2 --contains retail        # case ids, optionally filtered
compound-bench run tau2 --model ... --tasks retail:10,airline:3 --go
```

`run` accepts one of `--tasks` (explicit ids), `--partition` (every case in a
partition), or `--contains` (substring match). `--trials N` repeats each case.

## Partitions

Each benchmark manifest splits its cases into fixed partitions:
`optimizer_train`, `optimizer_validation`, and `decision_test`. The split is
hash-based, so it is stable across machines. If you use Compound's prompt
optimizer, it sees only the first two; `decision_test` is the sealed set.

## Sweeping hosts

```bash
compound-bench run tau2 \
    --model deepseek/deepseek-v4-flash-0731 \
    --providers openrouter/deepinfra/fp4,openrouter/baseten/fp8,doubleword/realtime,doubleword/flex \
    --tasks airline:6,retail:20 --trials 3 --max-tokens 8192 \
    --output artifacts/dsflash --go
```

Each host runs the same cases with the same settings. The output directory gets
one subdirectory per host, and [`bench_report`](../reports/) turns it into one
table.

Give `--max-tokens` headroom on agentic benchmarks. A low cap truncates long
tool-calling turns and records a failure the host did not earn.

## Terminal-Bench 1.x

`run terminal_bench` delegates to the original terminal-bench harness through
the pinning proxy:

```bash
compound-bench run terminal_bench \
    --model deepseek/deepseek-v4-flash-0731 \
    --providers openrouter/deepinfra/fp4,doubleword/flex \
    --tasks hello-world,fix-permissions --trials 3 \
    --call-ledger artifacts/tb/calls.jsonl --go
```

Flags specific to it:

- `--tb-agent` picks the harness agent (default `terminus`); `--tb-concurrent` sets tasks per host.
- `--tb-timeout-mult N` multiplies every task's agent time limit. Results are labeled non-official when set. Use it when a slow host would otherwise fail only for being slow.
- `--reasoning on|off` pins the model's reasoning mode through the proxy. Hosts disagree on the default, so an unpinned comparison mixes modes.
- `--no-cache-optin` turns off the prompt-cache markers injected for hosts whose
  cache is opt-in. They are injected by default.
- `--call-ledger PATH` records every model call. Cache-hit and routing claims need this; episode results only carry totals.

## Adding a benchmark

A benchmark is one registry entry: a manifest of partitioned case ids and a
runner that accepts `(models, case_ids)`. See
[`src/compound/adapters/`](https://github.com/aktasbatuhan/compound/tree/main/src/compound/adapters)
for the interface and the five existing adapters.
