title: Implementation and development
order: 80

# Implementation and development

Compound has two execution surfaces. The Python benchmark CLI compares serving
hosts on benchmark tasks and request shapes. The TypeScript trace pipeline
imports production traces and evaluates candidate changes on curated cases.

## Code map

| Area | Responsibility |
|---|---|
| `src/compound/bench.py` | Benchmark CLI and execution opt-in |
| `providers_registry.py`, `orproxy.py` | Provider tokens, request pinning, harness proxy |
| `adapters/`, `provider_sweep.py` | Official benchmark harness integration and host sweeps |
| `serving_metrics.py`, `call_ledger.py` | Serving measurements and per-call evidence |
| `bench_report.py`, `scripts/analyze_*.py` | Reports and comparison analysis |
| `packages/pipeline`, `ingest`, `redaction`, `contract` | Normalize, redact, validate, classify, persist |
| `packages/curation`, `assertions`, `judge`, `gate` | Case provenance, grading, calibration, sealed decisions |
| `packages/execution`, `storage`, `cli` | Replay, spend reservations, cache, durable state, orchestration |
| `src/compound/optimize_product.py`, `spend_ledger.py` | Python optimizer on the trace pipeline's SQLite budget |
| `packages/api`, `dashboard` | Local API and Next.js views |
| `site/docs-src`, `scripts/build_site.py` | Documentation source and generated pages |

Paths without a directory prefix in the Python rows are under `src/compound/`.
Use `compound-bench` for benchmarks and `bun run compound` for traces; the Python
package also installs a separate `compound` command for its experiment engine.

## Spend controls

| Surface | Paid opt-in | Budget boundary |
|---|---|---|
| TypeScript experiments, judging, gates, optimization | `--paid`, positive cap, enabled config budget | Shared SQLite reservations; optimizer uses the same tables |
| BFCL / DS-1000 benchmark runs | `--go`, Python budget configuration | Python benchmark budget controls |
| tau2 / MMLU / terminal-bench / Harbor | `--go` | No shared SQLite dollar cap; constrain workload size |
| Serving measurements | `--go` | No dollar cap; preview route/shape/mode/repetition count |
| Endpoint probes | `--probe --go` | One probe per discovered endpoint, no dollar cap |

The JSONL call ledger measures calls. It does not authorize or reserve spend.
SQLite reservations serialize estimated headroom, not guaranteed maximum charges.
An in-flight call may exceed its estimate, and reservations expire after 15 minutes
without checking process liveness. External billing remains the reconciliation source.

## Implemented behavior and remaining boundaries

| Capability | Implemented | Remaining boundary |
|---|---|---|
| Agentic trace replay | `experiment --agentic` and `gate --agentic`; recorded/mocked results, blocked tools, per-turn accounting | `live_read_only` execution is unsupported; this is replay, not arbitrary live tool execution |
| Importers | Langfuse, canonical JSON, OTel GenAI | Broader vendor exports still require mapping and validation |
| Native Anthropic | Python serving measurements; tau model conversion selects the native provider | Chat-completions harness proxy refuses it; TypeScript native provider support is separate work |
| First-party OpenAI | Configured direct standard/flex routes with provider-specific request fields | Support must be checked per harness and model |
| Host pinning | Routing constraint and provider-echo comparison when recorded | Quantization suffixes are labels; absent echoes remain unverified |
| Local API | Loopback default, shared storage and pipeline behavior | No authentication for exposure beyond the local machine |

Older design documents and open epics may describe only part of this behavior.
Review each execution path before marking a capability complete. Continuous
ingest, automatic curation, and drift-triggered re-decisions remain future work.

## Validation workflow

```bash
uv sync --extra dev
bun install --frozen-lockfile
uv run ruff check src tests scripts
uv run compound validate-config
uv run pytest -q
bun run lint
bun run typecheck
bun test packages
uv run --with markdown python scripts/build_site.py
```

Commit rebuilt documentation alongside its Markdown sources. Test paid paths
with stub providers; a green suite does not establish live provider compatibility,
benchmark harness availability, or the correctness of an unpublished experiment.
