# Compound

## Product (TypeScript, `packages/`)

The ingest path is working end to end. From a Langfuse export to queryable, redacted traces:

```bash
bun install
bun run --filter '@compound/cli' typecheck     # or: bun test packages
bun run packages/cli/src/main.ts import export.jsonl --importer langfuse  # or --importer json
bun run packages/cli/src/main.ts curate support   # traces -> partitioned eval cases
bun run packages/cli/src/main.ts experiment support <model>   # dry run; add --paid --cap USD
bun run packages/cli/src/main.ts status
bun run packages/cli/src/main.ts serve         # local API on 127.0.0.1:4319
cd packages/dashboard && bun run dev           # dashboard on localhost:3000 (needs the API up)
```

The dashboard (`@compound/dashboard`, Next.js) is a view over the API: a labeling/review
workflow, cases, a task×model matrix, the diagnostic queue, and imports. It holds no data and
re-implements no logic — set `COMPOUND_API_URL` to point it at a non-default API.

`experiment` is money-safe by default: without `--paid` it makes zero provider calls and reports
the estimated cost. `--paid` needs `budget.paid_runs_enabled: true`, a positive
`budget.hard_limit_usd`, and a `--cap`. Every completion is cached, so a re-run is $0. Paid runs
currently work against chat-completions providers (OpenRouter, OpenAI-compatible); the Flex
Responses route for Doubleword's cheap candidates is the next provider addition.

Packages: `contract` (the portable trace contract), `config` (one `compound.yaml` schema
shared with the Python engine), `storage` (SQLite/Drizzle), `ingest` (Langfuse + plain-JSON
normalizers), `redaction` (pre-persistence), `pipeline` (ingest composition), `curation` (cases,
provenance, sealed partitions), `assertions` (deterministic grading), `execution` (candidate
runner, budget ledger, cache), `api` (Hono), `cli`.

Design docs: `docs/product-plan-20260722.md`, `docs/trace-contract-v1.md`,
`docs/ingest-pipeline-v1.md`, `docs/curation-v1.md`, `docs/api-design-v1.md`,
`docs/langfuse-import-mapping.md`.

Note: the trace contract stays marked **draft** until a real Langfuse export imports
losslessly. The ingest fixtures are synthetic, built from the documented schema.

## Benchmark engine (Python, `src/compound/`)

Compound is currently benchmark-first: prove that cheaper open models can meet a fixed
quality gate — with prompt optimization where it helps — before building trace ingestion
or a dashboard.

Status (2026-07-22): the DS-1000 line is closed. Three GEPA prompt campaigns failed to beat
the unmodified GLM 5.2 `minimal` route, and the pre-declared final sealed gate then rejected
that route against the 80–90%-of-reference bar
(`docs/ds1000-final-sealed-gate-results-20260722.md`). DS-1000 fresh evidence is exhausted.
The BFCL single-turn baseline matrix is the active screening surface.

The initial proof pack is deliberately small and reproducible:

- DS-1000: 30 stratified single-shot data-processing cases.
- BFCL: 30 cases split evenly between single-turn and multi-turn tool use.
- tau-bench: 20 cases balanced across airline, retail, and telecom, with three trials.

Each manifest is divided into optimizer-train, optimizer-validation, and decision-test.
The optimizer runner rejects decision-test access unless final evaluation explicitly enables it.

## Local setup

Always run the Python engine through the project environment. `uv run pytest` without a synced
`dev` extra can fall back to a globally installed pytest, which may import a stale `gepa` from
your user site directory and produce failures that look like library API drift.
`tests/test_environment.py` guards against this and tells you to re-sync.

```bash
uv sync --extra dev
cp .env.example .env
uv run compound validate-config
uv run compound prepare-manifests
uv run pytest -q
uv run ruff check .
```

After preparing sources and building the DS-1000 evaluator image, run the bounded recursive
optimization proof with:

```bash
uv run compound run-ds1000-proof --max-metric-calls 20
```

The scaled GEPA v2 path uses perturbation-family isolation, correctness-dominant objectives,
trial-aware caches, disjoint GEPA train/validation IDs, and a resumable capped run:

```bash
uv run compound prepare-ds1000-gepa-v2
docker build -f docker/ds1000-numpy.Dockerfile \
  -t compound-ds1000-numpy:20260720-v3 .
uv run compound probe-ds1000-gepa-v2 --reasoning-effort low
uv run compound run-ds1000-gepa-v2 \
  --reasoning-effort low --max-metric-calls 120 --max-tokens 4096 \
  --validation-trials 2 --decision-trials 2 --experiment-cap-usd 4
```

The proof uses only the NumPy/Pandas optimizer-train and optimizer-validation cases. GEPA never
receives decision-test cases. Model calls and grader outputs are content-addressed and cached,
and every paid call is recorded against the hard cap in `artifacts/budget.json`.

Only after the optimizer has stopped, open the decision-test firewall for the final comparison:

```bash
uv run compound evaluate-ds1000-decision artifacts/optimization/<run-id>
```

For GEPA v2, use the corresponding one-time decision gate:

```bash
uv run compound evaluate-ds1000-gepa-v2-decision \
  artifacts/optimization/<v2-run-id>
```

Successful and failed provider calls are appended to
`artifacts/telemetry/model_calls.jsonl`. Each record includes provider, requested and resolved
model, latency, token counts, reasoning tokens, finish reason, and end-to-end output TPS. Generate
an aggregate model/provider report with:

```bash
uv run compound telemetry-report --output artifacts/telemetry/summary.json
```

The TPS value is completion tokens divided by full request latency, so it intentionally includes
provider queueing and time-to-first-token; it is not a server-only decode benchmark.

Screen models on a fresh origin-isolated DS-1000 cohort (excludes every prior `ds1000*`
manifest automatically), then run selected models with repeated trials:

```bash
uv run compound prepare-ds1000-baseline-matrix --output benchmarks/manifests/<name>.json
uv run compound run-ds1000-baseline-matrix \
  --manifest benchmarks/manifests/<name>.json \
  --model zai-org/GLM-5.2-FP8 --reasoning-effort minimal \
  --trials-per-case 3 --experiment-cap-usd 2.5 --output artifacts/baselines/<name>.json
```

Note: the flex path reserves $0.02 of cap headroom per new request up front, so the cap must
exceed `0.02 × cases × trials` even when the real spend is far lower. `--reasoning-effort`
applies only to Doubleword flex candidates and is part of the completion fingerprint.

Doubleword Flex transport compatibility can be smoke-tested with
`uv run compound run-doubleword-flex-smoke`.

The BFCL single-turn baseline matrix uses the official bfcl-eval prompting and AST checkers
(multi-turn cases are recorded but never graded — they need the official live harness):

```bash
uv sync --extra dev --extra bfcl
uv run compound run-bfcl-baseline-matrix --experiment-cap-usd 4.0
```

Legacy flat DS-1000 trace caches can be split without repeating model calls:

```bash
uv run compound migrate-ds1000-cache --archive
```

Regrade a frozen run under a corrected evaluator image exclusively from cached completions:

```bash
uv run compound regrade-ds1000-run artifacts/optimization/<run-id> \
  --evaluator-image compound-ds1000-numpy:20260717-v2 \
  --case-id ds1000_395
```

Credentials stay in `.env`, which is ignored by Git:

```env
OPENROUTER_API_KEY=
DOUBLEWORD_API_KEY=
```

OpenRouter supplies the GPT-5.6 Sol and Claude Opus 4.8 reference runs. Doubleword supplies
the GLM 5.2 and Nemotron 3 Ultra candidate runs. No paid evaluation should start without a
configured run budget. Paid calls are disabled by default in `compound.yaml`; enabling them
also requires a positive hard USD limit.

Benchmark checkouts are cached under `.compound/sources/` and are not committed. The selected
case IDs and exact source revisions are committed under `benchmarks/manifests/`.
