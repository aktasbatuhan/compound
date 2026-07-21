# Compound

Compound is currently benchmark-first: prove that recursive prompt optimization can make
cheaper models meet a fixed quality gate before building trace ingestion or a dashboard.

The initial proof pack is deliberately small and reproducible:

- DS-1000: 30 stratified single-shot data-processing cases.
- BFCL: 30 cases split evenly between single-turn and multi-turn tool use.
- tau-bench: 20 cases balanced across airline, retail, and telecom, with three trials.

Each manifest is divided into optimizer-train, optimizer-validation, and decision-test.
The optimizer runner rejects decision-test access unless final evaluation explicitly enables it.

## Local setup

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
