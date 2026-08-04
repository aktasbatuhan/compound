<div align="center">

# compound

**Backtest every model switch on your own traffic.**

Compound turns production traces into a standing answer to one question:
*can I move this workload to a cheaper model or a faster provider without losing quality?*
It replays candidates against your graded history, optimizes them until they clear your bar,
and hands you a verdict with a confidence interval, never a vibe.

`local-first` · `money-safe by default` · `statistically honest` · `Apache-2.0`

</div>

---

## The problem

Every leaderboard tells you which model is best on average. None of them can tell you
whether the switch is safe on **your** workload, and the gap is not academic:

- **The serving host moves quality as much as the model does.** In our pinned-host sweep,
  identical open-model weights swung from 8/13 to 11/13 tasks solved purely by changing
  the inference provider. The fastest hosts were the worst ones.
- **Evals rot.** Hand-written eval sets go stale the week after you write them, so most
  teams ship model switches on a demo and a prayer.
- **Experiments burn money invisibly.** One retry loop against a paid API and your
  evaluation budget is gone before the first honest number lands.

Routers pick the best of your options. Nobody proves your options on your traffic.

## What Compound does

One `compound.yaml`, one content-addressed cache, five layers:

| # | Layer | What happens | Status |
|---|---|---|---|
| 1 | **Ingest** | Traces you already produce (Langfuse export, portable JSON) become redacted, provenance-typed eval cases with a sealed decision partition. No eval authoring. | today |
| 2 | **Backtest** | Any model x provider x quant replays against your graded corpus from the cache. Free assertions filter before judge tokens; re-deciding costs $0. | today |
| 3 | **Optimize** | The real [GEPA](https://github.com/gepa-ai/gepa) library evolves a cheaper candidate's prompt on train/val cases, never the sealed set. "No improvement" is a first-class outcome. | today |
| 4 | **Verify live** | Fresh traffic replays against standing candidates; live signals confirm or veto the backtest. | roadmap |
| 5 | **Switch** | The verdict drives the router you already run: staged rollout, audit trail, automatic rollback. | roadmap |

## The payoff

```console
$ compound curate support
  84 traces -> 62 cases · sealed: 10 · train: 40 · val: 12

$ compound gate support --candidate kimi-k3 --reference opus-5 \
      --reason "quarterly cost review"
  reference  opus-5   task_success  0.94
  candidate  kimi-k3  task_success  0.93
  delta = -0.01   95% CI [-0.04, +0.02]   rule: max_regression 0.02

  VERDICT  MEETS GATE  (non-inferior at the declared bar)
  est. savings: $15.00 -> $0.14 /M out · re-decision from cache: $0
```

The gate emits one of five verdicts: **meets gate**, **fails**, **insufficient data**,
**judge abstained**, **no reliable improvement**. If the data cannot support a decision,
Compound says so instead of rounding noise up to a recommendation.

## Why it is easy

- **No eval set to write.** Your production traffic is the corpus; curation surfaces
  what needs review instead of asking you to author test cases.
- **Money-safe by default.** Without `--paid` nothing spends a cent: you get the cost
  estimate. Paid runs require an enabled budget, a hard USD limit, and a per-run `--cap`.
  Every completion is content-addressed and cached, so re-runs and re-decisions are $0.
- **Provider pinning is first-class.** `--openrouter-provider <slug>` isolates one
  serving host with fallbacks disabled, and the *served* host is verified on every call.
- **Your agent can drive it.** The repo ships a
  [`compound-backtest` skill](.claude/skills/compound-backtest/SKILL.md): any coding agent
  runs the loop under the same money rules you would.

## Honest by design

The parts most eval tools skip are the parts that make a verdict trustworthy:

- **A sealed decision partition.** Optimizers, prompt selection, and judge tuning never
  see it; opening it requires a stated `--reason`.
- **Pre-declared rules.** The non-inferiority bar is fixed and content-hashed before
  anyone looks at results; an optimized prompt is re-gated as a new declaration, never a
  quiet edit.
- **Calibration-gated judges.** An LLM judge feeds a gate only after out-agreeing human
  labels (Cohen's kappa with a bootstrap CI). Until then it abstains.
- **Intervals, never bare means.** Paired-bootstrap confidence bounds on every comparison.

## Quickstart

```bash
bun install
bun run packages/cli/src/main.ts import export.jsonl --importer langfuse  # or --importer json
bun run packages/cli/src/main.ts curate <task-key>
bun run packages/cli/src/main.ts experiment <task-key> <model>            # dry run, $0
bun run packages/cli/src/main.ts experiment <task-key> <model> --paid --cap 2.00
bun run packages/cli/src/main.ts gate <task-key> --candidate M --reference M --reason "..."
bun run packages/cli/src/main.ts view compare <task-key>                  # cost/latency/TPS/quality per route
bun run packages/cli/src/main.ts serve                                    # local API on 127.0.0.1:4319
cd packages/dashboard && bun run dev                                      # dashboard on localhost:3000
```

Everything runs locally with no account: SQLite storage, your keys in `.env`
(git-ignored), your traces never leave your machine.

```env
OPENROUTER_API_KEY=
DOUBLEWORD_API_KEY=
```

## The benchmark engine (Python)

`src/compound/` holds the machinery behind our published numbers: pinned-host provider
sweeps and GEPA prompt optimization over live interactive benchmark episodes, with the
same sealed-partition and budget-cap discipline as the product.

```bash
uv sync --extra dev
uv run compound validate-config && uv run pytest -q

# provider sweep: one model, many pinned hosts, full tau-bench protocol
PYTHONPATH=src python -m compound.tau_sweep --estimate     # free cost sheet
PYTHONPATH=src python -m compound.tau_sweep --run          # spends within declared caps

# GEPA: evolve an agent instruction on train/val, one-shot gate on the sealed set
PYTHONPATH=src python -m compound.tau_gepa --estimate
```

tau-bench is the first adapter, not the last: see
[#33](https://github.com/aktasbatuhan/compound/issues/33) for the generic benchmark
adapter interface and Terminal-Bench.

## Mission

Inference is becoming a market: every open model ships on ten hosts within weeks, at
different speeds, prices, and quantizations, and the spread changes monthly. Teams that
treat "which model, which host, which prompt" as a one-time choice overpay and under-serve.

Compound's mission is to make that choice **a measured, repeatable decision on your own
traffic**: backtested like a trading strategy, gated like a deployment, and eventually
automated like both, with a stop-loss. The proof is ambient; the switch is earned.

## Status

Layers 1-3 work today and are tested end to end (`bun test`, `uv run pytest`). Layers 4-5
and the continuous loop are tracked in the
[issues](https://github.com/aktasbatuhan/compound/issues). Expect sharp edges; expect
honest verdicts.

## License

[Apache-2.0](LICENSE)
