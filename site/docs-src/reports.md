title: Reports and ledgers
order: 50

# Reports and ledgers

A run leaves two kinds of evidence: episode results from the harness, and, when
enabled, one row per model call. Episode results tell you who passed. The
ledger tells you what each call cost, how long it took, whether the cache hit,
and which host actually answered.

## `bench_report`

```bash
PYTHONPATH=src python -m compound.bench_report artifacts/dsflash \
    --prices doubleword-flex=0.70,2.25 --prices doubleword-realtime=0.93,3.00
```

Reads the per-host episode dumps a `--providers` sweep writes and emits, under
`<run>/report/`:

| File | Contents |
|---|---|
| `episodes.csv` | one row per episode: host, task, trial, reward, tokens, cost, latency, served-by |
| `per_task.csv` | per host per task: context size, trials, solved, rate |
| `summary.json` | per host: accuracy, cost per task, latency, tokens per second, served-by |
| `transcripts.jsonl` | full ordered messages per episode |
| `charts.html` | success vs context size, cost vs quality |

### Where cost comes from

- OpenRouter routes: the per-call `usage.cost` OpenRouter reports. Measured.
- Doubleword: either the billed amount from the `dw` CLI for the run window (`--dw-model`, `--dw-usage-since`), or token counts times the `--prices` rate card you pass. Derived, and the report labels it so.
- Direct hosts: `--prices label=in,out` in USD per million tokens.

## The call ledger

`--call-ledger PATH` on `run`, or `--ledger-dir` on `harbor`, records one JSON
line per model call:

| Field | Meaning |
|---|---|
| `route` / `upstream` | route label and requested host; `provider_echo` separately names the responding host |
| `pin_honored` | whether the echo matches the host pin; `null` when unpinned or no echo was returned |
| `status`, `error`, `latency_ms` | HTTP status, error class (`hang_timeout`, ...), wall time |
| `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens` | from the usage block; `null` when the host did not report it |
| `cost_usd` | the host's own accounting; `null` when not reported |
| `abandoned` | true when the call returned 200 but no usage block arrived, so any charges are unknown |
| `reasoning_pin`, `cache_marked` | what the proxy injected |

Host verification does not verify quantization. Summaries count verified,
unverified, and violated host pins separately. Cost totals cover only calls
that report cost; missing counts identify partial subtotals. Cache-hit rates
use only calls reporting both prompt and cached tokens in their denominator.

Recording must be enabled before the run. Proxy write failures are warnings,
and malformed ledger rows now produce warnings when read. A ledger alone
cannot prove that no calls were lost: check the run logs and reconcile counts
with the harness before making per-call claims. A warning about missing rows
means the evidence is incomplete. The JSONL ledger is not a spend limiter.

A `null` means the host did not tell us. It is not the same claim as a measured
zero, and the tooling keeps the two apart.

```bash
compound-bench ledger artifacts/tb/calls.jsonl           # per-route rollup
compound-bench ledger artifacts/tb/calls.jsonl --hosts   # plus which upstreams answered
```

## Comparing arms

`scripts/analyze_arms.py <dir>` reads every ledger under a directory and prints,
per arm: calls, the share that never completed (with a Wilson 95% interval),
abandoned calls, cost, cost per prompt token, cache-hit share, p50 and p90
latency, and the number of distinct hosts that answered. It then tests each
arm against `openrouter/auto` with a two-proportion z-test, Holm-corrected
across the arms.

It also says whether the arms ran concurrently or one after another. When they
ran sequentially, it prints a warning: differences are then confounded with
time of day.

Cost figures are lower bounds on any arm with abandoned calls. The cache column
is the host's self-reported ratio; read it next to cost per token, not instead
of it.
