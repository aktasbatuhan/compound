title: Configuration
order: 80

# Configuration

`compound.yaml` at the repo root is shared by both halves. `compound-bench` reads
the `providers`, `budget`, and `models` sections; the trace pipeline reads the
rest. `bun run compound init` writes a starter file and `compound validate`
checks it.

## Providers

```yaml
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    type: openai_compatible
  doubleword:
    base_url: https://api.doubleword.ai/v1
    api_key_env: DOUBLEWORD_API_KEY
    type: flex
  myhost:                                   # any OpenAI-compatible server
    base_url: http://localhost:8000/v1
    api_key_env: MYHOST_API_KEY
    type: openai_compatible
    cache_strategy: none                    # implicit | explicit_marker | none
```

A block named `myhost` is addressable as the provider token `direct/myhost` in
any `--providers` list, and sits in the same report as the OpenRouter routes.

## Budget

```yaml
budget:
  paid_runs_enabled: true
  hard_limit_usd: 25.00
  smoke_cases_per_benchmark: 2
```

`bfcl` and `ds1000` enforce this limit and the per-run `--cap`. `tau2` bills
whatever the episodes cost, so size the subset with a dry run first. The trace
pipeline enforces the same limit with atomic reservations; see
[Money controls](../traces/#money-controls).

## Models

```yaml
models:
  candidates:
    - id: gpt-4o-mini
      provider: openai
      provider_ids:                # same weights, different id per host
        openrouter: openai/gpt-4o-mini
```

`provider_ids` lets one logical model keep a single identity in storage while
each host receives the id it expects.

## Environment variables

| Variable | Effect |
|---|---|
| `OPENROUTER_API_KEY`, `DOUBLEWORD_API_KEY` | credentials, read from `.env` |
| `COMPOUND_REASONING` (`on` or `off`) | pin reasoning mode for terminal-bench runs; the `--reasoning` flag wins when given |
| `COMPOUND_DW_CACHE=0` | turn prompt-cache markers **off** for opt-in hosts (they are on by default) |
| `COMPOUND_TB_TIMEOUT_MULT=N` | extended-limits mode for terminal-bench; wins over the flag |
| `COMPOUND_CALL_TIMEOUT` | the proxy's hang ceiling per call in seconds, default 300; 0 disables it |
| `COMPOUND_CALL_LEDGER` | ledger path; set by the CLI when you pass `--call-ledger` or `--ledger-dir` |

## Provider fields for serving comparisons

| Field | Meaning |
|---|---|
| `type` | `openai_compatible` (default) or `anthropic` for a host on the Messages API |
| `service_tier` | forwarded in every request body; OpenAI `flex`, Doubleword `flex` |
| `cache_strategy` | `implicit`, `explicit_marker` or `none`; decides whether a `cache_control` marker is injected |
| `timeout_s` | per-call timeout this host needs, when longer than the harness default (OpenAI flex: 900) |

`serving_rates_usd_per_million_tokens` maps route label to model id to a rate
card (`input`, `cached_input`, `output`, optional `cache_write`). It is only
consulted for hosts that return no per-call cost, and anything priced from it
is reported as derived.
