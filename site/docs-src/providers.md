title: Providers
order: 20

# Providers

A provider token names where a model is served, independent of which model.
Every sweep takes a comma-separated list of them in `--providers`, so "same model,
many hosts" is one flag.

## Token forms

| Token | Meaning |
|---|---|
| `openrouter/auto` | OpenRouter's own routing, no pin. Useful as a control arm. |
| `openrouter/<upstream>` | one OpenRouter upstream, fallbacks disabled: `openrouter/deepinfra` |
| `openrouter/<upstream>/<quant>` | the same, restricted to one quantization: `openrouter/deepinfra/fp8` |
| `doubleword/realtime` | Doubleword's realtime tier, addressed directly |
| `doubleword/flex` | Doubleword's flex (queued) tier |
| `direct/<name>` | any OpenAI-compatible host defined under `providers.<name>` in `compound.yaml` |

You do not need to know the upstream slugs. `providers <model>` reads them off
OpenRouter and prints paste-ready tokens:

```bash
compound-bench providers z-ai/glm-5.3-flash
# PROVIDER TOKEN              QUANT   CONTEXT   $IN/M  $OUT/M  STATUS
# openrouter/z-ai/fp8         fp8        1.0M  $0.075  $0.250  up
# openrouter/novita/fp8       fp8        1.0M  $0.075  $0.250  up
# openrouter/deepinfra/fp8    fp8        1.0M  $0.075  $0.250  up
# ...
```

`--json` gives the same list as machine-readable output.

## What pinning does

For an `openrouter/<upstream>` token every request carries

```json
"provider": {"only": ["deepinfra"], "allow_fallbacks": false, "require_parameters": true}
```

so OpenRouter may not silently reroute to another host, and a host that cannot
honor a request parameter (a JSON schema, say) fails fast instead of being
swapped out. For `doubleword/flex` the request carries `service_tier: flex`.

The served host is recorded on every call from the provider echo in the
response, so a pinned run can be checked after the fact rather than trusted.

## Your own host

Any OpenAI-compatible endpoint works without adapter code. For tau2 it is three
flags:

```bash
compound-bench run tau2 --model my-model \
    --provider myhost --api-base http://localhost:8000/v1 --api-key-env MYHOST_API_KEY --go
```

For sweeps, define the host once in `compound.yaml` and use `direct/<name>` as a
token. See [Configuration](../config/).

## The pinning proxy

In-process benchmarks (tau2, mmlu) set the pinning on each request directly.
Third-party harnesses (terminal-bench, Harbor) build their own model calls and
cannot. For those, each host gets its own localhost proxy that:

- forwards to the host's base URL with the host's real key, so the harness only
  sees a placeholder key
- merges the pin into every request body, overriding any routing the harness set
- asks OpenRouter for usage accounting so cost and cached-token counts come back
- optionally pins the reasoning mode (`--reasoning on|off`) and injects
  prompt-cache markers for hosts whose cache is opt-in (`--cache-optin`)
- writes one ledger row per call when a ledger is enabled
- retries 408, 429, and 5xx with backoff, and abandons a call that has not
  completed within 300 seconds so a stalled host cannot hold a trial forever

The proxy works only with agents that run in the harness process and reach
`localhost`. An agent that runs inside the task sandbox cannot see it, and the
CLI refuses to pin such an agent rather than run an unpinned arm by mistake.

## Cache behavior differs by host

OpenRouter's larger upstreams cache prompt prefixes on their own. Doubleword
caches only when a request carries an explicit `cache_control` marker, so a
stock client re-bills the whole growing transcript every agent turn. Each
provider spec carries a `cache_strategy` (`implicit`, `explicit_marker`, or
`none`); `--cache-optin` injects the marker for the hosts that need one. The
ledger's `cached_tokens` column is how you check what actually happened.
