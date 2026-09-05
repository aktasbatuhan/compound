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
| `openrouter/<upstream>/<quant>` | the same host pin, with a discovery label for quantization: `openrouter/deepinfra/fp8` |
| `doubleword/realtime` | Doubleword's realtime tier, addressed directly |
| `doubleword/flex` | Doubleword's flex (queued) tier |
| `direct/<name>` | any host defined under `providers.<name>` in `compound.yaml`: OpenAI-compatible by default, or Anthropic's Messages API with `type: anthropic` |

`compound.yaml` ships `direct/openai` (standard tier), `direct/openai-flex`
(the queued half-price tier, `service_tier: flex` on every call), `direct/anthropic`,
`direct/telnyx` and `direct/zai`. First-party hosts name the same weights
differently, so a mixed grid passes one id per host:

```bash
compound-bench serving --providers direct/openai,direct/openai-flex,direct/anthropic \
    --shapes profiles.json \
    --host-model openai=gpt-5.4-mini --host-model openai-flex=gpt-5.4-mini \
    --host-model anthropic=claude-sonnet-5 --go
```

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

The `STATUS` column is OpenRouter's own belief about the endpoint, which is not
the same claim as "will serve me right now". Without your own upstream key you
sit on OpenRouter's shared rate-limit pool for that host, where an endpoint
listed as up can return 429 on every call for one model while serving another
fine. `--probe --go` tests it by sending one small pinned call to each host:

```bash
compound-bench providers z-ai/glm-5.3-flash --probe --go
# PROVIDER TOKEN                  STATUS  SECONDS  DETAIL
# openrouter/z-ai/fp8                200      1.4  Z.AI
# openrouter/deepinfra/fp8           429      0.5  {"error":{"message":"Provider returned error"...
# ...
# 17 of 23 answered. Only these can carry an arm right now.
```

Measured on 2026-09-03, six of the twenty-three hosts listed as up for that
model returned 429. Probe before committing a sweep to a host list.

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
OpenAI echoes the tier that served each call the same way (`service_tier_echo`
in the ledger, next to `service_tier_requested`), so a flex arm that was quietly
served on the default tier shows up as such. When flex has no capacity it
returns a 429 and does not charge; the harness records that as a failure rather
than retrying, because that rate is the tier's reliability. Flex can also queue
longer than the harness timeout before its first byte, so the provider block
declares `timeout_s: 900`, OpenAI's own recommendation.

## Anthropic is measured on its native API

Anthropic offers an OpenAI-compatible endpoint, and the harness deliberately
does not use it. That layer has no prompt caching, returns empty token details,
ignores `service_tier` and caps temperature at 1. A comparison run through it
would score Anthropic at 0% cache for the same reason an unmarked call scores
Doubleword at 0%: the measurement, not the host. `type: anthropic` switches the
serving harness to `/v1/messages`, where cache reads and writes come back per
call and the same `cache_control` marker Doubleword needs is the one Anthropic
needs. Two consequences to read the numbers with:

- Current Claude models reject sampling parameters, so no temperature is sent
  and the ledger records `temperature: null` for that route. The agreement
  analysis sets such a route aside instead of counting it as a host that
  failed to reproduce itself.
- The pinning proxy speaks chat completions only and refuses this host rather
  than forward it through the lossy layer, so Anthropic runs in
  `compound-bench serving` and not behind a third-party harness.

Quantization suffixes such as `/fp8` are labels from discovery. Routing strips
the suffix and pins the host only; a matching provider echo does not verify
the quantization used for that call.

## Cost: measured or derived

OpenRouter returns the cost of every call, and that number is used as is.
OpenAI, Anthropic and Telnyx return none, so their cost is priced from the
rate cards under `serving_rates_usd_per_million_tokens` in `compound.yaml`
(input, cached input, cache write where billed, output) and shown with a `~`
prefix wherever it appears. Doubleword's cost comes from differencing its
billing meter. A derived figure is only as good as its rate card: the Telnyx
card was checked against a bill (920 calls priced at it summed to within 0.2%
of the dashboard), the OpenAI and Anthropic cards carry the date they were
copied from the vendor's pricing page.

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
  prompt-cache markers for hosts whose cache is opt-in (on by default,
  `--no-cache-optin` to disable)
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
`none`); the marker is injected for the hosts that need one **by default**, because
the alternative is re-billing the whole growing transcript every agent turn. Pass
`--no-cache-optin` when the unmarked path is what you mean to measure. The
ledger's `cached_tokens` column is how you check what actually happened.
