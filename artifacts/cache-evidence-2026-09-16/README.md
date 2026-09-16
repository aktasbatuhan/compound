# Cache evidence, 2026-09-16

Kept deliberately. The previous round measured the same thing and the artifacts
were deleted in a cleanup an hour later, so nothing corroborated the numbers
when a reviewer asked.

- Base commit: `ddb071c` (branch caching-always-on)
- Study spec sha256: `123de4ad9fc30685` (benchmarks/flex-agentic/tier-equivalence.json)
- Endpoint: https://api.doubleword.ai/v1/chat/completions
- Model: deepseek-ai/DeepSeek-V4-Flash-0731, service_tier priority, reasoning_effort medium

## probe-cold-controlled.json  (the one to trust)

Each variant uses a unique nonce prefix, so neither arm can read a cache the
other created. This is the controlled version; the two probes below it were each
confounded in opposite directions and are kept to show why.

| Variant, cold prefix | Call 1 | Call 2 |
|---|---|---|
| Unmarked | 0 cached, 0 written | 0 cached, 0 written |
| With cache_control | 0 cached, **4,929 written** | **4,929 cached** |

The marker is what creates the cache entry. An unmarked request can read an
entry that a marked request already wrote, but never writes one itself, so a run
with no markers anywhere caches nothing at all. `cache_creation_input_tokens` is
populated on the write, which is the field the cost fallback now prices at 1.25x
standard input for a 5 minute TTL.

## probe-marker.json

Four raw usage blocks. Two identical requests per variant; the second of each
pair is the one that could read from cache. Prefix is 7,084 prompt tokens.

| Variant | Call 1 cached | Call 2 cached |
|---|---:|---:|
| No cache_control marker | 0 (0.0%) | 0 (0.0%) |
| With cache_control, ttl 5m | 0 (0.0%) | 7,005 (98.9%) |

Caveat, found by re-running it: this file's two variants share one prefix
string, so ordering decides the answer. Run cold-first it reads 0% unmarked; run
again minutes later the unmarked arm reads 97.6%, because the marked arm had
already written that prefix. Use probe-cold-controlled.json for the claim and
treat this file as the demonstration of the confound.

Denominator is `usage.prompt_tokens`; cached is
`usage.prompt_tokens_details.cached_tokens`.

## e2e/

One episode through the real runner and gateway, not a hand-built request.

- 6 agent calls, 48,589 prompt tokens, 43,984 cached = **90.5%**
- `cache_requested` true on every agent call
- `cost_kind` = `derived_from_rate_card` on every call
- Agent cost $0.0028 for the episode

Denominator is the sum of `prompt_tokens` over agent calls with status 200.

## What this does NOT show

The episode ended `infrastructure_error` with `error_type: FileNotFoundError`
after 634.97s, and its `worker.log` is absent from the episode directory. A
probe comparing `reasoning_effort: medium` against omitting it shows medium is
not slower (0.9s versus 2.6s on a trivial prompt), so the duration is not
explained by the reasoning control that this branch restored. **This failure is
undiagnosed and must be resolved before another paid study runs.** The caching
numbers above stand on their own: they come from calls that returned 200 with
usage recorded.
