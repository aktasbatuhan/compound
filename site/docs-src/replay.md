title: Workload replay
order: 67

# Workload replay

The [cost estimate](../estimate/) says what your traffic should cost on a
host. `compound replay` checks it. It sends traffic with the same sizes, the
same rhythm and the same prompt-prefix structure as yours, without your text,
and measures what each host does with it: time to first token, decode speed,
errors, rate limits, cache hits and cost per call.

```bash
# see the plan first; nothing is sent without --go
compound replay shapes.jsonl --providers openrouter/deepinfra/fp8,openrouter/novita/fp8 \
    --model-or deepseek/deepseek-v4.1-flash --load 1,2,5 --window-minutes 10

compound replay shapes.jsonl ... --catalog prices.json --max-usd 5 --go
```

It reads anything [`compound shape`](../shape/) reads, so a shape export or
the SDK's records are enough. The prompts are never needed.

## How the traffic is rebuilt

Every original message becomes one message of real prose in the call's
language, sized to that message's share of the original input tokens. The
prose comes from FLORES-101, so it tokenizes like real Turkish or English
rather than like lorem ipsum. The text for each message is chosen from its
prefix hash, which means calls that shared a prompt prefix in production share
the exact same bytes in the replay, and a host's cache sees the reuse it would
have seen. A random value unique to each run goes into that choice too, so one
replay never warms the cache for the next.

Calls leave on your original schedule, divided by the load factor, whether
or not earlier calls have finished. That is how rate limits are met in real
life, and it is why the concurrency in a replay comes from your traffic and
not from a worker count. `--window-minutes` replays your busiest stretch
rather than the quiet start of a log. Each call is sent to every route at the
same moment, so hosts are compared at the same time of day.

Output length cannot be forced. Each call asks for an answer of about the
original length and caps `max_tokens` there, and the summary's size column
shows how close the synthetic prompts came to the original token counts.

## Money

A replay spends real money, so it is a dry run unless you pass `--go`, and
`--max-usd` caps it. The cap counts calls still in flight: each call reserves
the most it could cost (its input and full output budget at the listed price)
before it is sent, and settles to the reported cost when it returns. That needs
prices, so pass `--catalog` from `compound prices`. A route the catalog cannot
price reserves the average cost of finished calls instead, which a first burst
can overshoot, and the command says when that applies.

## Reading the result

The summary has one row per route and load factor: calls that worked, 429s
and other errors kept apart, time to first token at the median and 90th
percentile, decode speed, the share of input served from cache, and cost per
call. The profile's prediction of what a cache could serve is printed under
it, so the estimate and the measurement sit side by side.
