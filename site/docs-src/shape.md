title: Workload shape
order: 65

# Workload shape

Before you compare hosts, you need to know what your traffic looks like. How
many calls a month, how long the prompts and answers are, how bursty the
arrivals are, how many calls run at once, and how much of each prompt repeats
so a cache could serve it. `compound shape` answers that from logs you already
have, and it never keeps the text of a prompt or an answer.

```bash
compound shape traces/ --out profile.json --calls-out shapes.jsonl
```

It reads four kinds of input, and one folder may mix them:

- `compound.trace` documents, which `bun run compound import` writes from
  Langfuse, plain JSON or OpenTelemetry exports
- request logs, one JSON object per call with `request` (model, messages,
  tools), `response` (with `usage`) and a timestamp, which most gateways and
  loggers can produce
- Compound call ledgers from `--call-ledger`, which have sizes and timing but
  no prompts
- shape files written by an earlier `--calls-out`, or by the SDK

Anything else is counted as skipped and left out, never guessed at.

## From a gateway

OpenRouter and Vercel AI Gateway cannot list your calls through their APIs,
but both can look one up by its generation id, which every response carries
in its `id` field. Give `compound gateway-import` a file of those ids, or the
request logs that hold them, and it writes shape records with the gateway's
own timing, the provider's native token counts (cached and reasoning
included) and cost:

```bash
compound gateway-import ids.txt --gateway openrouter --content --out shapes.jsonl
compound gateway-import ids.txt --gateway vercel --out shapes.jsonl
```

With `--content`, OpenRouter also returns the stored prompt when the account
has input and output logging turned on. The importer uses it only to build
the prefix chain, roles and language hint, then drops it, so the output holds
no text. Without stored prompts the records still drive the cost estimate,
but a replay needs the prefix structure.

If you have no ids at all, `compound gateway-activity` reads OpenRouter's
daily totals per model for the last 30 days and prints the `compound
estimate` command for each model, with its monthly calls, average sizes and
cached share filled in. OpenRouter only shows account activity to a
management key.

## What the profile says

The terminal shows a short reading, and `--out` writes the full profile as
JSON. It covers token distributions (input, output, reasoning), latency,
arrivals per minute with the busiest minute, peak concurrency, and a monthly
projection. The projection appears only when the data covers at least a day,
and it states the basis it was scaled from.

It reports two cache numbers and keeps them apart. **Cache served today** is
what your current host said it served from cache. **Cache could serve** is
what the prompts themselves allow: the longest prefix each call shares with an
earlier call within the cache lifetime, counted only when it reaches the
minimum length a host caches. When the second number is much larger than the
first, the usual cause is prompt layout or a missing cache marker, and that is
money left on the table.

The cache assumptions are flags, because hosts differ:

```bash
compound shape logs.jsonl --cache-ttl 3600 --min-prefix-tokens 2048
```

## Sharing a shape without sharing your data

`--calls-out` writes one record per call with counts, timing, a language hint
and a chain of prefix hashes. The hashes are salted with a random value that
exists only while the command runs, so nobody can recover a common system
prompt by hashing guesses at it. The cost is that hashes only compare within
one export. That file is what we ask for when a team cannot send prompts: it
is enough to estimate cost and to replay the traffic at the same size and
rhythm, and it holds no words from the conversation.

The language hint is `tr` when Turkish-only letters make up at least 1% of
the letters, `latin` for other Latin-script text and `other` for the rest. It
picks a tokenizer ratio for cost estimates and is not language detection.
