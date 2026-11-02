---
title: Introduction
description: What Compound answers and how its parts fit together.
sidebar:
  order: 1
---

Compound answers one question for a team that runs AI in production: if we
moved this traffic to another model or another host, what would it cost, how
would it behave, and would it still do the job? Leaderboards answer a version
of that question for a generic workload, which is rarely yours. Compound
answers it on your own traffic, and says plainly when the evidence is not
strong enough to answer at all.

## How the parts fit

The work follows the order in which a decision gets made, and each step can be
used on its own.

**Bring your traffic.** Compound reads calls from where you already have them:
`compound.trace` exports from Langfuse, plain JSON or OpenTelemetry, request
logs from a gateway, generation ids from OpenRouter or Vercel AI Gateway, or the
[recording SDK](/traffic/sdk/) if you have nothing yet. It reduces every call to
its [shape](/traffic/shape/), meaning its sizes, its timing and the structure of
its prompt, and keeps none of the text, so a team that cannot share prompts can
still share what matters for cost and speed.

**Price it.** The [cost estimate](/price/estimate/) prices that shape on every
model, host and batch tier you name, call by call, with long-prompt and
time-of-day prices applied and the [tokenizer cost of your
language](/price/tokenizers/) counted. It answers with a range, and it says
which numbers are measured and which are assumed.

**Measure it.** A [replay](/measure/replay/) sends traffic with your sizes,
rhythm and prompt structure to real hosts at one, two or five times your load,
and records time to first token, decode speed, errors, rate limits, cache hits
and cost. [Serving metrics](/measure/serving/) and the
[benchmarks](/measure/benchmarks/) measure hosts on controlled prompts and
public task sets.

**Prove it.** A cheaper model is only cheaper if it is good enough.
[Simulated users](/prove/simulate/) replay your real conversations with a
candidate model in place of the production one and grade whether it takes the
same actions, and the [trace pipeline](/prove/traces/) gates single-call tasks
on a sealed set of your own cases.

## Three rules the tool keeps

Nothing that costs money happens by default. Every command that calls a paid
model is a dry run until you add `--go`, and every paid run has a spend cap.

Unknown is never written as zero. A host that does not report cached tokens
shows "not reported", not 0%, and a cost that could not be measured is labelled
as an estimate.

When the evidence is too thin, the answer says so. A simulator that cannot
reproduce your production model's own conversations returns
`simulator_unfaithful`, a judge that disagrees with human labels abstains, and
a small sample returns `insufficient_data` rather than a confident number.

## Two languages, one direction

The engine and the `compound` command are Python. The trace pipeline that
imports, curates and gates single-call tasks is TypeScript today
(`bun run compound`) and is moving into the Python package piece by piece. Both
read the same `compound.yaml`.
