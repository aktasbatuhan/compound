---
title: Quickstart
description: Install Compound and get a first answer in a few minutes.
sidebar:
  order: 2
---

## Install

```bash
git clone https://github.com/aktasbatuhan/compound
cd compound
uv sync --extra dev
```

Keys go in a git-ignored `.env` at the repo root. Nothing on this page needs
one; the commands that spend money ask for theirs when you add `--go`.

```env
OPENROUTER_API_KEY=
```

`compound` reads `.env` itself, so there is no need to `source` it.

## 1. Price a workload from four numbers

If you know roughly how many calls you make a month and how long they are, you
can get a first estimate before sharing any data. This fetches public prices
from OpenRouter and makes no model calls.

```bash
compound estimate --current openai/gpt-6-sol \
    --candidates deepseek/deepseek-v4.1-flash,z-ai/glm-5.3,qwen/qwen3.8-flash \
    --monthly-calls 1000000 --input-tokens 3500 --output-tokens 1000 \
    --cached-share 0.2 --turkish-share 0.7
```

You get today's monthly cost and a range for every place each candidate can be
bought, cheapest first, with the lever each row pulls and a list of what the
estimate does not model.

## 2. Profile your real traffic

Point `compound shape` at whatever logs you have. It reads trace exports,
gateway request logs and call ledgers, and it keeps no prompt text.

```bash
compound shape logs/ --out profile.json --calls-out shapes.jsonl
```

The profile shows call volume, token distributions, how bursty the traffic is,
peak concurrency, and how much of each prompt a cache could serve. Feed the
same logs to the estimate to replace the four numbers with your real calls:

```bash
compound estimate shapes.jsonl --current openai/gpt-6-sol \
    --candidates deepseek/deepseek-v4.1-flash,z-ai/glm-5.3
```

## 3. Plan a replay

A replay sends your traffic's shape to real hosts. Without `--go` it only
prints the plan.

```bash
compound replay shapes.jsonl \
    --providers openrouter/deepinfra/fp8,openrouter/novita/fp8 \
    --model-or deepseek/deepseek-v4.1-flash --load 1,2,5 --window-minutes 10
```

Add `--go` and a `--max-usd` cap when you are ready to spend.

## Where to go next

- No logs yet? Start recording with the [SDK](/traffic/sdk/).
- Using OpenRouter or Vercel AI Gateway? [Import from the gateway](/traffic/gateways/).
- Choosing between models for conversations with tools? Use [simulated users](/prove/simulate/).
- Every flag is in the [CLI reference](/reference/cli/).
