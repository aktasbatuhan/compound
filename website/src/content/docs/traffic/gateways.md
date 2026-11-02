---
title: From a gateway
sidebar:
  order: 2
---

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
