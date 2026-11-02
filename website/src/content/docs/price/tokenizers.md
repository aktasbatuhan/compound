---
title: Tokens per language
description: How many tokens the same meaning costs in each language on each model, as the estimate uses it.
sidebar:
  order: 2
---

A price per token only means something once you know how many tokens your
text becomes, and that depends on the model and on the language. The cost
estimate moves your token counts from one model to another with ratios
measured on FLORES-101, a set of 1,012 sentences professionally translated so
that every language says exactly the same thing.

## Turkish

On the English sentences almost every tokenizer we measured needs between
26,873 and 28,107 tokens, while Claude needs 42,908, about 60% more for the
same English text. The Turkish version of the same sentences spreads them much
further apart:

| Tokenizer | English tokens | Turkish tokens | Turkish ÷ English |
|---|---:|---:|---:|
| Mistral Large 4 | 27,831 | 36,808 | 1.32 |
| Gemini 4 Argon | 28,051 | 37,409 | 1.33 |
| Gemini 3.8 Flash | 28,107 | 38,134 | 1.36 |
| Gemma 4 | 27,114 | 37,142 | 1.37 |
| GPT-6, GPT-5.4 mini, gpt-oss (o200k) | 26,878 | 38,445 | 1.43 |
| Qwen 3.8 | 27,530 | 39,478 | 1.43 |
| Mistral Small 4 | 27,827 | 41,493 | 1.49 |
| GLM-5.3 | 27,185 | 44,714 | 1.64 |
| Claude Opus, Sonnet and Haiku 5.5 | 42,908 | 72,931 | 1.70 |
| DeepSeek V4 | 27,041 | 55,167 | 2.04 |

Each ratio has a 95% interval of about ±0.02 (±0.03 for the closed models,
which are counted in chunks). A model's answer is tokenized the same way, so
the ratio applies to output as well as input, and it can change which model is
cheapest. Per unit of meaning, DeepSeek V4.1 Flash's Turkish output costs more
than Mistral Small 4's and gpt-oss's, although its listed price per token is
lower. And Claude Sonnet 5.5, GPT-6 Sol and Gemini 4 Argon all list $2 per
million input and $10 per million output tokens, but the same Turkish answer
costs about $14 per million tokens of meaning on GPT and Gemini and about $27
on Claude.

## Arabic, Azerbaijani, Kazakh and Uzbek

The same set covers these languages for the open tokenizers. Arabic runs from
1.24 to 1.77 times English, Azerbaijani from 1.40 to 2.50, Kazakh from 1.33 to
2.71 and Uzbek from 1.43 to 2.14, and the full table is in the repository.

## What is measured and what is not

Open models are counted locally with their published tokenizers, and GPT models
with OpenAI's free token counter, which agreed with the public o200k tokenizer
to within 5 tokens out of 26,878. Claude and Gemini do not publish their
tokenizers, so they are counted from the native prompt tokens their providers
bill, sent through OpenRouter in fixed chunks, in English and Turkish only. A
spot check on Claude Haiku 5.5 matched the charged cost to the token. A model
nobody has measured is treated as falling somewhere within the range of the
tokenizers we did measure, and the estimate widens its answer accordingly.

FLORES is news and encyclopedia prose. Chat messages, legal text and
code-mixed text may tokenize differently, so for a decision that matters,
count a sample of your own messages with each candidate's tokenizer.

The data, including every per-sentence count, and the scripts that produce it
are in [`benchmarks/tokenizers/`](https://github.com/aktasbatuhan/compound/tree/main/benchmarks/tokenizers).
