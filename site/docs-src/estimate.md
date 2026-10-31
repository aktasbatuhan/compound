title: Cost estimate
order: 66

# Cost estimate

`compound estimate` prices a workload on other models and hosts without making
a single call. Give it your logs, or four numbers, and it tells you what you
pay today and what each alternative would cost a month, as a range.

```bash
# from logs (see Workload shape for the formats it reads)
compound estimate traces/ --current openai/gpt-6-sol \
    --candidates deepseek/deepseek-v4.1-flash,z-ai/glm-5.3 --monthly-calls 1000000

# from a few numbers
compound estimate --current openai/gpt-6-sol --candidates z-ai/glm-5.3 \
    --monthly-calls 1000000 --input-tokens 3500 --output-tokens 1000 \
    --cached-share 0.2 --reuse-share 0.6 --turkish-share 0.7
```

Each row is one place to buy a model: the model's own listing, a pinned host,
or a batch tier. Rows say which lever they pull. **Same model, other host**
and **batch tier** keep the model you use today. **Switch model** rows are
labelled "quality not yet measured", because a cheaper model is only cheaper
if it is good enough, and that takes an eval on your own traffic.

## How a call is priced

Every call is priced on its own, with the prices that apply to its size and
its time. Some listings charge more above a prompt size, and some hosts charge
less at certain UTC hours or on weekends. Both are applied per call, so traffic
that runs at night is priced as night traffic.

A call costs its uncached input at the input price, its cached input at the
cache price, and its output at the output price. A host that lists no cache
price gets no cache discount.

## Why the answer is a range

Two things are uncertain until we measure them on your traffic, and the range
shows both.

**Caching.** The cheap end assumes the host caches as much as your prompts
allow, which is the "cache could serve" share from the workload profile. The
expensive end assumes only what your current host already serves.

**Tokens.** The same text is a different number of tokens on different
models, and Turkish varies far more than English. On 1,012 FLORES sentences
with the same meaning in both languages, every tokenizer we measured needs
26,900 to 27,800 tokens for the English, but the Turkish takes 36,800 on
Mistral Large 4, 38,400 on GPT-6 and 55,200 on DeepSeek V4. The estimate
moves your token counts with these measured ratios, per language. GPT models
are measured through OpenAI's token counter; other OpenAI models fall back to
`o200k_base`, which matched those to within a few tokens. Claude and Gemini do
not publish their tokenizers and are not measured yet, so for them the range
covers every tokenizer we measured instead of assuming a match.

The ratios come from `scripts/measure_tokenizers.py`, which records the
dataset revision and every per-sentence count, and
[`benchmarks/tokenizers/`](https://github.com/aktasbatuhan/compound/tree/main/benchmarks/tokenizers)
has the full tables, including Arabic, Azerbaijani, Kazakh and Uzbek.

## What it does not model yet

The result lists these every time:

- how much a different model reasons on your workload; output, including
  reasoning, is assumed to scale with the tokenizer ratio
- cache write fees
- retries and failed calls
- whether a cheaper model is good enough

## Prices

Prices come from OpenRouter's public listings. `compound prices` saves a dated
snapshot, with per-host prices for the models you name, and `--catalog` pins
an estimate to it, so the same inputs give the same answer later:

```bash
compound prices --hosts openai/gpt-6-sol,z-ai/glm-5.3 --out prices.json
compound estimate ... --catalog prices.json
```

Without `--catalog`, the estimate fetches current prices. A snapshot from
31 October 2026 is kept in `benchmarks/prices/`.
