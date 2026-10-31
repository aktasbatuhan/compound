"""Measure what the same meaning costs in tokens, per language and per model.

The corpus is FLORES-101 devtest (gsarti/flores_101, CC BY-SA 4.0): 1,012
sentences professionally translated into each language, so English and
Turkish rows say the same thing. That makes the headline number fair: for one
sentence's meaning, how many tokens does Turkish cost against English on each
model. The revision used is recorded, and so is every per-sentence count.

Open models are counted locally from their published ``tokenizer.json``
(free). Closed models (OpenAI, Anthropic, Google) are counted from the usage
OpenRouter reports: the corpus is sent in fixed chunks with a one-token reply,
and a near-empty baseline call per model removes the chat template overhead.
That costs cents, is capped by ``--max-usd``, and is skipped without
``--closed``.

Outputs:

- ``--study``: per-model token totals per language, the premium over English
  with a 95% bootstrap interval (resampling sentences, or chunks for closed
  models), and per-sentence counts for open tokenizers.
- ``--table``: tokens per 1,000 characters per language, the table
  ``compound estimate`` uses to move token counts between models.

    uv run --with tokenizers python scripts/measure_tokenizers.py \\
        --hf deepseek-ai/DeepSeek-V4.1-Flash,zai-org/GLM-5.3 \\
        --closed openai/gpt-6-sol,anthropic/claude-sonnet-5.5 --env-file .env \\
        --study benchmarks/tokenizers/flores-premium.json \\
        --table src/compound/estimate/data/tokenizers.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "gsarti/flores_101"
SPLIT = "devtest"
LANGS = {"en": "eng", "tr": "tur", "ar": "ara", "az": "azj", "kk": "kaz", "uz": "uzb"}
#: Languages closed models are measured on. The rest are open tokenizers only,
#: to keep the paid part to cents.
CLOSED_LANGS = ("en", "tr")
PARQUET_URL = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_COUNT = "https://api.openai.com/v1/responses/input_tokens"
CHUNKS = 20
BOOTSTRAP = 2000
AGENT = "compound-tokenizer-study/1.0 (github.com/aktasbatuhan/compound)"


def _get(url: str, headers: dict | None = None, body: bytes | None = None) -> dict:
    req = urllib.request.Request(url, data=body, headers={"User-Agent": AGENT, **(headers or {})})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def dataset_revision() -> str:
    return _get(f"https://huggingface.co/api/datasets/{DATASET}")["sha"]


def fetch_corpus() -> dict[str, list[str]]:
    """Sentences per language, aligned by FLORES id, from the Hub's Parquet export."""
    import io

    import pyarrow.parquet as pq

    corpus = {}
    for lang, config in LANGS.items():
        (url,) = _get(PARQUET_URL.format(dataset=DATASET, config=config, split=SPLIT))
        req = urllib.request.Request(url, headers={"User-Agent": AGENT})
        with urllib.request.urlopen(req, timeout=120) as response:
            table = pq.read_table(io.BytesIO(response.read()), columns=["id", "sentence"])
        rows = dict(zip(table["id"].to_pylist(), table["sentence"].to_pylist(), strict=True))
        corpus[lang] = [rows[i] for i in sorted(rows)]
    sizes = {len(v) for v in corpus.values()}
    if len(sizes) != 1:
        raise SystemExit(f"languages are not aligned: {sizes}")
    return corpus


def ratio_interval(num: list[int], den: list[int], seed: int = 7) -> tuple[float, float, float]:
    """Ratio of sums with a paired bootstrap 95% interval over units."""
    rng = random.Random(seed)
    n = len(num)
    point = sum(num) / sum(den)
    stats = []
    for _ in range(BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(sum(num[i] for i in idx) / sum(den[i] for i in idx))
    stats.sort()
    return point, stats[int(0.025 * BOOTSTRAP)], stats[int(0.975 * BOOTSTRAP) - 1]


def summarize(counts: dict[str, list[int]], corpus: dict[str, list[str]]) -> dict:
    out = {"tokens": {lang: sum(v) for lang, v in counts.items()}, "premium_vs_en": {},
           "tokens_per_1k_chars": {}}
    for lang, values in counts.items():
        chars = sum(len(s) for s in corpus[lang])
        out["tokens_per_1k_chars"][lang] = round(sum(values) / chars * 1000, 2)
        if lang != "en":
            point, low, high = ratio_interval(values, counts["en"])
            out["premium_vs_en"][lang] = {"ratio": round(point, 4), "ci95": [round(low, 4),
                                          round(high, 4)]}
    return out


def measure_open(repo: str, corpus: dict[str, list[str]]) -> dict[str, list[int]]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(repo)
    return {lang: [len(e.ids) for e in tok.encode_batch(sents, add_special_tokens=False)]
            for lang, sents in corpus.items()}


class Budget:
    def __init__(self, max_usd: float) -> None:
        self.max_usd, self.spent = max_usd, 0.0

    def charge(self, usd: float | None) -> None:
        self.spent += usd or 0.0
        if self.spent > self.max_usd:
            raise SystemExit(f"stopped: spent ${self.spent:.4f}, over the ${self.max_usd} cap")


def _prompt_tokens(model: str, text: str, key: str, budget: Budget) -> int:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 16,
        "reasoning": {"effort": "minimal", "exclude": True},
        "usage": {"include": True},
    }).encode()
    data = _get(OPENROUTER, {"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"}, body)
    usage = data.get("usage") or {}
    budget.charge(usage.get("cost"))
    if not usage.get("prompt_tokens"):
        raise RuntimeError(f"{model}: no prompt_tokens in usage: {data}")
    return int(usage["prompt_tokens"])


def _openai_input_tokens(model: str, text: str, key: str) -> int:
    """OpenAI's free input-token count for a Responses request."""
    body = json.dumps({"model": model, "input": text}).encode()
    data = _get(OPENAI_COUNT, {"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"}, body)
    return int(data["input_tokens"])


def measure_chunks(count, corpus: dict[str, list[str]]) -> dict[str, list[int]]:
    """Token counts per chunk, with the request template overhead removed."""
    overhead = count(".") - 1
    n = len(corpus["en"])
    bounds = [round(i * n / CHUNKS) for i in range(CHUNKS + 1)]
    out = {}
    for lang in CLOSED_LANGS:
        out[lang] = []
        for a, b in zip(bounds, bounds[1:], strict=False):
            out[lang].append(count("\n".join(corpus[lang][a:b])) - overhead)
    return out


def _read_key(env_file: str | None, name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    if env_file:
        for line in Path(env_file).read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"this needs {name} (or --env-file)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf", default="", help="comma-separated Hugging Face repo ids")
    parser.add_argument("--tiktoken", default="", help="comma-separated tiktoken encodings")
    parser.add_argument(
        "--openai", default="", help="OpenAI model ids, counted with the free input_tokens API"
    )
    parser.add_argument("--closed", default="", help="comma-separated OpenRouter model ids")
    parser.add_argument("--env-file", help="file holding OPENROUTER_API_KEY")
    parser.add_argument("--max-usd", type=float, default=2.0, help="cap on paid calls")
    parser.add_argument("--study", required=True, help="where to write the study JSON")
    parser.add_argument("--table", help="where to write the estimate's tokenizer table")
    args = parser.parse_args()

    revision = dataset_revision()
    corpus = fetch_corpus()
    chars = {lang: sum(len(s) for s in v) for lang, v in corpus.items()}
    print(f"FLORES-101 {SPLIT} @ {revision[:10]}: {len(corpus['en'])} sentences x "
          f"{len(corpus)} languages; characters {chars}")

    results: dict[str, dict] = {}
    for repo in filter(None, (r.strip() for r in args.hf.split(","))):
        try:
            counts = measure_open(repo, corpus)
        except Exception as exc:  # gated, missing tokenizer.json, network
            print(f"skip {repo}: {str(exc).splitlines()[0]}", file=sys.stderr)
            continue
        results[repo] = {"kind": "open", "unit": "sentence", "n": len(corpus["en"]),
                         **summarize(counts, corpus), "per_sentence": counts}
        print(f"{repo}: tr/en {results[repo]['premium_vs_en']['tr']['ratio']:.3f}")

    for encoding in filter(None, (e.strip() for e in args.tiktoken.split(","))):
        import tiktoken

        enc = tiktoken.get_encoding(encoding)
        counts = {lang: [len(enc.encode(s)) for s in sents] for lang, sents in corpus.items()}
        name = f"tiktoken:{encoding}"
        results[name] = {"kind": "open", "unit": "sentence", "n": len(corpus["en"]),
                         **summarize(counts, corpus), "per_sentence": counts}
        print(f"{name}: tr/en {results[name]['premium_vs_en']['tr']['ratio']:.3f}")

    sub = {lang: corpus[lang] for lang in CLOSED_LANGS}

    def record_closed(name: str, count) -> None:
        try:
            counts = measure_chunks(count, corpus)
        except (RuntimeError, urllib.error.HTTPError, KeyError) as exc:
            print(f"skip {name}: {exc}", file=sys.stderr)
            return
        results[name] = {"kind": "closed", "unit": "chunk", "n": CHUNKS,
                         **summarize(counts, sub), "per_chunk": counts}
        print(f"{name}: tr/en {results[name]['premium_vs_en']['tr']['ratio']:.3f}")

    for model in filter(None, (m.strip() for m in args.openai.split(","))):
        key = _read_key(args.env_file, "OPENAI_API_KEY")
        record_closed(
            f"openai:{model}", lambda text, m=model, k=key: _openai_input_tokens(m, text, k)
        )

    budget = Budget(args.max_usd)
    closed = [m.strip() for m in args.closed.split(",") if m.strip()]
    if closed:
        key = _read_key(args.env_file, "OPENROUTER_API_KEY")
        for model in closed:
            record_closed(f"openrouter:{model}",
                          lambda text, m=model: _prompt_tokens(m, text, key, budget))
            print(f"  spent so far ${budget.spent:.4f}")

    study = {
        "schema": "compound.tokenizer_study",
        "schema_version": 1,
        "measured_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "corpus": {"dataset": DATASET, "revision": revision, "split": SPLIT,
                   "license": "CC BY-SA 4.0", "sentences": len(corpus["en"]),
                   "languages": LANGS, "characters": chars},
        "method": {
            "open": "tokenizer.json from the Hugging Face repo, no special tokens, per sentence",
            "closed": f"OpenRouter usage.prompt_tokens over {CHUNKS} fixed chunks of newline-"
                      "joined sentences, minus a one-character baseline call for the template",
            "interval": f"paired bootstrap of the ratio of sums, {BOOTSTRAP} resamples, seed 7",
        },
        "paid_usd": round(budget.spent, 6),
        "models": results,
    }
    Path(args.study).parent.mkdir(parents=True, exist_ok=True)
    Path(args.study).write_text(json.dumps(study, indent=1) + "\n")
    print(f"study: {args.study} (paid ${budget.spent:.4f})")

    if args.table:
        table = {
            "schema": "compound.tokenizer_ratios",
            "schema_version": 2,
            "measured_at": study["measured_at"],
            "method": "tokens per 1,000 characters of FLORES-101 devtest per language",
            "corpus": study["corpus"],
            "tokenizers": {
                name: {lang: {"tokens_per_1k_chars": v}
                       for lang, v in r["tokens_per_1k_chars"].items()}
                for name, r in results.items()
            },
        }
        Path(args.table).parent.mkdir(parents=True, exist_ok=True)
        Path(args.table).write_text(json.dumps(table, indent=2) + "\n")
        print(f"table: {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
