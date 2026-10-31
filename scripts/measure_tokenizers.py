"""Measure tokens per 1,000 characters of Turkish and English text per tokenizer.

The same sentence costs a different number of tokens on different models, and
Turkish swings far more than English because its long agglutinated words split
differently. The cost estimate needs that ratio to move a workload's token
counts from today's model to a candidate.

The corpus is Wikipedia's article intros on business topics, fetched in both
languages with their revision ids recorded so the run can be repeated. Open
models are measured with their published ``tokenizer.json``; OpenAI models with
tiktoken's ``o200k_base``. Models whose tokenizer is not public (Claude, Gemini)
are not measured, and the estimate treats them as unknown.

    uv run --with tokenizers --with tiktoken python scripts/measure_tokenizers.py \\
        --hf deepseek-ai/DeepSeek-V4.1-Flash,zai-org/GLM-5.3 \\
        --out src/compound/estimate/data/tokenizers.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TOPICS = [
    "Bank", "Inflation", "Insurance", "Invoice", "Contract", "Customer service",
    "Tax", "Loan", "Credit card", "Accounting", "Payment", "Supply chain",
    "Artificial intelligence", "Personal data", "Stock exchange", "Interest rate",
]
API = "https://{lang}.wikipedia.org/w/api.php"


def _get(url: str) -> dict:
    agent = "compound-tokenizer-ratios/1.0 (github.com/aktasbatuhan/compound)"
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _pages(lang: str, titles: list[str], **params: str) -> list[dict]:
    """One batched query; the API returns up to 20 intros per request."""
    base = {"action": "query", "format": "json", "formatversion": "2", "redirects": "1",
            "titles": "|".join(titles), "exintro": "1", "explaintext": "1", "exlimit": "20",
            "rvprop": "ids"}
    url = API.format(lang=lang) + "?" + urllib.parse.urlencode(base | params)
    return _get(url)["query"]["pages"]


def fetch_corpus() -> dict:
    """Intros of each topic in English and its Turkish counterpart."""
    en_pages = _pages("en", TOPICS, prop="extracts|langlinks|revisions", lllang="tr",
                      lllimit="max")
    pairs = {}
    for page in en_pages:
        links = page.get("langlinks") or []
        if links and page.get("extract"):
            pairs[links[0]["title"]] = page
    tr_pages = {p["title"]: p for p in _pages("tr", list(pairs), prop="extracts|revisions")}
    docs = {"en": [], "tr": []}
    for tr_title, en in pairs.items():
        tr = tr_pages.get(tr_title)
        if not tr or not tr.get("extract"):
            continue
        for lang, page in (("en", en), ("tr", tr)):
            docs[lang].append({"title": page["title"], "revid": page["revisions"][0]["revid"],
                               "text": page["extract"]})
    return docs


def measure(encode, docs: dict) -> dict:
    out = {}
    for lang, items in docs.items():
        chars = sum(len(d["text"]) for d in items)
        tokens = sum(len(encode(d["text"])) for d in items)
        out[lang] = {"tokens_per_1k_chars": round(tokens / chars * 1000, 2), "chars": chars}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf", default="", help="comma-separated Hugging Face repo ids")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tiktoken
    from tokenizers import Tokenizer

    docs = fetch_corpus()
    print(f"corpus: {len(docs['en'])} topics, {sum(len(d['text']) for d in docs['tr']):,} "
          f"Turkish and {sum(len(d['text']) for d in docs['en']):,} English characters")
    table: dict[str, dict] = {}
    enc = tiktoken.get_encoding("o200k_base")
    table["tiktoken:o200k_base"] = measure(enc.encode, docs)
    for repo in filter(None, (r.strip() for r in args.hf.split(","))):
        try:
            tok = Tokenizer.from_pretrained(repo)
        except Exception as exc:  # gated, missing tokenizer.json, network
            print(f"skip {repo}: {exc}", file=sys.stderr)
            continue
        table[repo] = measure(lambda text, t=tok: t.encode(text, add_special_tokens=False).ids,
                              docs)
        print(f"{repo}: tr {table[repo]['tr']['tokens_per_1k_chars']}, "
              f"en {table[repo]['en']['tokens_per_1k_chars']}")
    result = {
        "schema": "compound.tokenizer_ratios",
        "schema_version": 1,
        "measured_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "method": "tokens per 1,000 characters of Wikipedia article intros, no special tokens",
        "corpus": {lang: [{"title": d["title"], "revid": d["revid"]} for d in items]
                   for lang, items in docs.items()},
        "tokenizers": table,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
