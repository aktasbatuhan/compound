"""Real sentences to stand in for prompts a replay is not allowed to see.

A replay sends traffic the size and structure of a team's real calls without
their text. Lorem ipsum would be the easy filler, but it tokenizes nothing like
real language and caches nothing like a real prompt, so the speed and cache
numbers it produced would describe lorem ipsum. Instead the filler is real
prose in the call's language, from FLORES-101 (CC BY-SA 4.0), fetched once and
cached under ``~/.compound/corpus``.

Text is chosen deterministically from a seed, so the same seed always gives the
same words. The replay seeds each message with its prefix hash, which is what
makes two calls that shared a prompt prefix in production share it again.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "gsarti/flores_101"
CONFIGS = {"en": "eng", "tr": "tur"}
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_DIR = Path("~/.compound/corpus").expanduser()


class FillerCorpus:
    def __init__(self, sentences: dict[str, list[str]]) -> None:
        missing = [lang for lang in ("en", "tr") if not sentences.get(lang)]
        if missing:
            raise ValueError(f"filler corpus has no sentences for {missing}")
        self.sentences = sentences

    def text(self, seed: str, lang: str, chars: int) -> str:
        """About ``chars`` characters of real prose, the same for the same seed."""
        pool = self.sentences.get(lang) or self.sentences["en"]
        rng = random.Random(seed)
        parts: list[str] = []
        total = 0
        while total < chars:
            sentence = pool[rng.randrange(len(pool))]
            parts.append(sentence)
            total += len(sentence) + 1
        text = " ".join(parts)
        if len(text) > chars:
            cut = text.rfind(" ", 0, max(chars, 1))
            text = text[: cut if cut > 0 else chars]
        return text

    @classmethod
    def flores(cls, cache_dir: Path = DEFAULT_DIR) -> FillerCorpus:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sentences = {}
        for lang, config in CONFIGS.items():
            path = cache_dir / f"flores101-devtest-{lang}.json"
            if not path.exists():
                path.write_text(json.dumps(_fetch(config), ensure_ascii=False))
            sentences[lang] = json.loads(path.read_text())
        return cls(sentences)


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "compound-replay"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _fetch(config: str) -> list[str]:
    out: list[str] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": DATASET, "config": config, "split": "devtest", "offset": offset,
             "length": 100}
        )
        rows = _get(f"{ROWS_URL}?{query}")["rows"]
        if not rows:
            return out
        out += [r["row"]["sentence"] for r in rows]
        offset += len(rows)
        time.sleep(1)
