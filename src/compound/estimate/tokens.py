"""How many tokens the same text costs on one model compared with another.

Ratios come from ``data/tokenizers.json``, written by
``scripts/measure_tokenizers.py`` from the FLORES-101 parallel corpus: tokens
per 1,000 characters per language for each tokenizer. A model is matched by
its own measured id first (OpenAI models through OpenAI's token counter, any
model through OpenRouter's usage), then by its Hugging Face repo. Other OpenAI
models fall back to ``o200k_base``, which matched the measured GPT models to
within a few tokens in 27,000; the label still says "assumed".

A model whose tokenizer is not public (Claude, Gemini) is unknown. Rather than
pretend it matches, the estimate assumes it falls somewhere inside the range
of the tokenizers we did measure, and widens the cost range accordingly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from compound.estimate.catalog import PriceCard

LANGS = ("tr", "en")


def lang_key(hint: str | None) -> str:
    """Map a shape's language hint to a measured language."""
    return "tr" if hint == "tr" else "en"


@dataclass(frozen=True)
class Density:
    """Tokens per 1,000 characters for one model and language, as a range."""

    low: float
    high: float
    basis: str  # "measured", "assumed", or "unknown"

    @property
    def exact(self) -> bool:
        return self.low == self.high


class TokenizerTable:
    def __init__(self, data: dict[str, Any]) -> None:
        if data.get("schema") != "compound.tokenizer_ratios":
            raise ValueError("not a compound.tokenizer_ratios file")
        self.data = data
        self.table = data["tokenizers"]

    @classmethod
    def bundled(cls) -> TokenizerTable:
        text = resources.files("compound.estimate").joinpath("data/tokenizers.json").read_text()
        return cls(json.loads(text))

    def _span(self, lang: str) -> tuple[float, float]:
        values = [t[lang]["tokens_per_1k_chars"] for t in self.table.values() if lang in t]
        return min(values), max(values)

    def density(self, card: PriceCard, lang: str) -> Density:
        # Closed models measured through their own API or OpenRouter's usage
        # are keyed by model id; open models by their Hugging Face repo.
        keys = [f"openrouter:{card.model}", card.hugging_face_id]
        if card.model.startswith("openai/"):
            keys.insert(0, f"openai:{card.model.removeprefix('openai/')}")
        for key in keys:
            if key and key in self.table and lang in self.table[key]:
                value = self.table[key][lang]["tokens_per_1k_chars"]
                return Density(value, value, "measured")
        if card.tokenizer == "GPT" and "tiktoken:o200k_base" in self.table:
            value = self.table["tiktoken:o200k_base"][lang]["tokens_per_1k_chars"]
            return Density(value, value, "assumed")
        low, high = self._span(lang)
        return Density(low, high, "unknown")

    def ratio(self, current: PriceCard, candidate: PriceCard, lang: str) -> tuple[float, float]:
        """(low, high) multiplier from today's token counts to the candidate's."""
        if current.model == candidate.model:
            return 1.0, 1.0
        cur, cand = self.density(current, lang), self.density(candidate, lang)
        if cur.basis == "unknown" and cand.basis == "unknown":
            # Two unknown tokenizers: no evidence either way, so no adjustment,
            # but the range still shows how far apart measured ones can be.
            low, high = cur.low, cur.high
            return low / high, high / low
        return cand.low / cur.high, cand.high / cur.low

    def basis(self, card: PriceCard) -> str:
        return self.density(card, "en").basis
