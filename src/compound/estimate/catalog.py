"""Price cards for models and the hosts that serve them, from OpenRouter's public API.

A price card is one place a model can be bought: the model's own listing
(``route == "openrouter"``, OpenRouter's default routing) or one pinned host
(``route == "openrouter/<tag>"``). Prices are USD per million tokens.

Two things make a flat price wrong for real traffic, and both are kept:

- **Overrides.** Some listings charge more above a prompt size
  (``min_prompt_tokens``), and some hosts charge differently by UTC time of day
  or day of week. :meth:`PriceCard.rates_for` applies them per call. When more
  than one override matches, the first in the listed order wins; the API does
  not document precedence, so this is our assumption and it is stated here.
- **Cache prices.** ``cache_read`` is what a cached input token costs, and is
  ``None`` when the listing names no cache price, which the estimate treats as
  "this host does not discount repeated prefixes" rather than guessing.

A catalog is a dated snapshot. Prices move, and an estimate is only as fresh as
the catalog it used, so the snapshot keeps its fetch time and sources.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODELS_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def per_million(value: Any) -> float | None:
    """OpenRouter prices are USD per token as strings; convert to USD per 1M."""
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Rates:
    prompt: float | None
    completion: float | None
    cache_read: float | None
    cache_write: float | None


@dataclass
class PriceCard:
    model: str
    route: str
    provider: str
    prompt: float | None
    completion: float | None
    cache_read: float | None = None
    cache_write: float | None = None
    internal_reasoning: float | None = None
    overrides: list[dict[str, Any]] = field(default_factory=list)
    quantization: str | None = None
    tokenizer: str | None = None
    hugging_face_id: str | None = None
    up: bool = True

    @property
    def base(self) -> Rates:
        return Rates(self.prompt, self.completion, self.cache_read, self.cache_write)

    def rates_for(self, input_tokens: float, started_at: float | None) -> Rates:
        """The prices that apply to one call of this size at this time."""
        for override in self.overrides:
            if _matches(override, input_tokens, started_at):
                return Rates(
                    per_million(override["prompt"]) if "prompt" in override else self.prompt,
                    per_million(override["completion"])
                    if "completion" in override
                    else self.completion,
                    per_million(override["input_cache_read"])
                    if "input_cache_read" in override
                    else self.cache_read,
                    per_million(override["input_cache_write"])
                    if "input_cache_write" in override
                    else self.cache_write,
                )
        return self.base

    @property
    def time_dependent(self) -> bool:
        return any("utc_start" in o or "utc_days" in o for o in self.overrides)


def _matches(override: dict[str, Any], input_tokens: float, started_at: float | None) -> bool:
    if "min_prompt_tokens" in override and input_tokens < override["min_prompt_tokens"]:
        return False
    timed = "utc_start" in override or "utc_days" in override
    if timed and started_at is None:
        # Without a timestamp the base price is the honest default.
        return False
    if timed:
        moment = datetime.fromtimestamp(started_at, UTC)
        if "utc_days" in override and WEEKDAYS[moment.weekday()] not in override["utc_days"]:
            return False
        if "utc_start" in override:
            hhmm = moment.hour * 100 + moment.minute
            start, end = int(override["utc_start"]), int(override.get("utc_end", 0))
            # A window that wraps midnight (1400 to 0) covers start..2400.
            inside = start <= hhmm < end if start < end else (hhmm >= start or hhmm < end)
            if not inside:
                return False
    return True


def _pricing_fields(pricing: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": per_million(pricing.get("prompt")),
        "completion": per_million(pricing.get("completion")),
        "cache_read": per_million(pricing.get("input_cache_read")),
        "cache_write": per_million(pricing.get("input_cache_write")),
        "internal_reasoning": per_million(pricing.get("internal_reasoning")),
        "overrides": list(pricing.get("overrides") or []),
    }


def cards_from_models(payload: dict[str, Any]) -> dict[str, PriceCard]:
    """Each model's own listing, keyed by model id."""
    out = {}
    for m in payload.get("data") or []:
        mid = m.get("id")
        # ":batch" is a cheaper tier for work that can wait, and a real lever.
        # ":free" and other variants are rate-limited samples, not a product.
        if not mid or (":" in mid and not mid.endswith(":batch")):
            continue
        batch = mid.endswith(":batch")
        out[mid] = PriceCard(
            model=mid,
            route="openrouter:batch" if batch else "openrouter",
            provider="OpenRouter batch" if batch else "OpenRouter (default routing)",
            tokenizer=(m.get("architecture") or {}).get("tokenizer"),
            hugging_face_id=m.get("hugging_face_id") or None,
            **_pricing_fields(m.get("pricing") or {}),
        )
    return out


def cards_from_endpoints(model: PriceCard, payload: dict[str, Any]) -> list[PriceCard]:
    """One card per pinnable host serving ``model``."""
    out = []
    for e in (payload.get("data") or {}).get("endpoints") or []:
        tag = e.get("tag")
        if not tag:
            continue
        quant = e.get("quantization")
        out.append(
            PriceCard(
                model=model.model,
                route=f"openrouter/{tag}",
                provider=e.get("provider_name") or tag,
                quantization=None if quant in (None, "unknown") else quant,
                tokenizer=model.tokenizer,
                hugging_face_id=model.hugging_face_id,
                up=(e.get("status") or 0) >= 0,
                **_pricing_fields(e.get("pricing") or {}),
            )
        )
    return out


def _get(url: str) -> dict[str, Any]:
    headers = {"User-Agent": "compound-estimate"}
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.loads(r.read())


@dataclass
class Catalog:
    fetched_at: float
    sources: list[str]
    models: dict[str, PriceCard]
    hosts: dict[str, list[PriceCard]] = field(default_factory=dict)

    def cards(self, model: str, *, hosts: bool = True) -> list[PriceCard]:
        """Every place ``model`` can be bought: its listing, then pinned hosts."""
        if model not in self.models:
            raise KeyError(f"{model} is not in the price catalog")
        listing = [self.models[model]]
        return listing + (self.hosts.get(model, []) if hosts else [])

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "compound.price_catalog",
            "schema_version": 1,
            "fetched_at": self.fetched_at,
            "fetched_at_iso": datetime.fromtimestamp(self.fetched_at, UTC).isoformat(),
            "sources": self.sources,
            "unit": "USD per 1M tokens",
            "models": {k: asdict(v) for k, v in self.models.items()},
            "hosts": {k: [asdict(c) for c in v] for k, v in self.hosts.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Catalog:
        if data.get("schema") != "compound.price_catalog":
            raise ValueError("not a compound.price_catalog file")
        return cls(
            fetched_at=data["fetched_at"],
            sources=list(data.get("sources") or []),
            models={k: PriceCard(**v) for k, v in data["models"].items()},
            hosts={k: [PriceCard(**c) for c in v] for k, v in (data.get("hosts") or {}).items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        return cls.from_json(json.loads(Path(path).read_text()))


def fetch_catalog(host_models: list[str], get=_get) -> Catalog:
    """Fetch every model's listing, plus per-host prices for ``host_models``."""
    models = cards_from_models(get(MODELS_URL))
    sources = [MODELS_URL]
    hosts = {}
    for mid in host_models:
        if mid not in models:
            raise KeyError(f"{mid} is not listed on OpenRouter")
        url = ENDPOINTS_URL.format(model=urllib.parse.quote(mid, safe="/"))
        hosts[mid] = cards_from_endpoints(models[mid], get(url))
        sources.append(url)
    return Catalog(fetched_at=time.time(), sources=sources, models=models, hosts=hosts)
