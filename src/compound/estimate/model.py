"""The offline cost model: what a workload would cost on each model and host.

The unit is one call, priced with the rates that apply to that call's size and
time, so long-prompt tiers and time-of-day prices land where they belong. The
result for each candidate is a range, not a point, because two inputs are
genuinely uncertain before anything is measured:

- **Caching.** The low-cost end assumes a host's cache serves as much as the
  prompts allow (the profile's reuse share); the high-cost end assumes only what
  today's host already served. A host that lists no cache price gets no cache
  in either case.
- **Tokens.** Moving to another model scales token counts by the measured
  tokenizer ratio for the call's language. An unmeasured tokenizer widens the
  range instead of being assumed equal.

What is *not* modelled is stated on every result rather than hidden: reasoning
tokens are assumed to scale like output (no paid measurements yet), cache write
fees are not charged, and nothing here says a cheaper model is good enough.
Quality needs an eval; the result labels model switches accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compound.estimate.catalog import Catalog, PriceCard
from compound.estimate.tokens import TokenizerTable, lang_key
from compound.shape.records import CallShape

LIMITATIONS = [
    "Output tokens, including reasoning, are assumed to scale with the tokenizer ratio; "
    "how much a candidate model reasons on this workload is not measured.",
    "Cache write fees are not charged; hosts that bill writes cost slightly more.",
    "Retries and failed calls are not priced.",
    "A cheaper model is not shown to be good enough; that needs an eval on this workload.",
]


@dataclass(frozen=True)
class Call:
    """One priced unit: a call's sizes, weighted to a month."""

    weight: float
    input_tokens: float
    cached_tokens: float | None
    output_tokens: float
    started_at: float | None
    lang: str


@dataclass
class Workload:
    calls: list[Call]
    observed_cache_share: float | None
    reuse_share: float | None
    basis: str
    excluded_calls: int = 0

    @property
    def monthly_calls(self) -> float:
        return sum(c.weight for c in self.calls)


def workload_from_shapes(
    shapes: list[CallShape],
    profile: dict[str, Any],
    *,
    monthly_calls: float | None = None,
) -> Workload:
    """Price the observed calls, scaled to a month.

    Scaling uses ``monthly_calls`` when given, otherwise the profile's own
    projection, which exists only when the data covers at least a day.
    """
    usable = [s for s in shapes if s.input_tokens is not None and s.output_tokens is not None]
    if not usable:
        raise ValueError("no calls report both input and output tokens")
    if monthly_calls is not None:
        weight = monthly_calls / len(usable)
        basis = f"{len(usable):,} observed calls scaled to {monthly_calls:,.0f} a month"
    elif profile.get("monthly_projection"):
        weight = profile["monthly_projection"]["calls"] / len(shapes)
        basis = profile["monthly_projection"]["basis"]
    else:
        raise ValueError(
            "the data covers less than a day; pass the monthly call volume to scale it"
        )
    calls = [
        Call(
            weight=weight,
            input_tokens=float(s.input_tokens),
            cached_tokens=None if s.cached_input_tokens is None else float(s.cached_input_tokens),
            output_tokens=float(s.output_tokens),
            started_at=s.started_at,
            lang=lang_key(s.lang_hint),
        )
        for s in usable
    ]
    cache = profile.get("cache") or {}
    return Workload(
        calls=calls,
        observed_cache_share=(cache.get("observed") or {}).get("hit_share"),
        reuse_share=(cache.get("reuse") or {}).get("reuse_share"),
        basis=basis,
        excluded_calls=len(shapes) - len(usable),
    )


def workload_from_form(
    *,
    monthly_calls: float,
    input_tokens: float,
    output_tokens: float,
    cached_share: float | None = None,
    reuse_share: float | None = None,
    turkish_share: float = 0.0,
) -> Workload:
    """The calculator's simple inputs, as an average call split by language."""
    calls = []
    for lang, share in (("tr", turkish_share), ("en", 1 - turkish_share)):
        if share > 0:
            calls.append(
                Call(
                    weight=monthly_calls * share,
                    input_tokens=input_tokens,
                    cached_tokens=None if cached_share is None else input_tokens * cached_share,
                    output_tokens=output_tokens,
                    started_at=None,
                    lang=lang,
                )
            )
    return Workload(
        calls=calls,
        observed_cache_share=cached_share,
        reuse_share=reuse_share,
        basis=f"{monthly_calls:,.0f} calls a month at the stated average sizes",
    )


def _cost(
    card: PriceCard, calls: list[Call], ratio: dict[str, float], cache: float | None
) -> float:
    """Monthly USD. ``cache`` is a share of input; None means use each call's own."""
    total = 0.0
    for call in calls:
        r = ratio[call.lang]
        tokens_in = call.input_tokens * r
        rates = card.rates_for(tokens_in, call.started_at)
        if rates.prompt is None or rates.completion is None:
            raise ValueError(f"{card.route} for {card.model} lists no token price")
        if rates.cache_read is None:
            cached = 0.0
        elif cache is None:
            cached = (call.cached_tokens or 0.0) * r
        else:
            cached = tokens_in * cache
        cache_rate = rates.cache_read if rates.cache_read is not None else rates.prompt
        per_call = (
            (tokens_in - cached) * rates.prompt
            + cached * cache_rate
            + call.output_tokens * r * rates.completion
        ) / 1_000_000
        total += per_call * call.weight
    return total


def lever(current: PriceCard, card: PriceCard) -> str:
    if card.model.endswith(":batch") and card.model.removesuffix(":batch") == current.model:
        return "batch tier, same model"
    if card.model == current.model:
        return "same model, other host" if card.route != current.route else "today"
    return "switch model, quality not yet measured"


@dataclass
class Row:
    model: str
    route: str
    provider: str
    lever: str
    monthly_low: float
    monthly_high: float
    saving_low: float
    saving_high: float
    tokens: str
    cache: str
    notes: list[str] = field(default_factory=list)


def estimate(
    workload: Workload,
    catalog: Catalog,
    *,
    current: str,
    current_route: str = "openrouter",
    candidates: list[str],
    tokenizers: TokenizerTable | None = None,
    include_down: bool = False,
) -> dict[str, Any]:
    """Today's monthly cost and a cost range for every candidate place to buy."""
    tokenizers = tokenizers or TokenizerTable.bundled()
    current_cards = {c.route: c for c in catalog.cards(current)}
    if current_route not in current_cards:
        raise KeyError(f"{current} has no price card for route {current_route}")
    today_card = current_cards[current_route]
    same = {lang: 1.0 for lang in ("tr", "en")}
    today = _cost(today_card, workload.calls, same, None)

    observed = workload.observed_cache_share or 0.0
    best_cache = max(observed, workload.reuse_share or 0.0)

    cards: list[PriceCard] = []
    for model in [current, *[c for c in candidates if c != current]]:
        cards += catalog.cards(model)
        batch = f"{model}:batch"
        if batch in catalog.models:
            cards.append(catalog.models[batch])

    rows = []
    for card in cards:
        if not card.up and not include_down:
            continue
        if card.prompt is None or card.completion is None:
            continue
        low_ratio = {
            lang: tokenizers.ratio(today_card, _base(card, catalog), lang)[0] for lang in same
        }
        high_ratio = {
            lang: tokenizers.ratio(today_card, _base(card, catalog), lang)[1] for lang in same
        }
        has_cache = card.cache_read is not None
        low = _cost(card, workload.calls, low_ratio, best_cache if has_cache else 0.0)
        high = _cost(card, workload.calls, high_ratio, observed if has_cache else 0.0)
        notes = []
        if card.time_dependent:
            notes.append("price changes by UTC time or weekday; priced per call time")
        if card.cache_write:
            notes.append("bills cache writes, not included")
        if card.quantization:
            notes.append(f"quantization {card.quantization}")
        rows.append(
            Row(
                model=card.model,
                route=card.route,
                provider=card.provider,
                lever=lever(today_card, card),
                monthly_low=low,
                monthly_high=high,
                saving_low=1 - high / today if today else 0.0,
                saving_high=1 - low / today if today else 0.0,
                tokens=tokenizers.basis(_base(card, catalog))
                if card.model.removesuffix(":batch") != current
                else "same model",
                cache="none listed" if not has_cache else "listed",
                notes=notes,
            )
        )
    rows.sort(key=lambda r: (r.monthly_high, r.monthly_low))
    keep = [r for r in rows if not r.lever.startswith("switch")]
    best_keep = keep[0] if keep else None
    return {
        "schema": "compound.cost_estimate",
        "schema_version": 1,
        "current": {"model": current, "route": current_route, "monthly_usd": today},
        "workload": {
            "monthly_calls": workload.monthly_calls,
            "basis": workload.basis,
            "excluded_calls": workload.excluded_calls,
            "observed_cache_share": workload.observed_cache_share,
            "reuse_share": workload.reuse_share,
        },
        "catalog": {
            "fetched_at": catalog.fetched_at,
            "sources": catalog.sources,
        },
        "best_without_switching_model": None if best_keep is None else best_keep.__dict__,
        "rows": [r.__dict__ for r in rows],
        "limitations": LIMITATIONS,
    }


def _base(card: PriceCard, catalog: Catalog) -> PriceCard:
    """Tokenizer identity lives on the model listing, batch variants included."""
    return catalog.models.get(card.model.removesuffix(":batch"), card)


def format_estimate(result: dict[str, Any], top: int = 15) -> str:
    cur = result["current"]
    lines = [
        f"today: {cur['model']} via {cur['route']}, about ${cur['monthly_usd']:,.0f} a month",
        f"workload: {result['workload']['basis']}",
        "",
        f"{'cheapest first (high end of range)':<58} {'per month':>21}  saving",
    ]
    for row in result["rows"][:top]:
        name = f"{row['model']} via {row['route']}"
        cost = f"${row['monthly_low']:,.0f} to ${row['monthly_high']:,.0f}"
        saving = f"{row['saving_low']:.0%} to {row['saving_high']:.0%}"
        lines.append(f"{name[:58]:<58} {cost:>21}  {saving}  {row['lever']}")
    best = result["best_without_switching_model"]
    if best:
        lines += [
            "",
            f"best without changing model: {best['route']} ({best['provider']}), "
            f"{best['saving_low']:.0%} to {best['saving_high']:.0%} saving",
        ]
    lines += ["", "not modelled:"] + [f"  - {x}" for x in result["limitations"]]
    return "\n".join(lines)
