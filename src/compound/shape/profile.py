"""Summarise a set of call shapes into the workload profile other steps read.

The profile answers what a host comparison needs to know about traffic before
any quality question: how many calls, how large, how bursty, how many run at
once, and how much of each prompt a cache could serve. The cost estimate reads
it, and so will the replay that measures a host at this traffic's shape.

Two cache numbers are kept apart on purpose. ``observed`` is what the current
host reported serving from cache. ``reuse`` is what the prefix chains say a
cache *could* serve: the longest prefix each call shares with an earlier call
inside the cache's time-to-live. A big gap between them is itself a finding,
usually a prompt layout or a missing cache marker.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from compound.shape.records import CallShape

DAY = 86_400.0
#: A month for projections, in days. Stated so a reader can redo the sum.
MONTH_DAYS = 30.0
MIN_CALLS_FOR_BURSTINESS = 30


@dataclass(frozen=True)
class CacheAssumptions:
    """What a provider cache is assumed to do when estimating reuse.

    The defaults match the common published behaviour (OpenAI and Anthropic
    cache prefixes of at least 1,024 tokens for about five minutes). Hosts
    differ, which is why these are parameters and are printed in the profile.
    """

    ttl_s: float = 300.0
    min_prefix_tokens: int = 1024
    #: Characters per token when a call reports no input tokens. Rough; used
    #: only as a fallback and counted in ``chars_fallback_calls``.
    chars_per_token: float = 4.0


def quantiles(values: list[float]) -> dict[str, float] | None:
    """Mean and nearest-rank percentiles, or None for no data."""
    if not values:
        return None
    ordered = sorted(values)

    def rank(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p10": rank(0.10),
        "p50": rank(0.50),
        "p90": rank(0.90),
        "p99": rank(0.99),
        "max": ordered[-1],
    }


def _known(shapes: list[CallShape], attr: str) -> list[float]:
    return [float(v) for s in shapes if (v := getattr(s, attr)) is not None]


def peak_concurrency(shapes: list[CallShape]) -> int | None:
    """Most calls in flight at once, from start and completion times."""
    events = []
    for s in shapes:
        end = s.completed_at
        if s.started_at is None or end is None:
            continue
        events.append((s.started_at, 1))
        events.append((end, -1))
    if not events:
        return None
    # Ends sort before starts at the same instant, so back-to-back calls on one
    # connection do not count as overlapping.
    events.sort(key=lambda e: (e[0], e[1]))
    live = peak = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


def arrival_profile(shapes: list[CallShape]) -> dict[str, Any]:
    starts = sorted(s.started_at for s in shapes if s.started_at is not None)
    if len(starts) < 2:
        return {"timed_calls": len(starts)}
    span = starts[-1] - starts[0]
    per_minute = Counter(int(t // 60) for t in starts)
    active_minutes = (int(starts[-1] // 60) - int(starts[0] // 60)) + 1
    mean_per_minute = len(starts) / active_minutes
    by_hour = Counter(int((t % DAY) // 3600) for t in starts)
    return {
        "timed_calls": len(starts),
        "first": starts[0],
        "last": starts[-1],
        "span_s": span,
        "mean_rps": len(starts) / span if span > 0 else None,
        "peak_per_minute": max(per_minute.values()),
        "mean_per_minute": mean_per_minute,
        # Peak minute against the average minute over the whole span. A steady
        # stream is near 1; batch jobs and office-hours traffic run far higher.
        # A handful of calls makes any ratio meaningless, so it waits for data.
        "burstiness": (
            max(per_minute.values()) / mean_per_minute
            if len(starts) >= MIN_CALLS_FOR_BURSTINESS
            else None
        ),
        "calls_by_utc_hour": [by_hour.get(h, 0) for h in range(24)],
    }


def prefix_reuse(shapes: list[CallShape], cache: CacheAssumptions) -> dict[str, Any]:
    """Share of input tokens a prefix cache could serve, from the chains."""
    seen: dict[str, float] = {}
    eligible_input = reusable = 0.0
    calls = with_chain = fallback = 0
    for s in shapes:
        if not s.prefix_chain or s.started_at is None:
            continue
        calls += 1
        total_chars = s.prefix_chain[-1][1] or 0
        if s.input_tokens is not None:
            tokens_per_char = s.input_tokens / total_chars if total_chars else 0.0
            input_tokens = float(s.input_tokens)
        else:
            fallback += 1
            tokens_per_char = 1 / cache.chars_per_token
            input_tokens = total_chars * tokens_per_char
        shared_chars = 0
        for digest, chars in s.prefix_chain:
            last = seen.get(digest)
            if last is not None and s.started_at - last <= cache.ttl_s:
                shared_chars = chars
            else:
                break
        shared_tokens = shared_chars * tokens_per_char
        if shared_tokens < cache.min_prefix_tokens:
            shared_tokens = 0.0
        with_chain += 1
        eligible_input += input_tokens
        reusable += min(shared_tokens, input_tokens)
        for digest, _ in s.prefix_chain:
            seen[digest] = s.started_at
    return {
        "calls_with_chain": with_chain,
        "chars_fallback_calls": fallback,
        "input_tokens": eligible_input,
        "reusable_tokens": reusable,
        "reuse_share": reusable / eligible_input if eligible_input else None,
        "assumptions": {
            "ttl_s": cache.ttl_s,
            "min_prefix_tokens": cache.min_prefix_tokens,
            "chars_per_token": cache.chars_per_token,
        },
    }


def observed_cache(shapes: list[CallShape]) -> dict[str, Any]:
    reported = [s for s in shapes if s.cached_input_tokens is not None and s.input_tokens]
    total = sum(s.input_tokens or 0 for s in reported)
    cached = sum(s.cached_input_tokens or 0 for s in reported)
    return {
        "calls_reporting": len(reported),
        "input_tokens": total,
        "cached_tokens": cached,
        "hit_share": cached / total if total else None,
    }


def _group(shapes: list[CallShape], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[CallShape]] = defaultdict(list)
    for s in shapes:
        groups[getattr(s, key) or "(unknown)"].append(s)
    rows = []
    for name, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rows.append(
            {
                key: name,
                "calls": len(members),
                "input_tokens": sum(s.input_tokens or 0 for s in members),
                "output_tokens": sum(s.output_tokens or 0 for s in members),
                "reasoning_tokens": sum(s.reasoning_tokens or 0 for s in members),
            }
        )
    return rows


def build_profile(
    shapes: list[CallShape], cache: CacheAssumptions | None = None
) -> dict[str, Any]:
    """The workload profile for ``shapes``, as a JSON-ready dict."""
    cache = cache or CacheAssumptions()
    arrivals = arrival_profile(shapes)
    span_days = (arrivals.get("span_s") or 0) / DAY
    monthly = None
    if span_days >= 1:
        monthly = {
            "calls": len(shapes) / span_days * MONTH_DAYS,
            "input_tokens": sum(_known(shapes, "input_tokens")) / span_days * MONTH_DAYS,
            "output_tokens": sum(_known(shapes, "output_tokens")) / span_days * MONTH_DAYS,
            "basis": f"{len(shapes)} calls over {span_days:.1f} days, scaled to {MONTH_DAYS:g}",
        }
    langs = Counter(s.lang_hint or "(unknown)" for s in shapes)
    errors = sum(1 for s in shapes if s.status == "error")
    return {
        "schema": "compound.workload_profile",
        "schema_version": 1,
        "calls": len(shapes),
        "error_calls": errors,
        "tokens": {
            "input": quantiles(_known(shapes, "input_tokens")),
            "output": quantiles(_known(shapes, "output_tokens")),
            "reasoning": quantiles(_known(shapes, "reasoning_tokens")),
            "messages": quantiles(_known(shapes, "message_count")),
        },
        "totals": {
            "input_tokens": sum(_known(shapes, "input_tokens")),
            "output_tokens": sum(_known(shapes, "output_tokens")),
            "reasoning_tokens": sum(_known(shapes, "reasoning_tokens")),
        },
        "latency_ms": quantiles(_known(shapes, "latency_ms")),
        "arrivals": arrivals,
        "peak_concurrency": peak_concurrency(shapes),
        "monthly_projection": monthly,
        "cache": {"observed": observed_cache(shapes), "reuse": prefix_reuse(shapes, cache)},
        "language_hint": dict(langs),
        "by_model": _group(shapes, "model"),
        "by_task": _group(shapes, "task_key"),
    }


def format_profile(profile: dict[str, Any]) -> str:
    """A short plain-text reading of a profile for the terminal."""

    def num(value: float | None, digits: int = 0) -> str:
        return "unknown" if value is None else f"{value:,.{digits}f}"

    lines = [f"{profile['calls']:,} calls, {profile['error_calls']:,} errors"]
    for label, key in (("input", "input"), ("output", "output"), ("reasoning", "reasoning")):
        q = profile["tokens"][key]
        if q:
            lines.append(
                f"{label} tokens: p50 {num(q['p50'])}, p90 {num(q['p90'])}, "
                f"mean {num(q['mean'])} (n={q['n']:,})"
            )
    arr = profile["arrivals"]
    if arr.get("span_s"):
        burst = arr["burstiness"]
        lines.append(
            f"arrivals: {num(arr['mean_per_minute'], 2)} calls a minute on average, "
            f"busiest minute {arr['peak_per_minute']:,}, "
            + ("burstiness needs 30+ calls" if burst is None else f"burstiness {burst:.1f}x")
        )
    lines.append(f"peak concurrency: {num(profile['peak_concurrency'])}")
    month = profile["monthly_projection"]
    if month:
        lines.append(
            f"per month: {num(month['calls'])} calls, {num(month['input_tokens'])} input "
            f"and {num(month['output_tokens'])} output tokens ({month['basis']})"
        )
    else:
        lines.append("per month: not projected, the data covers less than one day")
    obs, reuse = profile["cache"]["observed"], profile["cache"]["reuse"]
    hit = obs["hit_share"]
    lines.append(
        "cache served today: "
        + (
            "not reported"
            if hit is None
            else f"{hit:.1%} of input ({obs['calls_reporting']:,} calls)"
        )
    )
    share = reuse["reuse_share"]
    lines.append(
        "cache could serve: "
        + (
            "unknown, no prompts to compare"
            if share is None
            else f"{share:.1%} of input (prefixes repeated within "
            f"{reuse['assumptions']['ttl_s']:g}s, at least "
            f"{reuse['assumptions']['min_prefix_tokens']:,} tokens)"
        )
    )
    langs = ", ".join(f"{k} {v:,}" for k, v in profile["language_hint"].items())
    lines.append(f"language hint: {langs}")
    return "\n".join(lines)
