"""Replay a workload's shape against hosts and measure how each one holds up.

``compound estimate`` says what a workload should cost on a host. A replay
checks it by sending traffic with the same sizes, the same rhythm and the same
prompt-prefix structure, without the original text, and measuring what came
back: time to first token, decode speed, errors, rate limits, cache hits and
cost per call.

Three choices keep the replay honest:

- **Structure, not just size.** Each original message becomes one message of
  real prose in the call's language, sized to its share of the original input
  tokens. The filler is seeded by the message's prefix hash, so calls that
  shared a prefix in production share the same bytes here and a host's cache
  sees what it would have seen. A per-run nonce goes into every seed, so one
  replay never warms the cache for the next.
- **Open loop.** Calls leave on the original schedule, divided by the load
  factor, whether or not earlier calls have finished. Concurrency and bursts
  then come from the traffic itself, which is how a rate limit is actually
  met in production; a closed loop of N workers would hide it.
- **Same time for every host.** Each scheduled call goes to every route at
  once, so hosts are compared under the same minute of the day.

Output length cannot be forced, so each call asks for an answer of about the
original length and caps ``max_tokens`` there. Every record keeps the target
and actual sizes, and the summary reports how closely they matched.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compound.core.providers_registry import ProviderSpec
from compound.serving import metrics
from compound.serving.filler import FillerCorpus
from compound.shape.records import CallShape

#: Tokens per 1,000 characters used to size filler when the target model's
#: tokenizer is unknown: o200k_base on FLORES-101 (benchmarks/tokenizers).
DEFAULT_DENSITY = {"en": 203.64, "tr": 283.15}
LANG_NAMES = {"en": "English", "tr": "Turkish"}
#: Roles a chat-completions request accepts for a plain text turn.
ROLE_MAP = {"system": "system", "developer": "system", "user": "user",
            "assistant": "assistant", "tool": "user", "function": "user"}


@dataclass(frozen=True)
class Planned:
    index: int
    offset_s: float
    shape: CallShape


def replayable(shape: CallShape) -> bool:
    return bool(shape.started_at is not None and shape.prefix_chain and shape.input_tokens)


def busiest_window(shapes: list[CallShape], minutes: float) -> list[CallShape]:
    """The calls inside the ``minutes``-long window with the most starts."""
    width = minutes * 60
    best: tuple[int, int] = (0, 0)
    j = 0
    for i, s in enumerate(shapes):
        while shapes[j].started_at < s.started_at - width:  # type: ignore[operator]
            j += 1
        if i - j + 1 > best[1] - best[0]:
            best = (j, i + 1)
    return shapes[best[0] : best[1]]


def plan(
    shapes: list[CallShape],
    *,
    load: float = 1.0,
    max_calls: int | None = None,
    window_minutes: float | None = None,
) -> list[Planned]:
    """Which calls to send and when, as offsets from the start of the run."""
    if load <= 0:
        raise ValueError("load must be positive")
    usable = sorted((s for s in shapes if replayable(s)), key=lambda s: s.started_at)
    if not usable:
        raise ValueError("no calls have a timestamp, input tokens and a prefix chain")
    if window_minutes:
        usable = busiest_window(usable, window_minutes)
    if max_calls:
        usable = usable[:max_calls]
    t0 = usable[0].started_at
    return [Planned(i, (s.started_at - t0) / load, s) for i, s in enumerate(usable)]


def peak_planned_concurrency(planned: list[Planned], load: float) -> int:
    """Calls in flight at once if each took its original latency (scaled)."""
    events = []
    for p in planned:
        latency = (p.shape.latency_ms or 0) / 1000
        events += [(p.offset_s, 1), (p.offset_s + latency, -1)]
    events.sort(key=lambda e: (e[0], e[1]))
    live = peak = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


def synthesize(
    shape: CallShape,
    corpus: FillerCorpus,
    *,
    run_nonce: str,
    density: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A request with the call's structure and size, filled with real prose."""
    density = density or DEFAULT_DENSITY
    lang = "tr" if shape.lang_hint == "tr" else "en"
    chain = shape.prefix_chain or []
    total_chars = chain[-1][1] if chain else 0
    target = float(shape.input_tokens or 0)
    roles = list(shape.roles or [])
    deltas = [chain[0][1]] + [b[1] - a[1] for a, b in zip(chain, chain[1:], strict=False)]
    messages = []
    for i in range(1, len(chain)):
        # The tool list has no message of its own here, so its size rides on
        # the first message, where it sits in the real prefix.
        chars = deltas[i] + (deltas[0] if i == 1 else 0)
        tokens = target * chars / total_chars if total_chars else 0
        filler_chars = max(1, round(tokens / density[lang] * 1000))
        role = ROLE_MAP.get(roles[i - 1] if i - 1 < len(roles) else "user", "user")
        text = corpus.text(f"{run_nonce}:{chain[i][0]}", lang, filler_chars)
        messages.append({"role": role, "content": text})
    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "Continue."})
    out_tokens = int(shape.output_tokens or 256)
    words = max(10, round(out_tokens * 0.6))
    messages[-1] = {
        "role": "user",
        "content": messages[-1]["content"]
        + f"\n\nReply in {LANG_NAMES[lang]} with about {words} words.",
    }
    return {"messages": messages, "max_tokens": max(16, out_tokens),
            "target_input_tokens": target, "lang": lang}


Call = Callable[..., dict[str, Any]]


#: (input, output) USD per million tokens for a route, when the catalog knows it.
Prices = dict[str, tuple[float, float]]


class Budget:
    """Spend cap that counts calls in flight, not just calls that finished.

    In an open-loop replay many calls are out at once, so checking only settled
    cost would let a burst sail past the cap. Each call reserves an upper bound
    before it is sent (its target input and full output budget at the route's
    listed price, no cache discount) and settles to the reported cost when it
    returns. A route with no known price reserves the average settled cost so
    far, which is weaker, and the docs say so.
    """

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent = 0.0
        self.reserved = 0.0
        self.settled_calls = 0
        self.lock = threading.Lock()

    def reserve(self, estimate: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + estimate > self.max_usd:
                return False
            self.reserved += estimate
            return True

    def settle(self, estimate: float, actual: float | None) -> None:
        with self.lock:
            self.reserved -= estimate
            self.spent += actual if actual is not None else estimate
            self.settled_calls += 1

    def average(self) -> float:
        with self.lock:
            return self.spent / self.settled_calls if self.settled_calls else 0.0


def call_estimate(body: dict[str, Any], route: str, prices: Prices | None, budget: Budget) -> float:
    if prices and route in prices:
        price_in, price_out = prices[route]
        return (body["target_input_tokens"] * price_in + body["max_tokens"] * price_out) / 1e6
    return budget.average()


def run_replay(
    planned: list[Planned],
    routes: list[tuple[ProviderSpec, str]],
    corpus: FillerCorpus,
    *,
    load: float,
    out_path: Path,
    mode: str = metrics.REASONING_OFF,
    max_usd: float = 5.0,
    timeout_s: int = metrics.DEFAULT_TIMEOUT_S,
    density: dict[str, float] | None = None,
    prices: Prices | None = None,
    call: Call = metrics.one_call,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    max_workers: int = 256,
) -> dict[str, Any]:
    """Send ``planned`` to every route on schedule and write one record per call."""
    run_nonce = secrets.token_hex(8)
    bodies = [synthesize(p.shape, corpus, run_nonce=run_nonce, density=density) for p in planned]
    budget = Budget(max_usd)
    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sent = skipped = 0

    def one(p: Planned, body: dict[str, Any], spec: ProviderSpec, model: str,
            scheduled: float, started: float, estimate: float) -> None:
        try:
            rec = call(spec, model, mode, f"replay-{p.index}", body, 0, 0,
                       timeout_s=timeout_s, cache_mode=metrics.CACHE_WARM)
        except Exception as exc:  # a crash in one call must not lose the reservation
            budget.settle(estimate, None)
            rec = {"route": spec.label, "status": None, "error": f"{type(exc).__name__}: {exc}"}
        else:
            budget.settle(estimate, rec.get("cost_usd"))
        rec.pop("text", None)  # the replay's own filler and answers are not evidence
        rec.update({
            "replay_load": load,
            "replay_index": p.index,
            "scheduled_offset_s": round(p.offset_s, 3),
            "send_lag_s": round(started - scheduled, 3),
            "target_input_tokens": round(body["target_input_tokens"]),
            "target_output_tokens": body["max_tokens"],
            "original_cached_tokens": p.shape.cached_input_tokens,
            "lang": body["lang"],
        })
        with lock, out_path.open("a") as handle:
            handle.write(json.dumps(rec, default=str) + "\n")

    stopped = False
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        start = clock()
        for p, body in zip(planned, bodies, strict=True):
            if stopped:
                skipped += len(routes)
                continue
            due = start + p.offset_s
            wait = due - clock()
            if wait > 0:
                sleep(wait)
            for spec, model in routes:
                estimate = call_estimate(body, spec.token, prices, budget)
                if not budget.reserve(estimate):
                    stopped = True
                    skipped += 1
                    continue
                pool.submit(one, p, body, spec, model, due, clock(), estimate)
                sent += 1
    return {"sent": sent, "skipped_over_budget": skipped, "spent_usd": round(budget.spent, 6),
            "run_nonce": run_nonce}


def _pct(values: list[float], p: float) -> float | None:
    return metrics.percentile(values, p)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per route and load: speed, reliability, cache and cost."""
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["route"], r.get("replay_load", 1.0)), []).append(r)
    rows = []
    for (route, load), recs in sorted(groups.items()):
        ok = [r for r in recs if r.get("status") == 200 and not r.get("error")]
        prompt = sum(r.get("prompt_tokens") or 0 for r in ok if r.get("cached_tokens") is not None)
        cached = sum(r.get("cached_tokens") or 0 for r in ok if r.get("cached_tokens") is not None)
        size = [r["prompt_tokens"] / r["target_input_tokens"] for r in ok
                if r.get("prompt_tokens") and r.get("target_input_tokens")]
        costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
        rows.append({
            "route": route,
            "load": load,
            "calls": len(recs),
            "ok": len(ok),
            "rate_limited": sum(1 for r in recs if r.get("status") == 429),
            "other_errors": len(recs) - len(ok) - sum(1 for r in recs if r.get("status") == 429),
            "ttft_p50_s": _pct([r["ttft_s"] for r in ok if r.get("ttft_s") is not None], 50),
            "ttft_p90_s": _pct([r["ttft_s"] for r in ok if r.get("ttft_s") is not None], 90),
            "decode_p50_tps": _pct([r["decode_tps"] for r in ok if r.get("decode_tps")], 50),
            "total_p90_s": _pct([r["total_s"] for r in ok if r.get("total_s") is not None], 90),
            "cache_share": cached / prompt if prompt else None,
            "size_match_p50": _pct(size, 50),
            "cost_usd": sum(costs) if costs else None,
            "cost_per_call_usd": sum(costs) / len(costs) if costs else None,
            "send_lag_p90_s": _pct([r["send_lag_s"] for r in recs if "send_lag_s" in r], 90),
        })
    return rows


def format_summary(rows: list[dict[str, Any]], reuse_share: float | None = None) -> str:
    def f(value: Any, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    lines = [
        f"{'route':34} {'load':>4} {'ok/calls':>9} {'429':>4} {'err':>4} {'ttft p50':>8} "
        f"{'p90':>6} {'tok/s':>6} {'cache':>6} {'$/call':>9} {'size':>5}"
    ]
    for r in rows:
        lines.append(
            f"{r['route'][:34]:34} {r['load']:>4g} {r['ok']:>4}/{r['calls']:<4} "
            f"{r['rate_limited']:>4} {r['other_errors']:>4} {f(r['ttft_p50_s'], '.2f'):>8} "
            f"{f(r['ttft_p90_s'], '.2f'):>6} {f(r['decode_p50_tps'], '.0f'):>6} "
            f"{f(r['cache_share'], '.0%'):>6} {f(r['cost_per_call_usd'], '.5f'):>9} "
            f"{f(r['size_match_p50'], '.2f'):>5}"
        )
    if reuse_share is not None:
        lines.append(
            f"\nthe profile predicted a cache could serve {reuse_share:.0%} of input; "
            "compare with the cache column"
        )
    lines.append(
        "size is actual prompt tokens over target, per call (1.00 means the filler "
        "matched the original size)"
    )
    return "\n".join(lines)
