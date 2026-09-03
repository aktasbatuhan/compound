"""Controlled serving-metrics harness: one model, many hosts, matched reasoning.

Replays a set of real agent payloads (configurable prompt *shapes*, including the
agent's ``json_schema`` response_format) against every provider token with
reasoning pinned explicitly ON and OFF, streaming, so time-to-first-token and
decode tok/s are measured from chunk timestamps rather than inferred from
whole-call duration. A nonce is prepended to the first message so prompt caches
never serve a warm prefix.

Errors are recorded, never retried: the error rate under the same 2-way
concurrency the benchmark used is itself a measurement (reliability axis).

The routes are not hardcoded. They come from the same provider-token grammar the
rest of the CLI uses (:func:`compound.providers_registry.parse_providers`), so
``openrouter/deepinfra``, ``doubleword/flex``, and ``openrouter/auto`` are one
comma-separated list, and each token carries its own pinning (OpenRouter
``provider.only`` with ``require_parameters``, or the Doubleword service tier).
Each host's reasoning dialect is applied per arm: OpenRouter takes a
``reasoning: {enabled}`` block, Doubleword takes ``reasoning_effort`` where
"none" disables.

Keys come from the environment via each token's registry entry
(``OPENROUTER_API_KEY`` / ``DOUBLEWORD_API_KEY``); this module never reads a
.env file and never prints a key.

Prompt payloads are not bundled here: a run loads them from a ``--shapes`` JSON
file mapping ``name -> {messages, response_format}``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from compound.providers_registry import (
    ProviderSpec,
    openrouter_provider_block,
    parse_providers,  # noqa: F401 — re-exported for callers building routes
)

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT_S = 420
DEFAULT_INTERVAL_S = 3600.0
DEFAULT_REPS = 2
#: 2 concurrent calls per route, mirroring the benchmark's own concurrency.
CONCURRENCY = 2

REASONING_ON = "reasoning-on"
REASONING_OFF = "reasoning-off"
MODES = (REASONING_ON, REASONING_OFF)

#: How a cell treats the host's prompt cache.
#:   "cold" prepends a per-call nonce, so no prefix can ever be served warm.
#:         This isolates raw serving speed and is the honest number to compare
#:         against a vendor benchmark that does not mention caching.
#:   "warm" sends a byte-identical prompt every rep, so the host may serve the
#:         prefix from cache. The cold/warm delta is the cache measurement.
CACHE_COLD = "cold"
CACHE_WARM = "warm"
CACHE_MODES = (CACHE_COLD, CACHE_WARM)

#: How much generated text to keep per call. Enough to locate where two hosts
#: first diverge at temperature 0 without turning the ledger into a corpus.
TEXT_KEEP_CHARS = 4000


def make_nonce() -> str:
    """A unique prefix that busts any prompt cache so latency is honest."""
    return f"[bench-nonce {uuid.uuid4()}]\n"


def prepend_nonce(messages: list[dict[str, Any]], nonce: str) -> list[dict[str, Any]]:
    """Return a deep copy of ``messages`` with ``nonce`` on the first message.

    The original list is never mutated. The nonce goes in front of the first
    message's content, handling both the string and the block-list content forms.
    """
    copied = json.loads(json.dumps(messages))
    first = copied[0]
    content = first.get("content")
    if isinstance(content, str):
        first["content"] = nonce + content
    else:
        first["content"] = [{"type": "text", "text": nonce}] + (content or [])
    return copied


def model_for(spec: ProviderSpec, model_or: str | None, model: str | None) -> str:
    """The model slug to send on this host, per provider kind.

    OpenRouter routes take ``--model-or`` (the OpenRouter slug); Doubleword and
    direct routes take ``--model`` (the host's own slug). Raises ``ValueError``
    when the needed flag is missing, so a run fails loudly before any spend.
    """
    if spec.kind == "openrouter":
        if not model_or:
            raise ValueError(
                f"route {spec.label!r} is an OpenRouter host but --model-or was not given"
            )
        return model_or
    if not model:
        raise ValueError(
            f"route {spec.label!r} is a {spec.kind} host but --model was not given"
        )
    return model


def build_body(
    spec: ProviderSpec,
    model: str,
    mode: str,
    shape: dict[str, Any],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    nonce: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Build the streamed chat-completions body for one (route, mode, shape).

    Reasoning is pinned in each host's own dialect: OpenRouter takes a
    ``reasoning: {enabled}`` block (plus ``usage: {include}`` so the response
    carries cost), Doubleword takes ``reasoning_effort`` medium/none. An
    OpenRouter upstream token also carries its ``provider.only`` pin with
    ``require_parameters`` on; ``openrouter/auto`` is deliberately unpinned. A
    Doubleword flex token forwards its ``service_tier``.
    """
    # None means "generate one"; an explicit empty string means "send the prompt
    # unchanged", which is what a warm-cache cell needs.
    if nonce is None:
        nonce = make_nonce()
    messages = prepend_nonce(shape["messages"], nonce) if nonce else shape["messages"]
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # A shape may set its own output budget: a profile grid varies input and
        # output length independently, and one global cap would flatten that axis.
        "max_tokens": int(shape.get("max_tokens") or max_tokens),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if shape.get("response_format"):
        body["response_format"] = shape["response_format"]
    if spec.kind == "openrouter":
        body["reasoning"] = {"enabled": mode == REASONING_ON}
        body["usage"] = {"include": True}
        if spec.upstream:
            body["provider"] = openrouter_provider_block(spec.upstream)
    else:
        # Doubleword / direct: OpenAI-style reasoning_effort, "none" disables.
        body["reasoning_effort"] = "medium" if mode == REASONING_ON else "none"
        if spec.service_tier:
            body["service_tier"] = spec.service_tier
    return body


def measure_stream(
    raw_lines: Iterable[bytes | str], now: Callable[[], float]
) -> dict[str, Any]:
    """Consume an SSE stream, sampling ``now()`` once per parsed data line.

    Returns the raw measurement handles the caller turns into TTFT / decode TPS:
    ``first_delta`` (first content OR reasoning delta), ``last_delta``, the
    content-chunk count, and the final usage / provider echo / finish reason.
    """
    first_delta: float | None = None
    last_delta: float | None = None
    n_content_chunks = 0
    usage: dict[str, Any] | None = None
    provider: str | None = None
    finish: str | None = None
    parts: list[str] = []
    for raw in raw_lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        raw = raw.strip()
        if not raw.startswith("data:") or raw == "data: [DONE]":
            continue
        try:
            d = json.loads(raw[5:])
        except json.JSONDecodeError:
            continue
        t = now()
        if d.get("provider"):
            provider = d["provider"]
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                if first_delta is None:
                    first_delta = t
                last_delta = t
                if delta.get("content"):
                    n_content_chunks += 1
                    parts.append(delta["content"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {
        "first_delta": first_delta,
        "last_delta": last_delta,
        "n_content_chunks": n_content_chunks,
        "usage": usage,
        "provider": provider,
        "finish": finish,
        "text": "".join(parts),
    }


def one_call(
    spec: ProviderSpec,
    model: str,
    mode: str,
    shape_name: str,
    shape: dict[str, Any],
    round_no: int,
    rep: int,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    cache_mode: str = CACHE_COLD,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Make one streamed call and return its result record.

    Timeouts, resets, and HTTP errors (429s included) are recorded on the record,
    never retried and never raised: unreliability is data, not a crash.

    A ``warm`` cell sends the prompt byte-identically on every rep so the host's
    prompt cache can serve the prefix; a ``cold`` cell prepends a fresh nonce so
    it cannot. The generated text is fingerprinted so two hosts running the same
    prompt at temperature 0 can be compared token for token.
    """
    body = build_body(
        spec, model, mode, shape,
        max_tokens=max_tokens,
        nonce="" if cache_mode == CACHE_WARM else None,
        temperature=temperature,
    )
    key = os.environ.get(spec.required_key_env(), "")
    rec: dict[str, Any] = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "round": round_no,
        "route": spec.label,
        "mode": mode,
        "shape": shape_name,
        "rep": rep,
        "cache_mode": cache_mode,
        "temperature": temperature,
    }
    req = urlrequest.Request(
        spec.forward_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    m: dict[str, Any] = {
        "first_delta": None,
        "last_delta": None,
        "n_content_chunks": 0,
        "usage": None,
        "provider": None,
        "finish": None,
        "text": "",
    }
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as r:
            rec["status"] = getattr(r, "status", None)
            m = measure_stream(r, time.monotonic)
    except urlerror.HTTPError as e:
        rec["status"] = e.code
        rec["error"] = e.read().decode("utf-8", "replace")[:500]
    except Exception as e:  # timeouts, resets: reliability data, not crashes
        rec["status"] = None
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
    t_end = time.monotonic()
    first_delta = m["first_delta"]
    last_delta = m["last_delta"]
    rec["total_s"] = round(t_end - t0, 3)
    rec["ttft_s"] = round(first_delta - t0, 3) if first_delta else None
    rec["finish_reason"] = m["finish"]
    rec["provider_echo"] = m["provider"]
    rec["n_content_chunks"] = m["n_content_chunks"]
    text = m.get("text") or ""
    # The hash identifies an exact generation; the head is what an analyst reads
    # when two hosts disagree and the question is where they first split.
    rec["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
    rec["text_len"] = len(text)
    rec["text"] = text[:TEXT_KEEP_CHARS]
    usage = m["usage"]
    if usage:
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        ctd = usage.get("completion_tokens_details") or {}
        rec["reasoning_tokens"] = ctd.get("reasoning_tokens")
        ptd = usage.get("prompt_tokens_details") or {}
        rec["cached_tokens"] = ptd.get("cached_tokens")
        rec["cost_usd"] = usage.get("cost")
        ct = usage.get("completion_tokens")
        if ct and first_delta and last_delta and last_delta > first_delta:
            rec["decode_tps"] = round(ct / (last_delta - first_delta), 2)
    return rec


def run_round(
    specs: list[ProviderSpec],
    model_or: str | None,
    model: str | None,
    shapes: dict[str, Any],
    round_no: int,
    reps: int,
    out_path: Path,
    lock: threading.Lock,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    modes: tuple[str, ...] = MODES,
    cache_modes: tuple[str, ...] = (CACHE_COLD,),
    temperature: float = 0.7,
) -> None:
    """Run every (route, mode, cache mode, shape, rep) cell once, routes in parallel."""
    shape_names = list(shapes)

    def route_worker(spec: ProviderSpec) -> None:
        route_model = model_for(spec, model_or, model)
        cells = [
            (mode, cmode, sname, rep)
            for mode in modes
            for cmode in cache_modes
            for sname in shape_names
            for rep in range(reps)
        ]
        # Warm cells run at concurrency 1: rep 0 populates the prefix cache that
        # the later reps are supposed to hit, and firing them together would race
        # that write and understate the hit rate.
        workers = 1 if CACHE_WARM in cache_modes and len(cache_modes) == 1 else CONCURRENCY
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    one_call,
                    spec,
                    route_model,
                    mode,
                    sname,
                    shapes[sname],
                    round_no,
                    rep,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    cache_mode=cmode,
                    temperature=temperature,
                )
                for mode, cmode, sname, rep in cells
            ]
            for fut in futures:
                rec = fut.result()
                with lock:
                    with open(out_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        list(pool.map(route_worker, specs))


def run_serving(
    specs: list[ProviderSpec],
    model_or: str | None,
    model: str | None,
    shapes: dict[str, Any],
    *,
    out_dir: Path,
    rounds: int = 1,
    interval: float = DEFAULT_INTERVAL_S,
    reps: int = DEFAULT_REPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    modes: tuple[str, ...] = MODES,
    cache_modes: tuple[str, ...] = (CACHE_COLD,),
    temperature: float = 0.7,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> Path:
    """Run ``rounds`` scheduled rounds, ``interval`` seconds apart, then summarize.

    Appends one JSON line per call to ``<out_dir>/results.jsonl`` and returns that
    path. Scheduled rounds capture time-of-day variance; the summary at the end
    aggregates every round.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    lock = threading.Lock()
    for r in range(1, rounds + 1):
        t0 = time.time()
        run_round(
            specs,
            model_or,
            model,
            shapes,
            r,
            reps,
            out_path,
            lock,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            modes=modes,
            cache_modes=cache_modes,
            temperature=temperature,
        )
        n = _round_count(out_path, r)
        log(f"round {r}/{rounds}: {n} calls in {time.time() - t0:.0f}s -> {out_path}")
        if r < rounds:
            sleep(interval)
    records = load_records(out_path)
    log(format_summary(summarize(records)))
    return out_path


def _round_count(out_path: Path, round_no: int) -> int:
    return sum(1 for rec in load_records(out_path) if rec.get("round") == round_no)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a results.jsonl into a list of records, skipping blank lines."""
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_shapes(path: Path) -> dict[str, Any]:
    """Load a shapes file: ``name -> {messages, response_format}``."""
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"shapes file {path} must be a non-empty object of name -> shape")
    for name, shape in data.items():
        if not isinstance(shape, dict) or "messages" not in shape:
            raise ValueError(f"shape {name!r} must have a 'messages' field")
    return data


def percentile(values: Iterable[Any], p: float) -> float | None:
    """Linear-interpolated p-th percentile of the non-null values, or None."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate result records into one row per (route, mode).

    A record counts as an error when it carries an ``error`` field (HTTP errors,
    timeouts, resets). Percentiles ignore null samples; cost is summed only over
    records that reported a provider-billed cost.
    """
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((rec.get("route"), rec.get("mode")), []).append(rec)
    rows: list[dict[str, Any]] = []
    for (route, mode), recs in groups.items():
        costs = [r["cost_usd"] for r in recs if r.get("cost_usd") is not None]
        rows.append(
            {
                "route": route,
                "mode": mode,
                "n": len(recs),
                "errors": sum(1 for r in recs if r.get("error")),
                "ttft_p50": percentile((r.get("ttft_s") for r in recs), 50),
                "ttft_p90": percentile((r.get("ttft_s") for r in recs), 90),
                "decode_tps_p50": percentile((r.get("decode_tps") for r in recs), 50),
                "total_p50": percentile((r.get("total_s") for r in recs), 50),
                "completion_p50": percentile(
                    (r.get("completion_tokens") for r in recs), 50
                ),
                "cost_usd": sum(costs) if costs else None,
            }
        )
    rows.sort(key=lambda row: (str(row["route"]), str(row["mode"])))
    return rows


def _fmt(value: Any, spec: str) -> str:
    return format(value, spec) if value is not None else "-"


def format_summary(rows: list[dict[str, Any]]) -> str:
    """Render the per-(route, mode) summary as a fixed-width text table."""
    header = (
        f"{'route':22s} {'mode':13s} {'n':>4s} {'err':>4s} "
        f"{'ttft50':>7s} {'ttft90':>7s} {'dec50':>7s} {'tot50':>7s} "
        f"{'ctok50':>7s} {'cost$':>9s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row['route'])[:22]:22s} {str(row['mode'])[:13]:13s} "
            f"{row['n']:>4d} {row['errors']:>4d} "
            f"{_fmt(row['ttft_p50'], '7.2f')} {_fmt(row['ttft_p90'], '7.2f')} "
            f"{_fmt(row['decode_tps_p50'], '7.1f')} {_fmt(row['total_p50'], '7.2f')} "
            f"{_fmt(row['completion_p50'], '7.0f')} {_fmt(row['cost_usd'], '9.5f')}"
        )
    return "\n".join(lines)
