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
"none" disables, Anthropic takes a ``thinking`` block.

Two wire dialects, one record shape. Every host but Anthropic speaks chat
completions. Anthropic is measured on its native Messages API rather than its
OpenAI-compatible layer, because that layer has no prompt caching, returns empty
token details and ignores ``service_tier``: it would score Anthropic at 0% cache
the way an unmarked call scores Doubleword. Both dialects are driven over the
same stdlib HTTP client with the same per-line clock sampling, so a TTFT from
one is comparable to a TTFT from the other; a vendor SDK would add its own
retries and buffering to exactly one arm.

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

from compound.orproxy import cache_optin_enabled, mark_cache_prefix
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

    A per-host id (``--host-model openai=gpt-5.4-mini``, carried on
    :attr:`ProviderSpec.wire_model`) wins outright: a grid that mixes first-party
    hosts has no single slug that every host knows. Otherwise OpenRouter routes
    take ``--model-or`` (the OpenRouter slug); Doubleword and direct routes take
    ``--model`` (the host's own slug). Raises ``ValueError`` when the needed
    flag is missing, so a run fails loudly before any spend.
    """
    if spec.wire_model:
        return spec.wire_model
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
    if spec.dialect == "anthropic":
        return build_messages_body(spec, model, mode, messages, shape, max_tokens=max_tokens)
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # A shape may set its own output budget: a profile grid varies input and
        # output length independently, and one global cap would flatten that axis.
        # The field name is the host's (OpenAI: max_completion_tokens).
        spec.max_tokens_field: int(shape.get("max_tokens") or max_tokens),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if shape.get("response_format"):
        body["response_format"] = shape["response_format"]
    # Mirror the pinning proxy exactly. A host whose cache is opt-in
    # (Doubleword) serves nothing warm without an explicit marker, so a harness
    # that skips the marker measures that host at its worst case while every
    # implicit-cache host is measured at its best. That is not a fair comparison,
    # it is the opt-in trap with our name on it.
    if spec.cache_strategy == "explicit_marker" and cache_optin_enabled():
        body["messages"] = mark_cache_prefix(body["messages"])
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


def build_messages_body(
    spec: ProviderSpec,
    model: str,
    mode: str,
    messages: list[dict[str, Any]],
    shape: dict[str, Any],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """The Anthropic Messages body for one call, from the same chat-shaped prompt.

    System and developer turns are hoisted into the top-level ``system`` field
    (Anthropic takes one), joined with a newline, which is what its own
    compatibility layer does. Reasoning on is adaptive thinking; reasoning off
    is ``thinking: disabled``, which current models accept only without a high
    effort setting and Claude Fable rejects outright, so an off cell on such a
    model records a 400 rather than silently thinking anyway.

    No ``temperature`` is sent. Claude 4.7 and later reject sampling parameters
    with a 400, so the harness cannot pin them; the record carries
    ``temperature: null`` and the agreement analysis treats such a route as
    unpinnable rather than as a temperature-0 host that failed to reproduce.

    A cache marker goes on the last message exactly as it does for Doubleword,
    since Anthropic's cache is the same opt-in ``cache_control`` design.
    ``response_format`` has no Messages equivalent and is dropped.
    """
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            content = m.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    str(c.get("text", "")) for c in content if isinstance(c, dict)
                )
            system_parts.append(str(content or ""))
        else:
            turns.append({"role": role, "content": m.get("content")})
    body: dict[str, Any] = {
        "model": model,
        "messages": turns,
        "max_tokens": int(shape.get("max_tokens") or max_tokens),
        "stream": True,
        "thinking": {"type": "adaptive"} if mode == REASONING_ON else {"type": "disabled"},
    }
    if system_parts:
        body["system"] = "\n".join(system_parts)
    if spec.cache_strategy == "explicit_marker" and cache_optin_enabled():
        body["messages"] = mark_cache_prefix(body["messages"])
    if spec.service_tier:
        body["service_tier"] = spec.service_tier
    return body


def measure_messages_stream(
    raw_lines: Iterable[bytes | str], now: Callable[[], float]
) -> dict[str, Any]:
    """Consume a Messages-API SSE stream into the same handles as :func:`measure_stream`.

    ``message_start`` carries the input side of usage (uncached input, cache
    reads, cache writes, the tier that served the call); ``message_delta``
    carries the cumulative output count and the stop reason. The usage dict
    returned is normalised to chat-completions field names so :func:`one_call`
    reads both dialects the same way: ``prompt_tokens`` is the whole prompt
    (uncached + cache read + cache write), ``cached_tokens`` is the cache read,
    and ``cache_write_tokens`` is the extra Anthropic exposes and OpenAI does not.
    """
    first_delta: float | None = None
    last_delta: float | None = None
    n_content_chunks = 0
    usage: dict[str, Any] | None = None
    finish: str | None = None
    error: str | None = None
    parts: list[str] = []
    for raw in raw_lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        raw = raw.strip()
        if not raw.startswith("data:"):
            continue
        try:
            d = json.loads(raw[5:])
        except json.JSONDecodeError:
            continue
        t = now()
        kind = d.get("type")
        if kind == "message_start":
            u = (d.get("message") or {}).get("usage") or {}
            uncached = u.get("input_tokens") or 0
            read = u.get("cache_read_input_tokens") or 0
            write = u.get("cache_creation_input_tokens") or 0
            usage = {
                "prompt_tokens": uncached + read + write,
                "prompt_tokens_details": {"cached_tokens": read},
                "cache_write_tokens": write,
                "completion_tokens": None,
                "service_tier": u.get("service_tier"),
            }
        elif kind == "content_block_delta":
            delta = d.get("delta") or {}
            if delta.get("type") in ("text_delta", "thinking_delta"):
                if first_delta is None:
                    first_delta = t
                last_delta = t
                if delta.get("type") == "text_delta" and delta.get("text"):
                    n_content_chunks += 1
                    parts.append(delta["text"])
        elif kind == "message_delta":
            finish = (d.get("delta") or {}).get("stop_reason") or finish
            out = (d.get("usage") or {}).get("output_tokens")
            if out is not None:
                usage = usage or {"prompt_tokens": None, "prompt_tokens_details": {}}
                usage["completion_tokens"] = out
        elif kind == "error":
            error = json.dumps(d.get("error") or d)[:500]
    return {
        "first_delta": first_delta,
        "last_delta": last_delta,
        "n_content_chunks": n_content_chunks,
        "usage": usage,
        "provider": None,
        "finish": finish,
        "text": "".join(parts),
        "error": error,
    }


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
    tier: str | None = None
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
        # OpenAI echoes the tier that actually served the call on every chunk
        # ("default", "flex", "priority"), so a flex arm is checkable per call
        # the way a pinned OpenRouter upstream is via its provider echo.
        if d.get("service_tier"):
            tier = d["service_tier"]
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
        "service_tier": tier,
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
    anthropic = spec.dialect == "anthropic"
    rec: dict[str, Any] = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "round": round_no,
        "route": spec.label,
        "model": model,
        "mode": mode,
        "shape": shape_name,
        "rep": rep,
        "cache_mode": cache_mode,
        # None means the host does not take a sampling temperature; see
        # build_messages_body. The value is what was sent, never what was asked.
        "temperature": None if anthropic else temperature,
        "cache_marked": spec.cache_strategy == "explicit_marker" and cache_optin_enabled(),
        "service_tier_requested": spec.service_tier,
    }
    if anthropic:
        url = spec.forward_base_url.rstrip("/") + "/messages"
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        url = spec.forward_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req = urlrequest.Request(url, data=json.dumps(body).encode(), headers=headers)
    # A host may declare that it needs longer (a queued tier before first byte);
    # the harness default is a floor, never a ceiling on such a host.
    timeout_s = max(timeout_s, spec.timeout_s or 0)
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
            m = (measure_messages_stream if anthropic else measure_stream)(r, time.monotonic)
            if m.get("error"):
                # A Messages stream can fail mid-way with an ``error`` event
                # after a 200; that is a failed call, not a short answer.
                rec["error"] = m["error"]
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
    rec["service_tier_echo"] = m.get("service_tier")
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
        if usage.get("cache_write_tokens") is not None:
            rec["cache_write_tokens"] = usage["cache_write_tokens"]
        if usage.get("service_tier"):
            rec["service_tier_echo"] = usage["service_tier"]
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


def derived_cost_usd(rec: dict[str, Any], rates: dict[str, Any]) -> float | None:
    """Price one call from a published rate card, for hosts that bill no cost field.

    ``rates`` is ``compound.yaml``'s ``serving_rates_usd_per_million_tokens``:
    route label -> model id -> ``{input, cached_input, output, cache_write?}``.
    The result is *derived*, not measured, and every consumer must label it so:
    OpenRouter's per-call ``cost`` is the provider's own bill, this is ours.
    Returns ``None`` when the record has no token counts, no matching rate, or
    when the route is one whose cost is measured (the measured figure wins).
    """
    if rec.get("cost_usd") is not None or rec.get("prompt_tokens") is None:
        return None
    per_route = rates.get(rec.get("route") or "") or {}
    card = per_route.get(rec.get("model") or "")
    if card is None and len(per_route) == 1:
        # Ledgers written before the model was recorded per call: one card for
        # the route is unambiguous.
        card = next(iter(per_route.values()))
    if not card:
        return None
    prompt = rec.get("prompt_tokens") or 0
    cached = rec.get("cached_tokens") or 0
    written = rec.get("cache_write_tokens") or 0
    out = rec.get("completion_tokens") or 0
    uncached = max(prompt - cached - written, 0)
    usd = (
        uncached * float(card.get("input", 0))
        + cached * float(card.get("cached_input", card.get("input", 0)))
        + written * float(card.get("cache_write", card.get("input", 0)))
        + out * float(card.get("output", 0))
    ) / 1e6
    return round(usd, 8)


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
