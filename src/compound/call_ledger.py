"""A structured per-call record of everything that passes through the proxy.

The harness reports episodes; this reports calls. That difference decides what
a run can honestly claim. An episode-level result set from a terminal-bench
matrix is n=42 per cell, which cannot separate two hosts on quality. The same
run is several thousand model calls, and the things that actually differ
between hosts by large margins live at that level: what a call cost, how much
of its prompt was served from cache, which upstream answered it, whether it
was rate-limited. Recording calls turns those from log archaeology into a
first-class artifact.

The record is deliberately what the proxy *knows*, never what it infers.
``upstream`` is the host we pinned; ``provider_echo`` is the host that says it
answered. On a pinned route those agree, and the pin is verified on every call
rather than assumed. On the unpinned ``openrouter/auto`` control arm they
differ freely, and that column is the measurement: it is how a run shows
routing hopping between upstreams mid-episode, and what that does to prompt
cache hits and the bill.

Enabled by ``COMPOUND_CALL_LEDGER=<path>`` (the ``--call-ledger`` run flag sets
it), matching how the other run-scoped signals reach the in-process proxy. When
it is unset the proxy does no extra work at all.

Field names match :mod:`compound.serving_metrics` wherever the two overlap
(``prompt_tokens``, ``cached_tokens``, ``cost_usd``, ``provider_echo``), so a
serving-harness record and a ledger row can be pooled without translation.
"""

from __future__ import annotations

import json
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any

#: Bytes of a response body retained for parsing. Chat completions are far
#: smaller; the cap only bounds memory against a pathological stream. Past it
#: the *tail* is kept, because a streamed response carries its usage in the
#: final chunks, which is the part worth having.
MAX_CAPTURE_BYTES = 8 * 1024 * 1024


def normalize_host(name: str | None) -> str | None:
    """Fold a host identifier to a form the pin and the echo can be compared on.

    The two sides speak different dialects. Our upstream tokens carry the
    endpoint tag from OpenRouter's ``/endpoints`` (``deepinfra/fp4``) while the
    response echoes a display name (``"DeepInfra"``). Dropping the quant tag,
    lowercasing, and removing separators maps both onto ``deepinfra``. Comparing
    the raw strings instead would mark every honored pin as violated.
    """
    if name is None:
        return None
    base = name.split("/", 1)[0]
    folded = "".join(ch for ch in base.lower() if ch.isalnum())
    return folded or None


#: Seconds a single call may stream before the proxy gives up on it. Observed on
#: a live run: a pinned host that had been answering in 9-24s streamed 159KB of
#: keep-alive padding for 1,044 seconds and never returned a completion, hanging
#: the agent. Without a ceiling one such call stalls a whole arm, and the tokens
#: are billed while the response that would have reported them never arrives.
DEFAULT_CALL_TIMEOUT_S = 300.0


def call_timeout_s() -> float | None:
    """How long one call may stream before it is treated as a hang.

    ``COMPOUND_CALL_TIMEOUT`` overrides the default; ``0`` disables the ceiling
    for a run that would rather wait than lose a slow but genuine completion.
    """
    raw = os.getenv("COMPOUND_CALL_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_CALL_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CALL_TIMEOUT_S
    return None if value <= 0 else value


def ledger_path_from_env() -> str | None:
    """Where to write the call ledger for this run, or ``None`` when disabled."""
    path = os.getenv("COMPOUND_CALL_LEDGER", "").strip()
    return path or None


def parse_response_payload(raw: bytes, content_type: str = "") -> dict[str, Any] | None:
    """Recover one usage-bearing object from a response body, streamed or not.

    A non-streamed response is a single JSON object. A streamed one is SSE:
    many ``data:`` lines, where the fields we want are spread across chunks
    (the provider echo arrives early, usage in the final chunk when the client
    asked for it). Both collapse to the same shape here: later values win, so
    the merged object carries whichever chunk supplied each field.

    Returns ``None`` when nothing parseable is present, which is the honest
    outcome for an empty body or a tail-truncated JSON response. Callers record
    the row with null token fields rather than guessing.
    """
    if not raw:
        return None
    text = raw.decode("utf-8", "replace")
    # Strip SSE comment lines before deciding how to parse. OpenRouter keeps a
    # long non-streaming request alive by emitting ": OPENROUTER PROCESSING"
    # lines ahead of the JSON body; leaving them in makes json.loads fail, and
    # the call records null cost. That silently biases a run's cost downward on
    # exactly the slow, expensive calls that emit them.
    if ":" in text:
        text = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith(":")
        )
    if not text.strip():
        return None
    if "data:" in text:
        merged: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:") :].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue  # a partial first line from a truncated tail
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if value is not None:
                        merged[key] = value
        return merged or None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def usage_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Token, cost and provider-echo fields from a response payload.

    Every field is ``None`` when the provider did not report it. That
    distinction matters downstream: a null ``cached_tokens`` means the host
    never told us, which is not the same claim as a measured zero cache hit,
    and only one of the two belongs in a cache-hit-rate denominator.
    """
    out: dict[str, Any] = {
        "provider_echo": None,
        "model_echo": None,
        "prompt_tokens": None,
        "cached_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "cost_usd": None,
        "finish_reason": None,
    }
    if not payload:
        return out
    out["provider_echo"] = payload.get("provider")
    out["model_echo"] = payload.get("model")
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        out["finish_reason"] = choices[0].get("finish_reason")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return out
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    out["cost_usd"] = usage.get("cost")
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        out["cached_tokens"] = ptd.get("cached_tokens")
    ctd = usage.get("completion_tokens_details")
    if isinstance(ctd, dict):
        out["reasoning_tokens"] = ctd.get("reasoning_tokens")
    return out


def request_fields(body: dict[str, Any] | None) -> dict[str, Any]:
    """What was asked for, as sent upstream after pinning was merged in.

    ``cache_marked`` records whether this specific request carried a
    ``cache_control`` marker, so a cache-hit rate can be read against the calls
    that actually opted in rather than against every call in the run.
    """
    out: dict[str, Any] = {
        "model_requested": None,
        "messages": None,
        "stream": None,
        "reasoning_pin": None,
        "cache_marked": False,
    }
    if not body:
        return out
    out["model_requested"] = body.get("model")
    messages = body.get("messages")
    if isinstance(messages, list):
        out["messages"] = len(messages)
        last = messages[-1] if messages else None
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, list):
                out["cache_marked"] = any(
                    isinstance(block, dict) and block.get("cache_control") for block in content
                )
    out["stream"] = body.get("stream")
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and "enabled" in reasoning:
        out["reasoning_pin"] = "on" if reasoning["enabled"] else "off"
    elif body.get("reasoning_effort") is not None:
        out["reasoning_pin"] = "off" if body["reasoning_effort"] == "none" else "on"
    return out


class CallLedger:
    """Append-only JSONL writer, safe across the proxy's request threads.

    One row per call, flushed as it is written: a run that dies mid-sweep keeps
    every call it already made. The lock covers the whole write because
    ``ThreadingHTTPServer`` serves concurrent requests and interleaved partial
    lines would corrupt the file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a") as handle:
            handle.write(line + "\n")
            handle.flush()


def build_record(
    *,
    route: str,
    upstream: str | None,
    status: int | None,
    latency_ms: float,
    request_body: dict[str, Any] | None,
    response_raw: bytes,
    content_type: str = "",
    error: str | None = None,
    truncated: bool = False,
    request_bytes: int | None = None,
) -> dict[str, Any]:
    """Assemble one ledger row from a completed (or failed) call.

    ``route`` is the provider token's stable label and ``upstream`` the host it
    pins, both known before the call. Everything else is read back from the
    exchange, so a row never asserts a fact the call did not produce.
    """
    payload = parse_response_payload(response_raw, content_type)
    record: dict[str, Any] = {
        "ts": time.time(),
        "run_label": os.getenv("COMPOUND_RUN_LABEL") or None,
        "route": route,
        "upstream": upstream,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "error": error,
        "response_truncated": truncated,
        # Wire sizes, kept even when the body never parsed. An abandoned call
        # reports no tokens, so this is the only surviving evidence of how large
        # the request that was lost actually was.
        "request_bytes": request_bytes,
        "response_bytes": len(response_raw),
    }
    record.update(request_fields(request_body))
    record.update(usage_fields(payload))
    # A 200 that never delivered a usage block is an abandoned call, not a free
    # one. OpenRouter pads a long request with whitespace while the model
    # generates and sends the JSON only at the end, so a client that times out
    # and disconnects first leaves us holding padding. Those tokens were still
    # generated and still billed; we simply cannot see them. Flagging the call
    # keeps a run from quietly understating the cost of its slowest host, which
    # is the one most likely to be abandoned.
    record["abandoned"] = bool(
        record["status"] == 200 and payload is None and record["prompt_tokens"] is None
    )
    if payload is None and response_raw:
        record["unparsed_head"] = response_raw[:200].decode("utf-8", "replace")
    # The pin is checked, not trusted: a pinned route whose echo names a
    # different host means routing silently escaped the pin, which invalidates
    # every per-host number in that cell. Null on either side means unknown.
    echo = normalize_host(record.get("provider_echo"))
    pinned = normalize_host(upstream)
    record["pin_honored"] = None if (pinned is None or echo is None) else (echo == pinned)
    # A provider echo identifies the host, not its quantization or endpoint.
    record["pin_scope"] = "host" if pinned is not None else None
    return record


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read usable rows, warning about lost evidence instead of silently skipping it."""
    records: list[dict[str, Any]] = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected an object")
                records.append(record)
            except ValueError:
                warnings.warn(
                    f"{path}:{line_no}: unreadable ledger row skipped; evidence is incomplete",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-route totals over a ledger.

    Two denominators are kept apart on purpose. ``calls`` counts everything the
    route attempted, so error rate is honest. ``cache_reported`` counts only the
    calls whose host actually reported a cached-token split, and the cache-hit
    rate is computed over those alone: folding silent hosts in as zeros would
    manufacture a finding out of missing data.

    ``upstreams`` is the distribution of hosts that answered. On a pinned route
    it should be a single host, and anything else means routing escaped the pin.
    On ``openrouter/auto`` it is the result: the spread is what unpinned routing
    did with identical work.
    """
    by_route: dict[str, dict[str, Any]] = {}
    for record in records:
        route = record.get("route") or "unknown"
        row = by_route.setdefault(
            route,
            {
                "route": route,
                "calls": 0,
                "errors": 0,
                "abandoned": 0,
                "pin_violations": 0,
                "pin_verified": 0,
                "pin_unverified": 0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "cache_reported": 0,
                "cache_prompt_tokens": 0,
                "cost_usd": 0.0,
                "cost_reported": 0,
                "upstreams": {},
                "latencies": [],
            },
        )
        row["calls"] += 1
        if record.get("error") or (record.get("status") not in (200, None)):
            row["errors"] += 1
        if record.get("abandoned"):
            row["abandoned"] += 1
        if record.get("pin_honored") is False:
            row["pin_violations"] += 1
        elif record.get("pin_honored") is True:
            row["pin_verified"] += 1
        elif record.get("upstream") is not None:
            row["pin_unverified"] += 1
        echo = record.get("provider_echo")
        if echo:
            row["upstreams"][echo] = row["upstreams"].get(echo, 0) + 1
        for field in ("prompt_tokens", "completion_tokens"):
            value = record.get(field)
            if isinstance(value, (int, float)):
                row[field] += value
        cached = record.get("cached_tokens")
        prompt = record.get("prompt_tokens")
        if isinstance(cached, (int, float)) and isinstance(prompt, (int, float)):
            row["cached_tokens"] += cached
            row["cache_prompt_tokens"] += prompt
            row["cache_reported"] += 1
        cost = record.get("cost_usd")
        if isinstance(cost, (int, float)):
            row["cost_usd"] += cost
            row["cost_reported"] += 1
        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)):
            row["latencies"].append(latency)

    out: list[dict[str, Any]] = []
    for row in by_route.values():
        latencies = sorted(row.pop("latencies"))
        row["latency_p50_ms"] = latencies[len(latencies) // 2] if latencies else None
        prompt = row["cache_prompt_tokens"]
        # Only meaningful when at least one call reported the split.
        row["cache_hit_rate"] = (
            row["cached_tokens"] / prompt if (prompt and row["cache_reported"]) else None
        )
        row["cost_usd"] = row["cost_usd"] if row["cost_reported"] else None
        row["cost_missing"] = row["calls"] - row["cost_reported"]
        row["distinct_upstreams"] = len(row["upstreams"])
        out.append(row)
    return sorted(out, key=lambda r: r["route"])


def format_summary(rows: list[dict[str, Any]]) -> str:
    """A fixed-width read of :func:`summarize`, one line per route."""
    header = (
        f"{'route':<24s} {'calls':>6s} {'err':>5s} {'aband':>6s} {'pin!':>5s} {'ptok':>9s} "
        f"{'cached':>9s} {'hit%':>6s} {'cost$':>10s} {'p50ms':>8s} {'hosts':>6s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        hit = "—" if row["cache_hit_rate"] is None else f"{row['cache_hit_rate'] * 100:.1f}"
        cost = "—" if row["cost_usd"] is None else f"{row['cost_usd']:.5f}"
        p50 = "—" if row["latency_p50_ms"] is None else f"{row['latency_p50_ms']:.0f}"
        lines.append(
            f"{row['route']:<24s} {row['calls']:>6d} {row['errors']:>5d} "
            f"{row['abandoned']:>6d} {row['pin_violations']:>5d} {row['prompt_tokens']:>9d} "
            f"{row['cached_tokens']:>9d} {hit:>6s} {cost:>10s} {p50:>8s} "
            f"{row['distinct_upstreams']:>6d}"
        )
        if row["cost_missing"]:
            lines.append(
                f"  {row['route']}: cost missing for {row['cost_missing']}/{row['calls']} "
                "calls; cost$ is the reported subtotal, not a complete bill."
            )
        if row["pin_unverified"] or row["pin_violations"]:
            lines.append(
                f"  {row['route']}: host pin verified on {row['pin_verified']} calls, "
                f"unverified on {row['pin_unverified']}, violated on {row['pin_violations']}."
            )
    lines.append("")
    lines.append(
        "hit% = cached / prompt tokens, over calls whose host reported the split; "
        "— = never reported."
    )
    lines.append(
        "pin! checks host identity only, not quantization. hosts = distinct "
        "upstreams that answered (1 is expected on a pinned route)."
    )
    lines.append(
        "aband = 200s that never delivered a usage block, typically the client "
        "timing out on a slow call. Those calls may have been billed but are not in cost$, "
        "so a route with abandoned calls is understated here; reconcile against "
        "the provider's own billing before quoting its cost."
    )
    return "\n".join(lines)
