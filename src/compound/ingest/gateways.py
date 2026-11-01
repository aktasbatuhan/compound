"""Import calls from LLM gateways (OpenRouter, Vercel AI Gateway) as shapes.

Neither gateway can list a team's calls through its API. Both can look one up
by its generation id, which every response already carries in its ``id``
field (``gen-...`` on OpenRouter, ``gen_...`` on Vercel), so a team that logs
those ids can hand us a list and get per-call shapes back: when the call
started, how long it took, native token counts from the provider (cached and
reasoning included), cost, provider and finish reason.

OpenRouter also keeps the prompt when the account turns on input and output
logging, and ``GET /generation/content`` returns it. With ``content=True`` the
importer reads it only long enough to build the salted prefix chain, the
language hint and the roles, then drops it, the same as every other loader.

The two gateways share one record schema, with one trap: on OpenRouter
``latency`` is the total time and ``generation_time`` the time spent
generating, while on Vercel ``latency`` is the time to first token and
``generation_time`` the total. Mixing them up would halve or double every
latency, so each gateway is mapped on its own.

For volume without per-call data, :func:`fetch_activity` reads OpenRouter's
daily totals per model and provider (it needs a management key).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compound.shape.loaders import parse_time
from compound.shape.records import CallShape, shape_from_messages

GATEWAYS = {
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "content": True,
    },
    "vercel": {
        "base": "https://ai-gateway.vercel.sh/v1",
        "key_env": "AI_GATEWAY_API_KEY",
        "content": False,
    },
}

Get = Callable[[str], dict[str, Any]]


class NotFound(Exception):
    """The gateway has no record (yet) for this id."""


def http_get(key: str, retries: int = 4) -> Get:
    def get(url: str) -> dict[str, Any]:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {key}", "User-Agent": "compound-import"}
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise NotFound(url) from exc
                if exc.code in (429, 500, 502, 503, 524, 529) and attempt < retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable")

    return get


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def shape_from_generation(
    gateway: str,
    record: dict[str, Any],
    *,
    content: dict[str, Any] | None = None,
    salt: bytes,
) -> CallShape:
    """One generation record (plus optional stored content) as a shape."""
    started = parse_time(record.get("created_at"))
    if gateway == "vercel":
        total_ms, ttft_ms = record.get("generation_time"), record.get("latency")
    else:
        total_ms, ttft_ms = record.get("latency"), None
    fields: dict[str, Any] = {
        "started_at": started,
        "latency_ms": total_ms,
        "ended_at": started + total_ms / 1000 if started is not None and total_ms else None,
        "model": record.get("model"),
        "provider": record.get("provider_name"),
        "trace_id": record.get("session_id") or record.get("request_id") or record.get("id"),
        # Native counts are the provider's own tokenizer, which is what it bills.
        "input_tokens": _int(record.get("native_tokens_prompt", record.get("tokens_prompt"))),
        "output_tokens": _int(
            record.get("native_tokens_completion", record.get("tokens_completion"))
        ),
        "reasoning_tokens": _int(record.get("native_tokens_reasoning")),
        "cached_input_tokens": _int(record.get("native_tokens_cached")),
        "status": "error" if record.get("cancelled") else "ok",
        "finish_reason": record.get("finish_reason"),
        "source": f"{gateway}_generation",
        "extra": {
            k: v
            for k, v in {
                "generation_id": record.get("id"),
                "ttft_ms": ttft_ms,
                "cost_usd": record.get("total_cost"),
                "streamed": record.get("streamed"),
                "service_tier": record.get("service_tier"),
            }.items()
            if v is not None
        },
    }
    messages = None
    if content is not None:
        stored = content.get("input")
        if isinstance(stored, dict):  # either {"messages": [...]} or {"prompt": "..."}
            stored = stored.get("messages", stored.get("prompt"))
        if isinstance(stored, str):
            messages = [{"role": "user", "content": stored}]
        elif isinstance(stored, list):
            messages = [m for m in stored if isinstance(m, dict)]
        if content.get("error"):
            fields["status"] = "error"
    if messages:
        return shape_from_messages(messages, tools=None, salt=salt, **fields)
    return CallShape(**fields)


def read_ids(paths: Iterable[str | Path]) -> list[str]:
    """Generation ids from text files (one per line) or JSON(L) with an id field."""
    ids: list[str] = []
    for path in map(Path, paths):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
                value = obj.get("generation_id") or obj.get("id") or response.get("id")
                if isinstance(value, str):
                    ids.append(value)
            else:
                ids.append(line.split(",")[0].strip())
    return list(dict.fromkeys(i for i in ids if i.startswith("gen")))


@dataclass
class ImportReport:
    shapes: list[CallShape] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    with_content: int = 0


def import_generations(
    gateway: str,
    ids: list[str],
    get: Get,
    *,
    salt: bytes,
    content: bool = False,
    workers: int = 4,
) -> ImportReport:
    """Look up every id; ids the gateway does not know are reported, not guessed."""
    if gateway not in GATEWAYS:
        raise ValueError(f"unknown gateway {gateway!r}; use {', '.join(GATEWAYS)}")
    if content and not GATEWAYS[gateway]["content"]:
        raise ValueError(f"{gateway} does not return stored prompts")
    base = GATEWAYS[gateway]["base"]
    report = ImportReport()

    def one(gen_id: str) -> tuple[str, CallShape | None, bool]:
        query = urllib.parse.urlencode({"id": gen_id})
        try:
            record = get(f"{base}/generation?{query}")["data"]
        except NotFound:
            return gen_id, None, False
        stored = None
        if content:
            try:
                stored = get(f"{base}/generation/content?{query}").get("data")
            except NotFound:
                stored = None  # logging was off for this call; keep the metadata
        shape = shape_from_generation(gateway, record, content=stored, salt=salt)
        return gen_id, shape, shape.prefix_chain is not None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for gen_id, shape, has_content in pool.map(one, ids):
            if shape is None:
                report.not_found.append(gen_id)
            else:
                report.shapes.append(shape)
                report.with_content += has_content
    report.shapes.sort(key=lambda s: (s.started_at is None, s.started_at or 0.0))
    return report


def fetch_activity(get: Get, days: list[str] | None = None) -> list[dict[str, Any]]:
    """OpenRouter's daily totals per model and provider for the last 30 days."""
    base = GATEWAYS["openrouter"]["base"]
    if not days:
        return list(get(f"{base}/activity").get("data") or [])
    rows: list[dict[str, Any]] = []
    for day in days:
        rows += get(f"{base}/activity?{urllib.parse.urlencode({'date': day})}").get("data") or []
    return rows


def summarize_activity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model: calls a month and average sizes, ready for the estimate's form."""
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        m = by_model.setdefault(row["model"], {
            "model": row["model"], "days": set(), "requests": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0, "usd": 0.0,
            "providers": {},
        })
        m["days"].add(row.get("date"))
        for k in ("requests", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                  "cached_tokens"):
            m[k] += int(row.get(k) or 0)
        m["usd"] += float(row.get("usage") or 0) + float(row.get("byok_usage_inference") or 0)
        provider = row.get("provider_name") or "unknown"
        m["providers"][provider] = m["providers"].get(provider, 0) + int(row.get("requests") or 0)
    out = []
    for m in sorted(by_model.values(), key=lambda x: -x["usd"]):
        days = len(m.pop("days")) or 1
        n = m["requests"] or 1
        out.append({
            **m,
            "days_with_traffic": days,
            "monthly_calls": m["requests"] / days * 30,
            "avg_input_tokens": m["prompt_tokens"] / n,
            "avg_output_tokens": m["completion_tokens"] / n,
            "cached_share": m["cached_tokens"] / m["prompt_tokens"] if m["prompt_tokens"] else None,
            "monthly_usd": m["usd"] / days * 30,
        })
    return out
