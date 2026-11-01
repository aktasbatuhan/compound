"""The ``compound.shape`` record, built in the app without keeping any text.

This mirrors ``compound.shape.records`` in the main package byte for byte (a
test there checks it), so the SDK can stay dependency-free while writing
records the ``compound shape`` and ``compound estimate`` commands read as-is.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA = "compound.shape"
SCHEMA_VERSION = 1
HASH_CHARS = 16
TURKISH_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _message_text(message: dict[str, Any]) -> str:
    keep = {k: message.get(k) for k in ("role", "content", "tool_calls", "tool_call_id", "name")}
    return _canonical({k: v for k, v in keep.items() if v is not None})


def _plain_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


def lang_hint(text: str) -> str | None:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    if sum(ch in TURKISH_LETTERS for ch in letters) / len(letters) >= 0.01:
        return "tr"
    latin = sum(ch.isascii() or "À" <= ch <= "ɏ" for ch in letters)
    return "latin" if latin / len(letters) >= 0.9 else "other"


def prefix_chain(
    messages: list[dict[str, Any]], tools: list[Any] | None, salt: bytes
) -> list[list[Any]]:
    digest = hashlib.sha256(salt)
    chain: list[list[Any]] = []
    chars = 0
    for text in [_canonical(tools or [])] + [_message_text(m) for m in messages]:
        digest.update(b"\x00" + text.encode("utf-8"))
        chars += len(text)
        chain.append([digest.copy().hexdigest()[:HASH_CHARS], chars])
    return chain


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def usage_counts(usage: Any) -> dict[str, int | None]:
    """Token counts from an OpenAI, Responses-API or Anthropic usage block."""
    if usage is not None and not isinstance(usage, dict):
        dump = getattr(usage, "model_dump", None)
        usage = dump() if callable(dump) else dict(getattr(usage, "__dict__", {}))
    usage = usage or {}
    prompt = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
    cached = None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if isinstance(details, dict):
        cached = _int(details.get("cached_tokens"))
    if cached is None and "cache_read_input_tokens" in usage:
        cached = _int(usage.get("cache_read_input_tokens"))
        if prompt is not None and "prompt_tokens" not in usage:
            prompt += (cached or 0) + (_int(usage.get("cache_creation_input_tokens")) or 0)
    reasoning = None
    cdetails = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if isinstance(cdetails, dict):
        reasoning = _int(cdetails.get("reasoning_tokens"))
    return {
        "input_tokens": prompt,
        "output_tokens": _int(usage.get("completion_tokens", usage.get("output_tokens"))),
        "reasoning_tokens": reasoning,
        "cached_input_tokens": cached,
    }


def build_record(
    *,
    messages: list[dict[str, Any]],
    tools: list[Any] | None,
    salt: bytes,
    started_at: float | None,
    ended_at: float | None,
    model: str | None,
    usage: Any = None,
    provider: str | None = None,
    task_key: str | None = None,
    trace_id: str | None = None,
    finish_reason: str | None = None,
    error: bool = False,
) -> dict[str, Any]:
    chain = prefix_chain(messages, tools, salt)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "ended_at": ended_at,
        "latency_ms": (ended_at - started_at) * 1000
        if started_at is not None and ended_at is not None
        else None,
        "model": model,
        "provider": provider,
        "task_key": task_key,
        "trace_id": trace_id,
        "message_count": len(messages),
        "roles": [str(m.get("role") or "user") for m in messages],
        "tool_count": len(tools or []),
        "prompt_chars": chain[-1][1] if chain else 0,
        "lang_hint": lang_hint(_plain_text(messages)),
        "prefix_chain": chain,
        "status": "error" if error else "ok",
        "finish_reason": finish_reason,
        "source": "sdk",
        **usage_counts(usage),
    }
    return {k: v for k, v in record.items() if v is not None}
