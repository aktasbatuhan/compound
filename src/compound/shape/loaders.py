"""Read calls from the formats teams already have and reduce them to shapes.

Four inputs are understood, detected per record so one file may mix them:

- ``compound.trace`` documents from the trace pipeline (``bun run compound
  import`` writes them; Langfuse, plain JSON and OTel exports all land here).
- Compound call-ledger rows, which carry sizes and timing but no messages, so
  their prefix chain stays unknown.
- Request logs: one JSON object per call with ``request`` (OpenAI-style
  ``model``/``messages``/``tools``) and ``response`` (with ``usage``), plus a
  timestamp. This is the lowest-effort export from a gateway or a logger.
- ``compound.shape`` records, already reduced, as the SDK writes them.

A record that matches none of these is counted and skipped, never guessed at.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from compound.shape.records import SCHEMA, CallShape, shape_from_messages


@dataclass
class LoadReport:
    """What a load produced, so a profile can say what it is built on."""

    shapes: list[CallShape] = field(default_factory=list)
    by_source: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)


def parse_time(value: Any) -> float | None:
    """Epoch seconds from an epoch number or an ISO-8601 string; else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        # Milliseconds are common in gateway logs; no real call predates 2001.
        return float(value) / 1000 if value > 1e12 else float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _openai_usage(usage: dict[str, Any] | None) -> dict[str, int | None]:
    """Token counts from OpenAI- or Anthropic-style usage blocks."""
    usage = usage or {}
    prompt = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
    cached = None
    # Chat Completions and the Responses API both count cached tokens inside the
    # input total; only the field names differ.
    details = usage.get("prompt_tokens_details", usage.get("input_tokens_details"))
    if isinstance(details, dict):
        cached = _int(details.get("cached_tokens"))
    if cached is None and "cache_read_input_tokens" in usage:
        cached = _int(usage.get("cache_read_input_tokens"))
        # Anthropic reports uncached input separately from cache reads/writes.
        if prompt is not None and "prompt_tokens" not in usage:
            prompt += (cached or 0) + (_int(usage.get("cache_creation_input_tokens")) or 0)
    reasoning = None
    cdetails = usage.get("completion_tokens_details", usage.get("output_tokens_details"))
    if isinstance(cdetails, dict):
        reasoning = _int(cdetails.get("reasoning_tokens"))
    return {
        "input_tokens": prompt,
        "output_tokens": _int(usage.get("completion_tokens", usage.get("output_tokens"))),
        "reasoning_tokens": reasoning,
        "cached_input_tokens": cached,
    }


def shapes_from_trace(doc: dict[str, Any], salt: bytes) -> list[CallShape]:
    """Every model call in a ``compound.trace`` document."""
    out = []
    for step in doc.get("steps") or []:
        if not isinstance(step, dict) or step.get("type") != "model_call":
            continue
        usage = step.get("usage") or {}
        started = parse_time(step.get("started_at"))
        ended = parse_time(step.get("ended_at"))
        messages = step.get("input") if isinstance(step.get("input"), list) else []
        out.append(
            shape_from_messages(
                messages,
                tools=step.get("tools_available"),
                salt=salt,
                started_at=started,
                ended_at=ended,
                latency_ms=(ended - started) * 1000 if started and ended else None,
                model=step.get("resolved_model") or step.get("model"),
                provider=step.get("provider"),
                task_key=doc.get("task_key"),
                trace_id=doc.get("trace_id"),
                input_tokens=_int(usage.get("input_tokens")),
                output_tokens=_int(usage.get("output_tokens")),
                reasoning_tokens=_int(usage.get("reasoning_tokens")),
                cached_input_tokens=_int(usage.get("cached_input_tokens")),
                status="error" if step.get("error") else "ok",
                finish_reason=step.get("finish_reason"),
                source="trace",
            )
        )
    return out


def shape_from_ledger(row: dict[str, Any]) -> CallShape:
    """A call-ledger row. ``ts`` is written when the call completes."""
    ended = parse_time(row.get("ts"))
    latency = row.get("latency_ms")
    started = ended - latency / 1000 if ended is not None and latency is not None else None
    status = row.get("status")
    return CallShape(
        started_at=started,
        ended_at=ended,
        latency_ms=latency,
        model=row.get("model_echo") or row.get("model_requested"),
        provider=row.get("provider_echo") or row.get("upstream"),
        task_key=row.get("run_label"),
        input_tokens=_int(row.get("prompt_tokens")),
        output_tokens=_int(row.get("completion_tokens")),
        reasoning_tokens=_int(row.get("reasoning_tokens")),
        cached_input_tokens=_int(row.get("cached_tokens")),
        message_count=_int(row.get("messages")),
        status="ok" if status == 200 and not row.get("abandoned") else "error",
        finish_reason=row.get("finish_reason"),
        source="ledger",
    )


def shape_from_request_log(row: dict[str, Any], salt: bytes) -> CallShape:
    """A ``{request, response, ts}`` log line from a gateway or logger."""
    request = row.get("request") or {}
    response = row.get("response") or {}
    started = parse_time(row.get("started_at", row.get("ts", row.get("timestamp"))))
    ended = parse_time(row.get("ended_at"))
    latency = row.get("latency_ms")
    if latency is None and started is not None and ended is not None:
        latency = (ended - started) * 1000
    choices = response.get("choices") if isinstance(response, dict) else None
    finish = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish = choices[0].get("finish_reason")
    messages = request.get("messages") if isinstance(request.get("messages"), list) else []
    return shape_from_messages(
        messages,
        tools=request.get("tools"),
        salt=salt,
        started_at=started,
        ended_at=ended,
        latency_ms=latency,
        model=(response.get("model") if isinstance(response, dict) else None)
        or request.get("model"),
        provider=row.get("provider")
        or (response.get("provider") if isinstance(response, dict) else None),
        task_key=row.get("task_key"),
        trace_id=row.get("trace_id"),
        status="error" if row.get("error") else "ok",
        finish_reason=finish,
        source="request_log",
        **_openai_usage(response.get("usage") if isinstance(response, dict) else None),
    )


def classify(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    if record.get("schema") == SCHEMA:
        return "shape"
    if record.get("schema") == "compound.trace":
        return "trace"
    if isinstance(record.get("request"), dict):
        return "request_log"
    if "route" in record and "latency_ms" in record and "ts" in record:
        return "ledger"
    return None


def _records(path: Path) -> Iterator[Any]:
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("[") or (stripped.startswith("{") and "\n{" not in stripped):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            yield from data if isinstance(data, list) else [data]
            return
    for line in text.splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None


def load_shapes(paths: Iterable[str | Path], salt: bytes) -> LoadReport:
    """Load every call from ``paths`` (files or directories of .json/.jsonl)."""
    report = LoadReport()
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix in (".json", ".jsonl"))
        else:
            files.append(p)
    for path in files:
        for record in _records(path):
            kind = classify(record)
            if kind == "shape":
                shapes = [CallShape.from_json(record)]
            elif kind == "trace":
                shapes = shapes_from_trace(record, salt)
            elif kind == "request_log":
                shapes = [shape_from_request_log(record, salt)]
            elif kind == "ledger":
                shapes = [shape_from_ledger(record)]
            else:
                report.skipped["unrecognised"] += 1
                continue
            report.shapes += shapes
            report.by_source[kind] += len(shapes)
    report.shapes.sort(key=lambda s: (s.started_at is None, s.started_at or 0.0))
    return report
