from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def summarize_model_calls(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                record["provider"],
                record["requested_model"],
                record.get("api_surface") or "chat.completions",
                record.get("requested_service_tier") or "default",
            )
        ].append(record)

    summaries = []
    for (provider, model, api_surface, service_tier), calls in sorted(groups.items()):
        successful = [call for call in calls if call["status"] == "ok"]
        timed = [call for call in successful if call.get("latency_ms") is not None]
        latencies = [float(call["latency_ms"]) for call in timed]
        throughputs = [
            float(call["e2e_output_tps"])
            for call in successful
            if call.get("e2e_output_tps") is not None
        ]
        output_tokens = sum(call.get("output_tokens", 0) for call in successful)
        timed_output_tokens = sum(call.get("output_tokens", 0) for call in timed)
        latency_ms_total = sum(latencies)
        summaries.append(
            {
                "provider": provider,
                "model": model,
                "api_surface": api_surface,
                "service_tier": service_tier,
                "calls": len(calls),
                "successful_calls": len(successful),
                "error_calls": len(calls) - len(successful),
                "timed_calls": len(latencies),
                "input_tokens": sum(call.get("input_tokens", 0) for call in successful),
                "output_tokens": output_tokens,
                "timed_output_tokens": timed_output_tokens,
                "reasoning_tokens": sum(
                    call.get("reasoning_tokens", 0) for call in successful
                ),
                "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
                "latency_ms_p50": _percentile(latencies, 0.50),
                # Percentiles below this sample size are not meaningful (p95 of a
                # handful of points is just the max); suppress rather than mislead.
                "latency_ms_p95": (
                    _percentile(latencies, 0.95) if len(latencies) >= 20 else None
                ),
                "e2e_output_tps_mean": (
                    statistics.fmean(throughputs) if throughputs else None
                ),
                "e2e_output_tps_p50": _percentile(throughputs, 0.50),
                # Aggregate (sum output tokens / sum latency) — the correct
                # throughput to compare across routes. The per-call mean above is a
                # mean-of-ratios and overweights short/fast calls.
                "e2e_output_tps_aggregate": (
                    timed_output_tokens / (latency_ms_total / 1000)
                    if latency_ms_total > 0
                    else None
                ),
            }
        )
    return {
        "metric_note": (
            "e2e_output_tps is completion tokens divided by full request latency; "
            "it includes queueing and time-to-first-token and is not server decode-only TPS. "
            "e2e_output_tps_mean is a per-call mean-of-ratios; prefer e2e_output_tps_aggregate "
            "for cross-route comparison. Aggregate TPS uses only calls with recorded latency. "
            "latency_ms_p95 is suppressed below 20 timed calls."
            " Rows are separated by provider, model, API surface, and requested service tier."
        ),
        "models": summaries,
    }


def ingest_tau_results(
    results_path: str | Path,
    *,
    telemetry_path: str | Path,
    agent_provider: str,
    agent_model: str,
    user_provider: str,
    user_model: str,
    source_label: str | None = None,
    agent_upstream: str | None = None,
) -> int:
    """Normalize tau message-level timings and usage into the shared JSONL schema.

    ``source_label`` is the portable identity of the results file used to build the
    idempotency key. It defaults to ``<parent-dir>/<filename>`` so the dedup key does
    not embed an absolute local filesystem path (which would leak the machine path
    and silently break dedup when the repo is cloned or run from a different cwd).
    """
    source = Path(results_path)
    destination = Path(telemetry_path)
    label = source_label or f"{source.parent.name}/{source.name}"
    existing_ids: set[str] = set()
    if destination.exists():
        for line in destination.read_text().splitlines():
            if not line.strip():
                continue
            source_id = json.loads(line).get("context", {}).get("source_id")
            if source_id:
                existing_ids.add(source_id)

    payload = json.loads(source.read_text())
    appended = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as stream:
        for simulation in payload.get("simulations", []):
            for message in simulation.get("messages", []):
                usage = message.get("usage")
                if not usage:
                    continue
                role = message.get("role")
                if role == "assistant":
                    provider, model, call_type = agent_provider, agent_model, "tau_agent"
                elif role == "user":
                    provider, model, call_type = user_provider, user_model, "tau_user"
                else:
                    continue
                source_id = ":".join(
                    [
                        label,
                        str(simulation.get("id")),
                        str(message.get("turn_idx")),
                        role,
                    ]
                )
                if source_id in existing_ids:
                    continue
                raw = message.get("raw_data") or {}
                raw_usage = raw.get("usage") or {}
                details = raw_usage.get("completion_tokens_details") or {}
                choices = raw.get("choices") or []
                latency_seconds = message.get("generation_time_seconds")
                output_tokens = int(usage.get("completion_tokens", 0) or 0)
                record = {
                    "timestamp": message.get("timestamp"),
                    "provider": provider,
                    "requested_model": model,
                    "resolved_model": raw.get("model"),
                    # The upstream host we PINNED (agent side only) and the one
                    # OpenRouter reports having SERVED; a mismatch is a routing bug.
                    "requested_upstream": agent_upstream if role == "assistant" else None,
                    "served_upstream": raw.get("provider"),
                    "status": "ok",
                    "api_surface": "chat.completions",
                    "latency_ms": (
                        round(float(latency_seconds) * 1000)
                        if latency_seconds is not None
                        else None
                    ),
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": output_tokens,
                    "reasoning_tokens": int(details.get("reasoning_tokens", 0) or 0),
                    "cached_tokens": int(
                        (raw_usage.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0
                        )
                        or 0
                    ),
                    "finish_reason": (
                        choices[0].get("finish_reason") if choices else None
                    ),
                    "e2e_output_tps": (
                        output_tokens / float(latency_seconds)
                        if latency_seconds and float(latency_seconds) > 0
                        else None
                    ),
                    "context": {
                        "benchmark": "tau_bench",
                        "call_type": call_type,
                        "task_id": simulation.get("task_id"),
                        "simulation_id": simulation.get("id"),
                        "turn_idx": message.get("turn_idx"),
                        "source_id": source_id,
                    },
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                existing_ids.add(source_id)
                appended += 1
    return appended
