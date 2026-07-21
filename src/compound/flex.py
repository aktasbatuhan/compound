from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compound.budget import BudgetLedger
from compound.contracts import Usage
from compound.costs import TokenPrices, estimate_cost
from compound.providers import OpenAICompatibleProvider

PENDING_RESPONSE_STATUSES = frozenset({"queued", "in_progress"})


@dataclass(frozen=True, slots=True)
class FlexRequest:
    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    cache_discriminator: str | None = None
    telemetry_context: dict[str, Any] = field(default_factory=dict)


def flex_fingerprint(request: FlexRequest, *, service_tier: str = "flex") -> str:
    payload = {
        "schema_version": 1,
        "model": request.model,
        "input": request.input,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "service_tier": service_tier,
        "background": True,
    }
    # Preserve the original no-effort/no-trial fingerprints so existing paid
    # responses remain reusable after these optional controls are introduced.
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.cache_discriminator is not None:
        payload["cache_discriminator"] = request.cache_discriminator
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def responses_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and content.get("type") in {"output_text", "text"}:
                parts.append(text)
    return "".join(parts)


def responses_usage(response: dict[str, Any]) -> Usage:
    raw = response.get("usage") or {}
    output_details = raw.get("output_tokens_details") or {}
    input_details = raw.get("input_tokens_details") or {}
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
        cached_tokens=int(input_details.get("cached_tokens", 0) or 0),
    )


def _finish_reason(response: dict[str, Any]) -> str | None:
    incomplete = response.get("incomplete_details") or {}
    if incomplete.get("reason"):
        return str(incomplete["reason"])
    status = response.get("status")
    return str(status) if status else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(started_at: str, ended_at: str) -> int:
    started = datetime.fromisoformat(started_at)
    ended = datetime.fromisoformat(ended_at)
    return max(0, round((ended - started).total_seconds() * 1000))


class CachedFlexRunner:
    """Submit and resume cached Open Responses background jobs without duplicate calls."""

    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        prices_by_model: dict[str, TokenPrices],
        ledger: BudgetLedger,
        cache_dir: str | Path,
        poll_interval_seconds: float = 2.0,
        timeout_seconds: float = 3600.0,
        reserve_per_new_request_usd: float = 0.25,
        service_tier: str = "flex",
        headroom_checker: Callable[[float], None] | None = None,
    ) -> None:
        if poll_interval_seconds < 0:
            raise ValueError("poll interval cannot be negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        if reserve_per_new_request_usd < 0:
            raise ValueError("request reserve cannot be negative")
        self.provider = provider
        self.prices_by_model = prices_by_model
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.reserve_per_new_request_usd = reserve_per_new_request_usd
        self.service_tier = service_tier
        self.headroom_checker = headroom_checker or ledger.require_headroom

    def _cache_path(self, fingerprint: str) -> Path:
        return self.cache_dir / f"{fingerprint}.json"

    def _pricing_payload(self, model: str) -> dict[str, float]:
        try:
            prices = self.prices_by_model[model]
        except KeyError as error:
            raise KeyError(f"no {self.service_tier} pricing configured for {model}") from error
        return {
            "input_per_million": prices.input_per_million,
            "output_per_million": prices.output_per_million,
        }

    def _load_cached(self, request: FlexRequest, fingerprint: str) -> dict[str, Any] | None:
        path = self._cache_path(fingerprint)
        if not path.exists():
            return None
        state = json.loads(path.read_text())
        if state.get("fingerprint") != fingerprint or state.get("requested_model") != request.model:
            raise ValueError(f"invalid Flex cache record: {path}")
        return state

    def _submit(self, request: FlexRequest, fingerprint: str) -> dict[str, Any]:
        snapshot = self.provider.submit_background_response(
            model=request.model,
            input=request.input,
            instructions=request.instructions,
            max_output_tokens=request.max_output_tokens,
            service_tier=self.service_tier,
            reasoning_effort=request.reasoning_effort,
        )
        submitted_at = _utc_now()
        state = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "response_id": snapshot.response_id,
            "requested_model": request.model,
            "requested_service_tier": self.service_tier,
            "reasoning_effort": request.reasoning_effort,
            "background": True,
            "status": snapshot.status,
            "submitted_at": submitted_at,
            "submit_latency_ms": snapshot.request_latency_ms,
            "poll_count": 0,
            "poll_request_latency_ms_total": 0,
            "pricing_usd_per_million_tokens": self._pricing_payload(request.model),
            "telemetry_context": request.telemetry_context,
            "raw_response": snapshot.output,
        }
        _write_json_atomic(self._cache_path(fingerprint), state)
        return state

    def _poll(self, state: dict[str, Any]) -> None:
        snapshot = self.provider.retrieve_background_response(state["response_id"])
        state["status"] = snapshot.status
        state["raw_response"] = snapshot.output
        state["last_polled_at"] = _utc_now()
        state["poll_count"] = int(state.get("poll_count", 0)) + 1
        state["poll_request_latency_ms_total"] = (
            int(state.get("poll_request_latency_ms_total", 0)) + snapshot.request_latency_ms
        )
        _write_json_atomic(self._cache_path(state["fingerprint"]), state)

    def _finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        cached_result = state.get("result")
        if isinstance(cached_result, dict):
            return cached_result

        response = state.get("raw_response") or {}
        usage = responses_usage(response)
        pricing = state["pricing_usd_per_million_tokens"]
        prices = TokenPrices(
            input_per_million=float(pricing["input_per_million"]),
            output_per_million=float(pricing["output_per_million"]),
        )
        cost_usd = estimate_cost(usage, prices)
        fingerprint = state["fingerprint"]
        self.ledger.record(cost_usd, label=f"flex:{fingerprint}")

        completed_at = state.get("completed_at") or _utc_now()
        e2e_latency_ms = max(
            int(state.get("submit_latency_ms", 0)),
            _elapsed_ms(state["submitted_at"], completed_at),
        )
        output_text = responses_output_text(response)
        resolved_service_tier = response.get("service_tier")
        result = {
            "fingerprint": fingerprint,
            "response_id": state["response_id"],
            "requested_model": state["requested_model"],
            "resolved_model": response.get("model"),
            "status": state["status"],
            "requested_service_tier": state["requested_service_tier"],
            "resolved_service_tier": resolved_service_tier,
            "billing_tier_confirmed_from_response": resolved_service_tier
            == state["requested_service_tier"],
            "background": True,
            "reasoning_effort": state.get("reasoning_effort"),
            "submitted_at": state["submitted_at"],
            "completed_at": completed_at,
            "submit_latency_ms": int(state.get("submit_latency_ms", 0)),
            "e2e_latency_ms": e2e_latency_ms,
            "poll_count": int(state.get("poll_count", 0)),
            "poll_request_latency_ms_total": int(state.get("poll_request_latency_ms_total", 0)),
            "output_text": output_text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cached_tokens": usage.cached_tokens,
            "finish_reason": _finish_reason(response),
            "e2e_output_tps": (
                usage.output_tokens / (e2e_latency_ms / 1000) if e2e_latency_ms > 0 else None
            ),
            "estimated_cost_usd": cost_usd,
        }
        telemetry_status = "ok" if state["status"] == "completed" else "error"
        self.provider.record_telemetry(
            {
                "event_id": f"flex:{fingerprint}",
                "timestamp": state["submitted_at"],
                "provider": self.provider.name,
                "requested_model": state["requested_model"],
                "resolved_model": response.get("model"),
                "status": telemetry_status,
                "terminal_status": state["status"],
                "api_surface": "responses",
                "requested_service_tier": state["requested_service_tier"],
                "resolved_service_tier": resolved_service_tier,
                "billing_tier_confirmed_from_response": result[
                    "billing_tier_confirmed_from_response"
                ],
                "background": True,
                "reasoning_effort": state.get("reasoning_effort"),
                "response_id": state["response_id"],
                "submitted_at": state["submitted_at"],
                "completed_at": completed_at,
                "latency_ms": e2e_latency_ms,
                "submit_latency_ms": result["submit_latency_ms"],
                "poll_count": result["poll_count"],
                "poll_request_latency_ms_total": result["poll_request_latency_ms_total"],
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "cached_tokens": usage.cached_tokens,
                "finish_reason": result["finish_reason"],
                "e2e_output_tps": result["e2e_output_tps"],
                "estimated_cost_usd": cost_usd,
                "context": {
                    **state.get("telemetry_context", {}),
                    "fingerprint": fingerprint,
                },
            }
        )
        state["completed_at"] = completed_at
        state["result"] = result
        _write_json_atomic(self._cache_path(fingerprint), state)
        return result

    def run_many(self, requests: list[FlexRequest]) -> list[dict[str, Any]]:
        if not requests:
            return []
        keyed: list[tuple[FlexRequest, str]] = [
            (request, flex_fingerprint(request, service_tier=self.service_tier))
            for request in requests
        ]
        fingerprints = [fingerprint for _, fingerprint in keyed]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("duplicate Flex requests are not allowed in one run")

        states: dict[str, dict[str, Any]] = {}
        new_requests: list[tuple[FlexRequest, str]] = []
        for request, fingerprint in keyed:
            cached = self._load_cached(request, fingerprint)
            if cached is None:
                new_requests.append((request, fingerprint))
            else:
                states[fingerprint] = cached

        self.headroom_checker(self.reserve_per_new_request_usd * len(new_requests))
        for request, fingerprint in new_requests:
            states[fingerprint] = self._submit(request, fingerprint)

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            pending = False
            for _, fingerprint in keyed:
                state = states[fingerprint]
                if isinstance(state.get("result"), dict):
                    continue
                if state.get("status") in PENDING_RESPONSE_STATUSES:
                    pending = True
                    self._poll(state)
                if state.get("status") not in PENDING_RESPONSE_STATUSES:
                    self._finalize(state)

            if all(
                isinstance(states[fingerprint].get("result"), dict) for fingerprint in fingerprints
            ):
                return [states[fingerprint]["result"] for fingerprint in fingerprints]
            if time.monotonic() >= deadline:
                ids = [
                    states[fingerprint]["response_id"]
                    for fingerprint in fingerprints
                    if not isinstance(states[fingerprint].get("result"), dict)
                ]
                raise TimeoutError(
                    "Flex responses did not reach terminal status before timeout; "
                    f"cached response ids: {', '.join(ids)}"
                )
            if pending and self.poll_interval_seconds:
                time.sleep(self.poll_interval_seconds)
