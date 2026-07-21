from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from compound.contracts import Usage


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    output: dict[str, Any]
    usage: Usage
    latency_ms: int


@dataclass(frozen=True, slots=True)
class BackgroundResponseSnapshot:
    response_id: str
    status: str
    output: dict[str, Any]
    request_latency_ms: int


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key_env: str,
        telemetry_path: str | Path | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 1,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable: {api_key_env}")
        self.name = name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None

    def _record_telemetry(self, record: dict[str, Any]) -> None:
        if self.telemetry_path is None:
            return
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        event_id = record.get("event_id")
        if event_id and self.telemetry_path.exists():
            for line in self.telemetry_path.read_text().splitlines():
                if line.strip() and json.loads(line).get("event_id") == event_id:
                    return
        with self.telemetry_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def record_telemetry(self, record: dict[str, Any]) -> None:
        """Append one normalized model-call event, idempotently when event_id is set."""
        self._record_telemetry(record)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        telemetry_context: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        request: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            request.update({"tools": tools, "tool_choice": "auto"})
        if extra_body:
            request["extra_body"] = extra_body
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        started = time.perf_counter()
        timestamp = datetime.now(UTC).isoformat()
        try:
            response = self.client.chat.completions.create(**request)
        except BaseException as error:
            latency_ms = round((time.perf_counter() - started) * 1000)
            self._record_telemetry(
                {
                    "timestamp": timestamp,
                    "provider": self.name,
                    "requested_model": model,
                    "status": "error",
                    "api_surface": "chat.completions",
                    "error_type": type(error).__name__,
                    "latency_ms": latency_ms,
                    "context": telemetry_context or {},
                }
            )
            raise
        latency_ms = round((time.perf_counter() - started) * 1000)
        raw = response.model_dump(mode="json")
        raw_usage = raw.get("usage") or {}
        details = raw_usage.get("completion_tokens_details") or {}
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens", 0) or 0,
            output_tokens=raw_usage.get("completion_tokens", 0) or 0,
            reasoning_tokens=details.get("reasoning_tokens", 0) or 0,
            cached_tokens=prompt_details.get("cached_tokens", 0) or 0,
        )
        choices = raw.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        e2e_output_tps = usage.output_tokens / (latency_ms / 1000) if latency_ms > 0 else None
        self._record_telemetry(
            {
                "timestamp": timestamp,
                "provider": self.name,
                "requested_model": model,
                "resolved_model": raw.get("model"),
                "status": "ok",
                "api_surface": "chat.completions",
                "latency_ms": latency_ms,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "cached_tokens": usage.cached_tokens,
                "finish_reason": finish_reason,
                "e2e_output_tps": e2e_output_tps,
                "context": telemetry_context or {},
            }
        )
        return ProviderResponse(output=raw, usage=usage, latency_ms=latency_ms)

    def list_models(self) -> list[str]:
        return sorted(model.id for model in self.client.models.list().data)

    def submit_background_response(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        service_tier: str = "flex",
        reasoning_effort: str | None = None,
    ) -> BackgroundResponseSnapshot:
        request: dict[str, Any] = {
            "model": model,
            "input": input,
            "background": True,
            "service_tier": service_tier,
        }
        if instructions is not None:
            request["instructions"] = instructions
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            request["reasoning"] = {"effort": reasoning_effort}

        started = time.perf_counter()
        response = self.client.responses.create(**request)
        latency_ms = round((time.perf_counter() - started) * 1000)
        raw = response.model_dump(mode="json")
        response_id = raw.get("id")
        status = raw.get("status")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("background response submission did not return an id")
        if not isinstance(status, str) or not status:
            raise ValueError("background response submission did not return a status")
        return BackgroundResponseSnapshot(
            response_id=response_id,
            status=status,
            output=raw,
            request_latency_ms=latency_ms,
        )

    def retrieve_background_response(self, response_id: str) -> BackgroundResponseSnapshot:
        started = time.perf_counter()
        response = self.client.responses.retrieve(response_id)
        latency_ms = round((time.perf_counter() - started) * 1000)
        raw = response.model_dump(mode="json")
        resolved_id = raw.get("id")
        status = raw.get("status")
        if resolved_id != response_id:
            raise ValueError(
                f"retrieved response id {resolved_id!r} does not match {response_id!r}"
            )
        if not isinstance(status, str) or not status:
            raise ValueError("retrieved background response did not return a status")
        return BackgroundResponseSnapshot(
            response_id=response_id,
            status=status,
            output=raw,
            request_latency_ms=latency_ms,
        )
