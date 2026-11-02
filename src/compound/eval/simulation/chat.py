"""One chat-completions call per turn, behind a small function type.

The simulation only needs "send these messages and tools, get the assistant
message back with usage and cost". Keeping that a plain function lets tests
script a model, and lets the real path reuse the engine's provider client with
the route's pinning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from compound.core.providers_registry import ProviderSpec, openrouter_provider_block


@dataclass(frozen=True)
class ChatResult:
    message: dict[str, Any]
    usage: dict[str, Any]
    cost_usd: float | None
    latency_ms: int


ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], ChatResult]


def provider_chat(spec: ProviderSpec, model: str, *, temperature: float | None = None) -> ChatFn:
    """A ChatFn that calls ``model`` on the host ``spec`` pins."""
    from compound.core.providers import OpenAICompatibleProvider

    client = OpenAICompatibleProvider(
        name=spec.label, base_url=spec.forward_base_url, api_key_env=spec.required_key_env()
    )
    extra: dict[str, Any] = {}
    if spec.kind == "openrouter":
        extra["usage"] = {"include": True}
        if spec.upstream:
            extra["provider"] = openrouter_provider_block(spec.upstream)
    if spec.service_tier:
        extra["service_tier"] = spec.service_tier
    if temperature is not None:
        extra["temperature"] = temperature

    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResult:
        response = client.complete(model=model, messages=messages, tools=tools,
                                   extra_body=extra or None)
        raw = response.output
        choices = raw.get("choices") or [{}]
        message = dict(choices[0].get("message") or {})
        message["role"] = "assistant"
        return ChatResult(
            message={k: v for k, v in message.items()
                     if k in ("role", "content", "tool_calls") and v is not None},
            usage=raw.get("usage") or {},
            cost_usd=(raw.get("usage") or {}).get("cost"),
            latency_ms=response.latency_ms,
        )

    return chat
