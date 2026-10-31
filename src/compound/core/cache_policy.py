"""Request opt-in caching without changing message text.

Markers request caching; only provider usage establishes a cache hit. This helper
is not proof that every request path or upstream supports caching.
"""

from typing import Any

#: Hosts for which this benchmark requires explicit cache markers.
MARKER_REQUIRED_HOSTS = ("api.doubleword.ai", "api.anthropic.com")

#: Legacy exclusion for this message-block helper. Current Doubleword Responses
#: supports top-level cache_control; it needs a different request builder.
UNCACHEABLE_ENDPOINTS = ("/v1/responses",)

CACHE_MARKER = {"type": "ephemeral", "ttl": "5m"}


def needs_marker(base_url: str) -> bool:
    """Whether our message-block request policy requires an explicit marker."""
    if any(path in base_url for path in UNCACHEABLE_ENDPOINTS):
        return False
    return any(host in base_url for host in MARKER_REQUIRED_HOSTS)


def mark_cache_prefix(messages: Any, *, ttl: str = "5m") -> Any:
    """Attach ``cache_control`` to the last content block of the last message.

    The marker is a breakpoint: everything up to and including it is cached, so
    marking the newest message caches the entire conversation prefix that the
    next turn will re-send.
    """
    if ttl not in {"5m", "1h"}:
        raise ValueError("cache TTL must be 5m or 1h")
    marker = {"type": "ephemeral", "ttl": ttl}
    if not isinstance(messages, list) or not messages:
        return messages
    msgs = list(messages)
    last = dict(msgs[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": dict(marker)}]
    elif isinstance(content, list) and content:
        blocks = list(content)
        final = dict(blocks[-1])
        final["cache_control"] = dict(marker)
        blocks[-1] = final
        last["content"] = blocks
    else:
        return messages
    msgs[-1] = last
    return msgs


def is_marked(payload: dict) -> bool:
    """True when the outgoing payload carries a cache marker."""
    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("cache_control") for block in content
        ):
            return True
    return False
