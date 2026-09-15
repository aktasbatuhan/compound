"""One place that decides how a paid request opts into prompt caching.

Doubleword and Anthropic-style endpoints cache only when the request carries an
explicit ``cache_control`` marker. A marker-less agent loop re-reads its whole
context at full price on every turn, which on a 16:1 input-to-output workload is
most of the bill. Every module that builds a paid request for a cacheable host
must route through :func:`mark_cache_prefix`, and ``tests/test_cache_policy.py``
fails the build if one does not.

Caching does not change the tokens a model sees, so it never changes an answer.
It changes what the run costs, which is why it is not optional.
"""

from typing import Any

#: Hosts whose caching is opt-in: no marker means a guaranteed 0% hit rate.
MARKER_REQUIRED_HOSTS = ("api.doubleword.ai", "api.anthropic.com")

#: Doubleword serves prompt caching on chat completions only. Its Responses
#: endpoint accepts the marker and never reports a hit, measured 2026-09-15.
UNCACHEABLE_ENDPOINTS = ("/v1/responses",)

CACHE_MARKER = {"type": "ephemeral", "ttl": "5m"}


def needs_marker(base_url: str) -> bool:
    """True when this endpoint caches, but only if asked explicitly."""
    if any(path in base_url for path in UNCACHEABLE_ENDPOINTS):
        return False
    return any(host in base_url for host in MARKER_REQUIRED_HOSTS)


def mark_cache_prefix(messages: Any) -> Any:
    """Attach ``cache_control`` to the last content block of the last message.

    The marker is a breakpoint: everything up to and including it is cached, so
    marking the newest message caches the entire conversation prefix that the
    next turn will re-send.
    """
    if not isinstance(messages, list) or not messages:
        return messages
    msgs = list(messages)
    last = dict(msgs[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": dict(CACHE_MARKER)}]
    elif isinstance(content, list) and content:
        blocks = list(content)
        final = dict(blocks[-1])
        final["cache_control"] = dict(CACHE_MARKER)
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
