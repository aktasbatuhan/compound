"""Caching is not optional on opt-in hosts, and this file is what enforces it.

A marker-less request to Doubleword or Anthropic gets a guaranteed 0% cache hit.
On an agent loop that re-reads its context every turn that is most of the bill:
measured on a real run, 59.3M input tokens at a 16:1 input-to-output ratio cost
$5.37 uncached against roughly $1.70 with markers.

The scan at the bottom is the part that matters. It fails the build when a new
module learns to talk to a cacheable host without going through the one helper
that marks requests.
"""

import re
from pathlib import Path

from compound.cache_policy import (
    MARKER_REQUIRED_HOSTS,
    is_marked,
    mark_cache_prefix,
    needs_marker,
)

SRC = Path(__file__).resolve().parents[1] / "src/compound"

# Modules that name a cacheable host but hand request construction to an
# upstream harness. They are only safe behind the marking proxy in orproxy.py,
# and each entry records why it cannot mark its own payloads.
DELEGATES_TO_PROXY = {
    "tau_sweep.py": "passes a base_url to the tau2 harness; point it at orproxy",
    "tau_gepa.py": "passes a base_url to GEPA; point it at orproxy",
    "bench.py": "resolves a base_url for external benchmarks; point them at orproxy",
}


def test_marker_attaches_to_plain_string_content():
    marked = mark_cache_prefix([{"role": "user", "content": "hello"}])
    block = marked[0]["content"][0]
    assert block["text"] == "hello"
    assert block["cache_control"]["type"] == "ephemeral"
    assert is_marked({"messages": marked})


def test_marker_lands_on_the_newest_message_so_the_whole_prefix_caches():
    conversation = [
        {"role": "system", "content": "tools"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    marked = mark_cache_prefix(conversation)
    assert is_marked({"messages": marked})
    assert marked[-1]["content"][0]["cache_control"]
    # Earlier turns stay untouched; the breakpoint covers everything before it.
    assert marked[0]["content"] == "tools"


def test_an_unmarked_payload_is_detected_as_unmarked():
    assert not is_marked({"messages": [{"role": "user", "content": "hello"}]})
    assert not is_marked({})


def test_needs_marker_knows_which_endpoints_cache_on_request():
    assert needs_marker("https://api.doubleword.ai/v1/chat/completions")
    assert needs_marker("https://api.anthropic.com/v1/messages")
    # Doubleword's Responses endpoint accepts the marker and never reports a hit,
    # so it is not a cacheable endpoint and must not be used for agent loops.
    assert not needs_marker("https://api.doubleword.ai/v1/responses")
    # OpenRouter upstreams cache implicitly; no marker is required.
    assert not needs_marker("https://openrouter.ai/api/v1/chat/completions")


def test_the_gateway_refuses_to_send_an_unmarked_request():
    source = (SRC / "agentic_gateway.py").read_text()
    assert "refusing to send an uncached request" in source
    assert "needs_marker(base) and not is_marked(payload)" in source


def test_the_agent_gateway_uses_the_caching_endpoint():
    source = (SRC / "agentic_gateway.py").read_text()
    assert "api.doubleword.ai/v1/chat/completions" in source
    assert "api.doubleword.ai/v1/responses" not in source
    assert 'mark_cache_prefix(payload["messages"])' in source


def test_no_module_talks_to_a_cacheable_host_without_marking_it():
    """The guard. A new unmarked paid path fails here instead of on the bill."""
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        if path.name in ("cache_policy.py", "orproxy.py"):
            continue
        text = path.read_text()
        if not any(host in text for host in MARKER_REQUIRED_HOSTS):
            continue
        marks = "mark_cache_prefix" in text or "cache_control" in text
        if marks or path.name in DELEGATES_TO_PROXY:
            continue
        offenders.append(path.name)
    assert not offenders, (
        f"{offenders} build requests for a cacheable host without a cache marker. "
        "Route them through compound.cache_policy.mark_cache_prefix, or add them to "
        "DELEGATES_TO_PROXY with the reason they cannot mark their own payloads."
    )


def test_every_delegating_module_still_exists_so_the_allowlist_cannot_rot():
    for name in DELEGATES_TO_PROXY:
        assert (SRC / name).exists(), f"{name} is allowlisted but gone; drop the entry"


def test_no_module_hardcodes_the_uncacheable_responses_endpoint():
    offenders = [
        p.name
        for p in sorted(SRC.glob("*.py"))
        if re.search(r"api\.doubleword\.ai/v1/responses", p.read_text())
        and p.name not in ("migration_io.py",)
    ]
    assert not offenders, f"{offenders} send agent traffic to the uncacheable endpoint"


def test_the_runner_halts_when_caching_is_asked_for_and_never_delivered():
    """The runtime check the static scan cannot make: a host that accepts the
    marker and returns nothing must stop the run, not bill a whole study."""
    from compound.agentic_run import CACHE_GATE_MIN_CALLS
    from compound.serving_metrics import cache_effectiveness

    asked_and_denied = [
        {
            "cache_requested": True,
            "usage": {"prompt_tokens": 9884, "prompt_tokens_details": {"cached_tokens": 0}},
        }
        for _ in range(CACHE_GATE_MIN_CALLS)
    ]
    assert cache_effectiveness(asked_and_denied)["requested_but_never_observed"]

    served = [
        {
            "cache_requested": True,
            "usage": {"prompt_tokens": 9884, "prompt_tokens_details": {"cached_tokens": 9805}},
        }
        for _ in range(CACHE_GATE_MIN_CALLS)
    ]
    assert not cache_effectiveness(served)["requested_but_never_observed"]

    runner = (SRC / "agentic_run.py").read_text()
    assert "prompt caching requested and never observed" in runner, "the halt was removed"
    assert "cache_effectiveness(asked)" in runner, "the gate must reuse the shared detector"


def test_the_declared_reasoning_effort_reaches_doubleword():
    """Dropped silently once already; chat completions takes a flat parameter."""
    gateway = (SRC / "agentic_gateway.py").read_text()
    assert 'payload["reasoning_effort"] = effort' in gateway
    assert 'self.spec["controls"].get("reasoning_effort")' in gateway


def test_missing_usage_never_settles_as_free_inference():
    gateway = (SRC / "agentic_gateway.py").read_text()
    assert "CACHE_WRITE_MULTIPLIER = 1.25" in gateway
    assert "cache_creation_input_tokens" in gateway
    assert "derived_responses_no_cache" not in gateway, "stale cost label"
