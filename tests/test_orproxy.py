from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from compound.orproxy import inject, serve_provider, target_url
from compound.providers_registry import ProviderSpec, parse_provider


PIN = {"only": ["deepinfra"], "allow_fallbacks": False, "require_parameters": True}


def test_inject_adds_openrouter_provider_only():
    spec = parse_provider("openrouter/deepinfra")
    body = {"model": "deepseek/deepseek-v4-flash-0731", "messages": []}
    out = inject(body, spec)
    assert out["provider"] == PIN
    assert out["messages"] == []  # untouched
    assert body == {"model": "deepseek/deepseek-v4-flash-0731", "messages": []}  # non-destructive


def test_inject_overrides_caller_supplied_routing():
    spec = parse_provider("openrouter/baseten/fp8")
    out = inject({"provider": {"only": ["someone-else"]}}, spec)
    assert out["provider"]["only"] == ["baseten"]  # base provider slug, not the tag


def test_inject_adds_service_tier_for_flex():
    out = inject({"model": "x"}, parse_provider("doubleword/flex"))
    assert out["service_tier"] == "flex"


def test_inject_noop_for_realtime():
    out = inject({"model": "x"}, parse_provider("doubleword/realtime"))
    assert "service_tier" not in out and "provider" not in out


def test_inject_marks_doubleword_cache_prefix_when_enabled(monkeypatch):
    monkeypatch.setenv("COMPOUND_DW_CACHE", "1")
    body = {"model": "x", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]}
    out = inject(body, parse_provider("doubleword/realtime"))
    last = out["messages"][-1]["content"]
    assert last == [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]
    # earlier messages untouched, original body not mutated
    assert out["messages"][0]["content"] == "s"
    assert body["messages"][-1]["content"] == "hi"
    # block-form content gets the marker on its final block
    body2 = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}
    out2 = inject(body2, parse_provider("doubleword/flex"))
    assert out2["messages"][-1]["content"][-1]["cache_control"]["type"] == "ephemeral"
    assert "cache_control" not in out2["messages"][-1]["content"][0]


def test_inject_cache_marker_never_touches_openrouter(monkeypatch):
    # OpenRouter's strategy is "implicit": it caches on its own, so the proxy
    # adds no marker even with the opt-in on.
    monkeypatch.setenv("COMPOUND_DW_CACHE", "1")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out = inject(body, parse_provider("openrouter/deepinfra"))
    assert out["messages"][-1]["content"] == "hi"


def test_inject_no_cache_marker_without_env(monkeypatch):
    monkeypatch.delenv("COMPOUND_DW_CACHE", raising=False)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out = inject(body, parse_provider("doubleword/realtime"))
    assert out["messages"][-1]["content"] == "hi"


def test_inject_cache_marker_is_strategy_driven_not_kind(monkeypatch):
    # A non-doubleword host that declares explicit_marker gets the marker; a
    # doubleword-kind spec overridden to implicit does not. Proves the injection
    # follows cache_strategy, not a hardcoded host name (issue #43).
    monkeypatch.setenv("COMPOUND_DW_CACHE", "1")
    direct_marker = ProviderSpec(
        token="direct/vllm",
        kind="direct",
        base_url="https://x/v1",
        api_key_env="VLLM_KEY",
        cache_strategy="explicit_marker",
    )
    out = inject({"messages": [{"role": "user", "content": "hi"}]}, direct_marker)
    assert out["messages"][-1]["content"][-1]["cache_control"]["type"] == "ephemeral"

    dw_implicit = ProviderSpec(
        token="doubleword/realtime",
        kind="doubleword",
        base_url="https://api.doubleword.ai/v1",
        api_key_env="DOUBLEWORD_API_KEY",
        cache_strategy="implicit",
    )
    out2 = inject({"messages": [{"role": "user", "content": "hi"}]}, dw_implicit)
    assert out2["messages"][-1]["content"] == "hi"


def test_cache_optin_enabled_reads_env(monkeypatch):
    from compound.orproxy import cache_optin_enabled

    monkeypatch.setenv("COMPOUND_DW_CACHE", "on")
    assert cache_optin_enabled() is True
    monkeypatch.setenv("COMPOUND_DW_CACHE", "0")
    assert cache_optin_enabled() is False
    monkeypatch.delenv("COMPOUND_DW_CACHE", raising=False)
    assert cache_optin_enabled() is False


@pytest.mark.parametrize(
    "base,path,expected",
    [
        ("https://openrouter.ai/api/v1", "/v1/chat/completions", "https://openrouter.ai/api/v1/chat/completions"),
        ("https://openrouter.ai/api/v1/", "/v1/models", "https://openrouter.ai/api/v1/models"),
        ("https://host/v1", "/chat/completions", "https://host/v1/chat/completions"),
    ],
)
def test_target_url_avoids_doubled_v1(base, path, expected):
    assert target_url(base, path) == expected


class _FakeUpstream(BaseHTTPRequestHandler):
    received: dict = {}

    def log_message(self, *a):  # noqa: D401
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        _FakeUpstream.received = {
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(length)),
        }
        payload = json.dumps({"id": "ok", "choices": [{"message": {"content": "hi"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def test_proxy_forwards_with_pinning_and_auth(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sekret")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    up_port = upstream.server_address[1]

    # a spec whose forward target is the fake upstream, still pinning deepinfra
    spec = ProviderSpec(
        token="openrouter/deepinfra",
        kind="openrouter",
        base_url=f"http://127.0.0.1:{up_port}/v1",
        api_key_env="OPENROUTER_API_KEY",
        upstream="deepinfra",
    )
    try:
        with serve_provider(spec) as base:
            req = urlrequest.Request(
                base + "/chat/completions",
                data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "yo"}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=5) as resp:
                got = json.loads(resp.read())
    finally:
        upstream.shutdown()

    assert got["id"] == "ok"
    rec = _FakeUpstream.received
    assert rec["path"] == "/v1/chat/completions"  # base ends /v1, request /v1 stripped then rejoined
    assert rec["auth"] == "Bearer sekret"
    assert rec["body"]["provider"] == PIN
    assert rec["body"]["messages"][0]["content"] == "yo"


class _FlakyUpstream(BaseHTTPRequestHandler):
    """429s twice (with Retry-After), then succeeds; also serves a permanent 400."""

    hits = 0

    def log_message(self, *a):  # noqa: D401
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        _FlakyUpstream.hits += 1
        if self.path.endswith("/always-400"):
            payload = json.dumps({"error": {"code": 400, "message": "bad capability"}}).encode()
            self.send_response(400)
        elif _FlakyUpstream.hits <= 2:
            payload = json.dumps({"error": {"code": 429, "message": "rate limited"}}).encode()
            self.send_response(429)
            self.send_header("Retry-After", "0")
        else:
            payload = json.dumps({"id": "ok-after-retries"}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def _flaky_server():
    _FlakyUpstream.hits = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyUpstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_open_with_retries_survives_transient_429s():
    from compound.orproxy import open_with_retries

    server = _flaky_server()
    port = server.server_address[1]
    try:
        req = lambda: urlrequest.Request(  # noqa: E731
            f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}", method="POST"
        )
        with open_with_retries(req, attempts=4, base_delay=0, sleep=lambda s: None) as resp:
            got = json.loads(resp.read())
    finally:
        server.shutdown()
    assert got["id"] == "ok-after-retries"
    assert _FlakyUpstream.hits == 3  # two 429s absorbed, third try succeeded


def test_open_with_retries_does_not_retry_400():
    import pytest as _pytest

    from compound.orproxy import open_with_retries

    server = _flaky_server()
    port = server.server_address[1]
    try:
        req = lambda: urlrequest.Request(  # noqa: E731
            f"http://127.0.0.1:{port}/v1/always-400", data=b"{}", method="POST"
        )
        with _pytest.raises(urlerror.HTTPError) as exc_info:
            open_with_retries(req, attempts=4, base_delay=0, sleep=lambda s: None)
    finally:
        server.shutdown()
    assert exc_info.value.code == 400
    assert _FlakyUpstream.hits == 1  # capability errors surface immediately


def test_open_with_retries_gives_up_and_surfaces_last_429():
    import pytest as _pytest

    from compound.orproxy import open_with_retries

    server = _flaky_server()
    port = server.server_address[1]
    try:
        req = lambda: urlrequest.Request(  # noqa: E731
            f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}", method="POST"
        )
        with _pytest.raises(urlerror.HTTPError) as exc_info:
            open_with_retries(req, attempts=2, base_delay=0, sleep=lambda s: None)
    finally:
        server.shutdown()
    assert exc_info.value.code == 429  # exhausted retries surface unchanged
    assert _FlakyUpstream.hits == 2


def test_proxy_end_to_end_retries_through_flaky_upstream(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sekret")
    monkeypatch.setenv("ORPROXY_RETRY_BASE", "0")
    server = _flaky_server()
    up_port = server.server_address[1]
    spec = ProviderSpec(
        token="openrouter/deepinfra",
        kind="openrouter",
        base_url=f"http://127.0.0.1:{up_port}/v1",
        api_key_env="OPENROUTER_API_KEY",
        upstream="deepinfra",
    )
    try:
        with serve_provider(spec) as base:
            req = urlrequest.Request(
                base + "/chat/completions", data=b'{"model":"m"}',
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urlrequest.urlopen(req, timeout=10) as resp:
                got = json.loads(resp.read())
    finally:
        server.shutdown()
    assert got["id"] == "ok-after-retries"  # client never saw the 429s


class _StreamingUpstream(BaseHTTPRequestHandler):
    """Serves an SSE response whose usage lands only in the final chunk."""

    def log_message(self, *a):  # noqa: D401
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        chunks = [
            {"provider": "DeepInfra", "choices": [{"delta": {"content": "hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {
                "choices": [{"finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 5,
                    "cost": 0.00004,
                    "prompt_tokens_details": {"cached_tokens": 96},
                },
            },
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def test_proxy_records_a_call_ledger_row_without_altering_the_stream(monkeypatch, tmp_path):
    """The ledger is a tee: the client sees the identical bytes either way."""
    import compound.orproxy as orproxy

    ledger_path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sekret")
    monkeypatch.setenv("COMPOUND_CALL_LEDGER", str(ledger_path))
    monkeypatch.setenv("COMPOUND_RUN_LABEL", "auto-vs-pinned")
    orproxy._LEDGERS.clear()  # the cache is keyed by path and outlives one test

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    spec = ProviderSpec(
        token="openrouter/deepinfra/fp4",
        kind="openrouter",
        base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        api_key_env="OPENROUTER_API_KEY",
        upstream="deepinfra/fp4",
    )
    try:
        with serve_provider(spec) as base:
            req = urlrequest.Request(
                base + "/chat/completions",
                data=json.dumps(
                    {"model": "m", "stream": True, "messages": [{"role": "user", "content": "yo"}]}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
    finally:
        upstream.shutdown()

    # The stream reached the client intact.
    assert body.count("data:") == 4
    assert "[DONE]" in body

    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["route"] == "deepinfra/fp4"   # spec.label keeps the quant tag
    assert row["run_label"] == "auto-vs-pinned"
    assert row["status"] == 200
    assert row["provider_echo"] == "DeepInfra"
    assert row["pin_honored"] is True          # slug vs display name folded
    assert row["prompt_tokens"] == 120
    assert row["cached_tokens"] == 96          # only the final chunk carried it
    assert row["cost_usd"] == 0.00004
    assert row["stream"] is True
    assert row["latency_ms"] > 0


def test_proxy_writes_no_ledger_when_disabled(monkeypatch, tmp_path):
    import compound.orproxy as orproxy

    monkeypatch.delenv("COMPOUND_CALL_LEDGER", raising=False)
    orproxy._LEDGERS.clear()
    assert orproxy.get_ledger() is None


def test_inject_requests_usage_accounting_for_openrouter():
    """Without these flags OpenRouter returns no cost and no cached-token split."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = inject(body, ProviderSpec(
        token="openrouter/deepinfra", kind="openrouter",
        base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
        upstream="deepinfra",
    ))
    assert out["usage"] == {"include": True}
    assert "stream_options" not in out  # not a streaming request


def test_inject_adds_include_usage_only_on_streaming_requests():
    body = {"model": "m", "stream": True, "messages": []}
    out = inject(body, ProviderSpec(
        token="openrouter/auto", kind="openrouter",
        base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
        upstream=None,
    ))
    assert out["stream_options"] == {"include_usage": True}
    assert "provider" not in out  # auto stays unpinned


def test_inject_leaves_doubleword_usage_untouched():
    """Doubleword's API does not take OpenRouter's usage accounting block."""
    out = inject({"model": "m", "messages": []}, ProviderSpec(
        token="doubleword/flex", kind="doubleword",
        base_url="https://api.doubleword.ai/v1", api_key_env="DOUBLEWORD_API_KEY",
        service_tier="flex",
    ))
    assert "usage" not in out
