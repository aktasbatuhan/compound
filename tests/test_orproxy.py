from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlrequest

import pytest

from compound.orproxy import inject, serve_provider, target_url
from compound.providers_registry import ProviderSpec, parse_provider


def test_inject_adds_openrouter_provider_only():
    spec = parse_provider("openrouter/deepinfra")
    body = {"model": "deepseek/deepseek-v4-flash-0731", "messages": []}
    out = inject(body, spec)
    assert out["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False}
    assert out["messages"] == []  # untouched
    assert body == {"model": "deepseek/deepseek-v4-flash-0731", "messages": []}  # non-destructive


def test_inject_overrides_caller_supplied_routing():
    spec = parse_provider("openrouter/baseten/fp8")
    out = inject({"provider": {"only": ["someone-else"]}}, spec)
    assert out["provider"]["only"] == ["baseten/fp8"]


def test_inject_adds_service_tier_for_flex():
    out = inject({"model": "x"}, parse_provider("doubleword/flex"))
    assert out["service_tier"] == "flex"


def test_inject_noop_for_realtime():
    out = inject({"model": "x"}, parse_provider("doubleword/realtime"))
    assert "service_tier" not in out and "provider" not in out


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
    assert rec["body"]["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False}
    assert rec["body"]["messages"][0]["content"] == "yo"
