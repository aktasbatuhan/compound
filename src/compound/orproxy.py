"""A localhost OpenAI-compatible proxy that pins one serving host.

External agent harnesses (terminal-bench's terminus, and SWE-bench-style runners)
build their own model calls and give us no hook to add OpenRouter's
``provider.only`` routing block. So a run through them, with only an OpenRouter
key, cannot be pinned to a specific upstream: it lands on OpenRouter's default
pick and the "same model, many hosts" comparison collapses to one opaque row.

This proxy closes that gap. It speaks the OpenAI chat API on ``localhost``, and on
every request it merges in exactly the pinning a :class:`ProviderSpec` describes
(``provider.only`` for an OpenRouter upstream, ``service_tier`` for Doubleword
flex), then forwards to the real upstream with the right key. Point the harness
at ``OPENAI_API_BASE=http://127.0.0.1:<port>/v1`` and run it once per provider;
each run is now pinned, with just the credentials the project already holds.

The body-mutation is a pure function (:func:`inject`) so it is unit-tested
without a socket; the server is a thin stdlib wrapper around it.

CLI:
    python -m compound.orproxy --provider openrouter/deepinfra --port 8900
    # then: OPENAI_API_BASE=http://127.0.0.1:8900/v1 OPENAI_API_KEY=x <harness>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from compound.providers_registry import ProviderSpec, parse_provider


def inject(body: dict[str, Any], spec: ProviderSpec) -> dict[str, Any]:
    """Return ``body`` with this host's pinning merged in (non-destructive).

    OpenRouter ``provider`` and ``service_tier`` are set to the spec's values;
    an existing ``provider`` block from the caller is overridden so pinning is
    never silently defeated by a harness that sets its own routing.
    """
    merged = dict(body)
    for key, value in spec.proxy_injection().items():
        merged[key] = value
    return merged


def target_url(base_url: str, path: str) -> str:
    """Map an incoming request path onto the upstream base.

    The harness calls ``/v1/chat/completions``; every upstream base already ends
    in ``/v1``, so the leading ``/v1`` is stripped before joining to avoid a
    doubled ``/v1/v1``.
    """
    base = base_url.rstrip("/")
    if path.startswith("/v1"):
        path = path[len("/v1") :]
    return base + path


def _auth_header(spec: ProviderSpec) -> str:
    key = os.getenv(spec.required_key_env())
    if not key:
        raise RuntimeError(f"{spec.required_key_env()} is not set")
    return f"Bearer {key}"


class _Handler(BaseHTTPRequestHandler):
    spec: ProviderSpec  # set on the server subclass

    def log_message(self, *args: Any) -> None:  # noqa: D401 — silence default logging
        return

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        spec = self.spec
        url = target_url(spec.forward_base_url, self.path)

        data: bytes | None = None
        if raw:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                data = raw  # pass opaque bodies through untouched
            else:
                data = json.dumps(inject(body, spec)).encode()
        headers = {"Content-Type": "application/json", "Authorization": _auth_header(spec)}
        # OpenRouter asks for these attribution headers; harmless elsewhere.
        headers.setdefault("HTTP-Referer", "https://github.com/aktasbatuhan/compound")
        headers.setdefault("X-Title", "compound-bench")

        req = urlrequest.Request(url, data=data, headers=headers, method=method)
        try:
            with urlrequest.urlopen(req, timeout=600) as upstream:
                self.send_response(upstream.status)
                ctype = upstream.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", ctype)
                self.end_headers()
                while True:
                    chunk = upstream.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urlerror.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:  # noqa: BLE001 — surface upstream failure as 502
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(exc)}}).encode())

    def do_POST(self) -> None:  # noqa: N802 — stdlib naming
        self._forward("POST")

    def do_GET(self) -> None:  # noqa: N802 — /v1/models and health probes
        self._forward("GET")


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, spec: ProviderSpec, port: int) -> None:
        handler = type("_BoundHandler", (_Handler,), {"spec": spec})
        super().__init__(("127.0.0.1", port), handler)
        self.spec = spec


@contextmanager
def serve_provider(spec: ProviderSpec, port: int = 0):
    """Run the proxy for ``spec`` in a background thread; yield its base URL.

    ``port=0`` binds a free port (useful for tests). The base URL already ends in
    ``/v1`` so it drops straight into ``OPENAI_API_BASE``.
    """
    server = ProxyServer(spec, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bound_port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{bound_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", required=True, help="provider token, e.g. openrouter/deepinfra"
    )
    parser.add_argument("--port", type=int, default=8900)
    args = parser.parse_args()
    spec = parse_provider(args.provider)
    if not os.getenv(spec.required_key_env()):
        print(f"error: {spec.required_key_env()} is not set", file=sys.stderr)
        return 2
    server = ProxyServer(spec, args.port)
    print(
        f"orproxy: pinning {spec.label} -> {spec.forward_base_url}\n"
        f"  point your harness at OPENAI_API_BASE=http://127.0.0.1:{args.port}/v1",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
