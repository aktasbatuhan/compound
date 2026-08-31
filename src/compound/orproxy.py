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
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from compound.call_ledger import (
    MAX_CAPTURE_BYTES,
    CallLedger,
    build_record,
    ledger_path_from_env,
)
from compound.providers_registry import ProviderSpec, parse_provider

#: One ledger per path, shared by every request thread. Cached because the
#: writer owns a lock: two instances on one file would interleave lines.
_LEDGERS: dict[str, CallLedger] = {}
_LEDGERS_LOCK = threading.Lock()


def get_ledger() -> CallLedger | None:
    """The call ledger for this run, or ``None`` when recording is off."""
    path = ledger_path_from_env()
    if path is None:
        return None
    with _LEDGERS_LOCK:
        ledger = _LEDGERS.get(path)
        if ledger is None:
            ledger = CallLedger(path)
            _LEDGERS[path] = ledger
        return ledger

#: Transient upstream failures worth retrying. A shared-pool 429 aborted 25
#: terminal-bench episodes in one sweep because the agent harness treats any
#: HTTP error as fatal; retrying here, before the client sees anything, makes
#: rate-limit weather invisible to every harness behind the proxy. 4xx
#: capability/auth errors (400/401/404) are NOT retried — those must surface.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 524, 529})


def retry_delays(attempts: int, base: float, cap: float = 30.0) -> list[float]:
    """Exponential backoff schedule between ``attempts`` tries (len = attempts-1)."""
    return [min(base * (2**i), cap) for i in range(max(attempts - 1, 0))]


def _retry_after_seconds(headers: Any, fallback: float, cap: float = 60.0) -> float:
    """Honor an upstream ``Retry-After`` (seconds form) when present and sane."""
    try:
        value = float((headers.get("Retry-After") or "").strip())
    except (AttributeError, ValueError):
        return fallback
    return min(value, cap) if value > 0 else fallback


def open_with_retries(
    make_request,
    *,
    attempts: int | None = None,
    base_delay: float | None = None,
    sleep=time.sleep,
    log=lambda msg: print(msg, file=sys.stderr),
):
    """urlopen with backoff on transient failures; returns the open response.

    ``make_request()`` builds a fresh :class:`urllib.request.Request` per try (a
    Request must not be reused after a failed send). Non-retryable HTTP errors
    and exhausted retries re-raise for the caller to surface unchanged.
    Tunables (env): ``ORPROXY_RETRIES`` total attempts (default 6),
    ``ORPROXY_RETRY_BASE`` first backoff in seconds (default 2).
    """
    if attempts is None:
        attempts = int(os.getenv("ORPROXY_RETRIES", "6"))
    if base_delay is None:
        base_delay = float(os.getenv("ORPROXY_RETRY_BASE", "2"))
    delays = retry_delays(attempts, base_delay)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return urlrequest.urlopen(make_request(), timeout=600)
        except urlerror.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUSES or attempt == attempts - 1:
                raise
            exc.read()  # drain so the connection can be reused
            delay = _retry_after_seconds(exc.headers, delays[attempt])
            log(f"orproxy: upstream {exc.code}, retry {attempt + 1}/{attempts - 1} in {delay:.1f}s")
            last_exc = exc
        except urlerror.URLError as exc:
            if attempt == attempts - 1:
                raise
            delay = delays[attempt]
            log(f"orproxy: connection error ({exc.reason}), retry {attempt + 1}/{attempts - 1} in {delay:.1f}s")
            last_exc = exc
        sleep(delay)
    raise last_exc if last_exc else RuntimeError("unreachable")  # pragma: no cover


def inject(body: dict[str, Any], spec: ProviderSpec) -> dict[str, Any]:
    """Return ``body`` with this host's pinning merged in (non-destructive).

    OpenRouter ``provider`` and ``service_tier`` are set to the spec's values;
    an existing ``provider`` block from the caller is overridden so pinning is
    never silently defeated by a harness that sets its own routing.

    On cache opt-in, a host whose :attr:`ProviderSpec.cache_strategy` is
    ``"explicit_marker"`` (Doubleword by default) has the final message of every
    request marked with an Anthropic-style ``cache_control`` block. That host's
    prompt cache is explicit opt-in, so a stock OpenAI client re-bills the full
    growing transcript every agent turn; the OpenRouter majors (``"implicit"``)
    cache the same prefixes on their own and need no marker. Marking the last
    message caches the whole conversation prefix, which the next turn reads.

    The opt-in signal is :func:`cache_optin_enabled` (the ``--cache-optin`` CLI
    flag threads through the ``COMPOUND_DW_CACHE`` env var), so the injection is
    driven by the provider's declared strategy, never by a hardcoded host name.
    """
    merged = dict(body)
    for key, value in spec.proxy_injection().items():
        merged[key] = value
    if spec.cache_strategy == "explicit_marker" and cache_optin_enabled():
        merged["messages"] = _mark_cache_prefix(merged.get("messages"))
    if spec.kind == "openrouter":
        merged = _request_usage_accounting(merged)
    # The cache-hit-rate source for #43 is the call ledger, not results.json:
    # terminal-bench reports only total input/output tokens per trial, with no
    # cached-token split. Every call through this proxy now records its own
    # ``cached_tokens`` (see :mod:`compound.call_ledger`), so a run's cache-hit
    # rate is computed from the calls that carried the marker.
    return merged


def _request_usage_accounting(body: dict[str, Any]) -> dict[str, Any]:
    """Ask OpenRouter to return the usage block the ledger reads.

    Cost and the cached-token split are opt-in on OpenRouter: without
    ``usage: {include: true}`` the response carries neither, and a streamed
    response carries no usage at all unless the request also sets
    ``stream_options: {include_usage: true}``. An external harness sets neither,
    so a run behind this proxy would record null cost and null cache hits on
    every call, which is precisely the data the run exists to collect.

    Both flags are additive accounting requests: they change what the response
    reports, not what is served or billed. ``stream_options`` is only sent on a
    streaming request, since the API rejects it otherwise.
    """
    merged = dict(body)
    usage = merged.get("usage")
    merged["usage"] = {**usage, "include": True} if isinstance(usage, dict) else {"include": True}
    if merged.get("stream"):
        options = merged.get("stream_options")
        merged["stream_options"] = (
            {**options, "include_usage": True}
            if isinstance(options, dict)
            else {"include_usage": True}
        )
    return merged


def cache_optin_enabled() -> bool:
    """Whether explicit prompt-cache markers are turned on for this run.

    Enabled by ``COMPOUND_DW_CACHE`` (1/true/on), which the ``--cache-optin`` run
    flag also sets so the signal reaches the in-process proxy. Kept as an env var
    so a harness-level run can force it on without the CLI.
    """
    return os.getenv("COMPOUND_DW_CACHE", "").lower() in ("1", "true", "on")


def _mark_cache_prefix(messages: Any) -> Any:
    """Attach ``cache_control`` to the last content block of the last message."""
    if not isinstance(messages, list) or not messages:
        return messages
    marker = {"type": "ephemeral", "ttl": "5m"}
    msgs = list(messages)
    last = dict(msgs[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": marker}]
    elif isinstance(content, list) and content:
        blocks = list(content)
        final = dict(blocks[-1])
        final["cache_control"] = marker
        blocks[-1] = final
        last["content"] = blocks
    else:
        return messages
    msgs[-1] = last
    return msgs


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

        debug_path = os.getenv("ORPROXY_DEBUG_LOG")
        if debug_path and data:
            with open(debug_path, "a") as dbg:
                dbg.write(">>> REQUEST\n" + data.decode("utf-8", "replace")[:4000] + "\n")
        def make_request() -> urlrequest.Request:
            return urlrequest.Request(url, data=data, headers=headers, method=method)

        # Ledger state. The body is teed as it streams rather than buffered
        # first, so recording never delays a byte reaching the harness.
        ledger = get_ledger()
        sent_body: dict[str, Any] | None = None
        if ledger and data:
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                parsed = None  # an opaque body forwarded untouched
            sent_body = parsed if isinstance(parsed, dict) else None
        started = time.monotonic()
        captured = bytearray()
        truncated = False
        status: int | None = None
        ctype = ""
        error: str | None = None

        def capture(chunk: bytes) -> None:
            nonlocal truncated
            if len(captured) + len(chunk) <= MAX_CAPTURE_BYTES:
                captured.extend(chunk)
                return
            # Past the cap keep a rolling tail: a streamed response carries its
            # usage in the final chunks, so the end is the part worth holding.
            truncated = True
            captured.extend(chunk)
            del captured[: len(captured) - MAX_CAPTURE_BYTES]

        try:
            with open_with_retries(make_request) as upstream:
                status = upstream.status
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
                    if ledger:
                        capture(chunk)
        except urlerror.HTTPError as exc:
            payload = exc.read()
            status, error = exc.code, f"http_{exc.code}"
            if ledger:
                capture(payload)
            if debug_path:
                with open(debug_path, "a") as dbg:
                    dbg.write(f"<<< {exc.code} {payload.decode('utf-8', 'replace')[:2000]}\n")
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:  # noqa: BLE001 — surface upstream failure as 502
            status, error = 502, str(exc)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(exc)}}).encode())
        finally:
            # A failed call is a measurement too: a 429 that cost an episode is
            # exactly the reliability data the run exists to collect, so the row
            # is written on every path. Recording must never take the proxy
            # down, so a ledger fault degrades to a warning.
            if ledger is not None:
                try:
                    ledger.write(
                        build_record(
                            route=spec.label,
                            upstream=spec.upstream,
                            status=status,
                            latency_ms=(time.monotonic() - started) * 1000,
                            request_body=sent_body,
                            response_raw=bytes(captured),
                            content_type=ctype,
                            error=error,
                            truncated=truncated,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 — never fail a call over logging
                    print(f"orproxy: call ledger write failed: {exc}", file=sys.stderr)

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
