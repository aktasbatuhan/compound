"""Offline money-safety and wire-shape tests for the migration experiment I/O."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from compound.migration_io import (
    BudgetExceeded,
    HTTPCallError,
    HTTPResponse,
    InferenceService,
    Ledger,
    RequestIdentityError,
    SafetyStop,
    UnresolvedCall,
    extract_visible_output_text,
    repair_stored_output_text,
)


def route(**overrides):
    value = {
        "id": "or-test",
        "model": "test/model",
        "api": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "TEST_MIGRATION_KEY",
    }
    value.update(overrides)
    return value


def rates(**overrides):
    value = {"input_per_million": 1, "output_per_million": 2}
    value.update(overrides)
    return {"or-test": value, "dw-flex": value, "dw-realtime": value}


def chat_payload(**overrides):
    value = {
        "id": "chat-1",
        "provider": "raw-provider/variant",
        "choices": [{"message": {"content": "hello"}}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "cost": 0.0002,
        },
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("total_cap", "stage_cap"),
    [(0.001, 1), (1, 0.001)],
    ids=["total-cap", "stage-cap"],
)
def test_atomic_concurrent_reservations_enforce_total_and_stage_caps(
    tmp_path, total_cap, stage_cap
):
    ledger = Ledger(
        tmp_path / "ledger.sqlite", cap_usd=total_cap, stage_caps={"x": stage_cap}
    )
    barrier = threading.Barrier(8)
    accepted = []
    rejected = []

    def reserve(index):
        barrier.wait()
        try:
            ledger.reserve(
                call_id=f"call-{index}",
                fingerprint=f"fp-{index}",
                route_id="r",
                stage="x",
                api="responses",
                reserved_usd=Decimal("0.0006"),
                input_token_bound=1,
                output_token_bound=1,
                request_artifact=f"request-{index}",
            )
            accepted.append(index)
        except BudgetExceeded:
            rejected.append(index)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(accepted) == 1
    assert len(rejected) == 7
    assert ledger.used() == pytest.approx(0.0006)
    assert ledger.used("x") == pytest.approx(0.0006)


def test_private_request_and_response_artifacts_are_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    service = InferenceService(
        ledger,
        rates(),
        transport=lambda *args: HTTPResponse(200, json.dumps(chat_payload()).encode()),
    )
    service.call(
        route(), [{"role": "user", "content": "private"}], {"max_tokens": 10},
        call_id="artifacts", stage="x",
    )
    row = ledger.get("artifacts")
    assert row is not None
    assert (ledger.artifact_dir.stat().st_mode & 0o777) == 0o700
    assert (ledger.path.stat().st_mode & 0o777) == 0o600
    assert (ledger.artifact_dir / (row["request_artifact"].split("/")[-1])).exists()
    assert (ledger.artifact_dir / (row["response_artifact"].split("/")[-1])).exists()


def test_post_timeout_stays_reserved_and_is_never_blindly_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    post_count = 0

    def timeout_transport(method, url, headers, body, timeout):
        nonlocal post_count
        post_count += 1
        raise TimeoutError("socket timed out")

    service = InferenceService(ledger, rates(), transport=timeout_transport)
    with pytest.raises(UnresolvedCall, match="POST transport failure"):
        service.call(
            route(), [{"role": "user", "content": "hi"}], {"max_tokens": 10},
            call_id="timeout-1", stage="x",
        )
    reserved = ledger.used()
    assert reserved > 0
    assert ledger.get("timeout-1")["status"] == "unknown"

    with pytest.raises(UnresolvedCall, match="refusing another POST"):
        service.call(
            route(), [{"role": "user", "content": "hi"}], {"max_tokens": 10},
            call_id="timeout-1", stage="x",
        )
    assert post_count == 1
    assert ledger.used() == reserved


def test_absolute_deadline_stops_continuously_chunked_post_without_retry(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    requests = 0
    payload = json.dumps(chat_payload()).encode()
    pieces = [payload[index:index + 12] for index in range(0, len(payload), 12)]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            nonlocal requests
            requests += 1
            self.rfile.read(int(self.headers.get("content-length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for piece in pieces:
                    self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.04)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    service = InferenceService(ledger, rates(), timeout_s=0.10)
    local_route = route(base_url=f"http://127.0.0.1:{server.server_port}")
    started = time.monotonic()
    try:
        with pytest.raises(UnresolvedCall, match="deadline_exceeded"):
            service.call(
                local_route,
                [{"role": "user", "content": "hi"}],
                {"max_tokens": 10},
                call_id="chunked-deadline",
                stage="x",
                poll_deadline_s=0.22,
            )
        elapsed = time.monotonic() - started
        row = ledger.get("chunked-deadline")
        assert row["status"] == "deadline_exceeded"
        assert row["accounted_nano"] == row["reserved_nano"]
        assert row["result_json"] is None
        artifact = ledger.artifact_dir / Path(row["response_artifact"]).name
        saved = json.loads(artifact.read_text())
        assert saved["status"] == 200
        assert saved["incomplete"] is True

        with pytest.raises(UnresolvedCall, match="refusing another POST"):
            service.call(
                local_route,
                [{"role": "user", "content": "hi"}],
                {"max_tokens": 10},
                call_id="chunked-deadline",
                stage="x",
                poll_deadline_s=0.22,
            )
        assert requests == 1
        assert elapsed < 0.45
        assert any(event["kind"] == "deadline_exceeded" for event in ledger.events())
    finally:
        server.shutdown()
        server.server_close()


def test_absolute_deadline_does_not_wait_for_silent_partial_body_close(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    requests = 0

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            nonlocal requests
            requests += 1
            self.rfile.read(int(self.headers.get("content-length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b'{"partial":')
            self.wfile.flush()
            time.sleep(0.60)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    service = InferenceService(ledger, rates(), timeout_s=0.50)
    local_route = route(base_url=f"http://127.0.0.1:{server.server_port}")
    started = time.monotonic()
    try:
        with pytest.raises(UnresolvedCall, match="deadline_exceeded"):
            service.call(
                local_route,
                [{"role": "user", "content": "hi"}],
                {"max_tokens": 10},
                call_id="silent-partial-deadline",
                stage="x",
                poll_deadline_s=0.12,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 0.30
        assert ledger.get("silent-partial-deadline")["status"] == "deadline_exceeded"

        with pytest.raises(UnresolvedCall, match="refusing another POST"):
            service.call(
                local_route,
                [{"role": "user", "content": "hi"}],
                {"max_tokens": 10},
                call_id="silent-partial-deadline",
                stage="x",
                poll_deadline_s=0.12,
            )
        assert requests == 1
    finally:
        server.shutdown()
        server.server_close()


def test_background_post_and_poll_share_one_absolute_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    now = [0.0]
    get_timeouts = []
    polls = 0

    def clock():
        return now[0]

    def sleep(seconds):
        now[0] += seconds

    def transport(method, url, headers, body, timeout):
        nonlocal polls
        if method == "POST":
            now[0] += 4
            return HTTPResponse(200, b'{"id":"shared-budget","status":"in_progress"}')
        polls += 1
        get_timeouts.append(timeout)
        if polls == 1:
            now[0] += 3
            return HTTPResponse(200, b'{"id":"shared-budget","status":"in_progress"}')
        now[0] += 2
        return HTTPResponse(
            200,
            b'{"id":"shared-budget","status":"completed",'
            b'"service_tier":"flex","output_text":"done",'
            b'"usage":{"input_tokens":10,"output_tokens":2}}',
        )

    dw_route = route(
        id="dw-flex",
        api="responses",
        base_url="https://api.doubleword.ai/v1",
        service_tier="flex",
        background=True,
    )
    service = InferenceService(
        ledger,
        rates(),
        transport=transport,
        clock=clock,
        sleep=sleep,
        poll_interval_s=2,
    )

    with pytest.raises(UnresolvedCall, match="deadline_exceeded"):
        service.call(
            dw_route,
            [{"role": "user", "content": "hi"}],
            {"max_tokens": 10},
            call_id="shared-total-budget",
            stage="x",
            poll_deadline_s=10,
        )

    row = ledger.get("shared-total-budget")
    assert get_timeouts == pytest.approx([6, 1], abs=0.01)
    assert row["status"] == "deadline_exceeded"
    assert row["response_id"] == "shared-budget"
    assert row["cost_usd"] is not None
    assert row["cost_kind"] == "derived"
    assert json.loads(row["result_json"])["output_text"] == "done"


def test_completed_call_cache_requires_identical_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    exchanges = []

    def transport(method, url, headers, body, timeout):
        exchanges.append((method, body))
        return HTTPResponse(200, json.dumps(chat_payload()).encode())

    service = InferenceService(ledger, rates(), transport=transport)
    first = service.call(
        route(), [{"role": "user", "content": "same"}], {"max_tokens": 20},
        call_id="stable-attempt", stage="x",
    )
    cached = service.call(
        route(), [{"role": "user", "content": "same"}], {"max_tokens": 20},
        call_id="stable-attempt", stage="x",
    )
    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert cached.latency_s == first.latency_s
    assert len(exchanges) == 1

    with pytest.raises(RequestIdentityError):
        service.call(
            route(), [{"role": "user", "content": "changed"}], {"max_tokens": 20},
            call_id="stable-attempt", stage="x",
        )

    changed_rates = InferenceService(
        ledger,
        rates(output_per_million=3),
        transport=transport,
    )
    with pytest.raises(RequestIdentityError):
        changed_rates.call(
            route(), [{"role": "user", "content": "same"}], {"max_tokens": 20},
            call_id="stable-attempt", stage="x",
        )
    assert len(exchanges) == 1


@pytest.mark.parametrize("bad_rate", ["NaN", "Infinity", "not-a-number"])
def test_nonfinite_or_malformed_rates_fail_before_reservation(tmp_path, bad_rate):
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    with pytest.raises(ValueError):
        InferenceService(
            ledger,
            rates(input_per_million=bad_rate),
            transport=lambda *args: pytest.fail("transport must not run"),
        )
    assert ledger.used() == 0


def test_openrouter_reported_cost_does_not_double_bill_reasoning(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    payload = chat_payload(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 80,
            "completion_tokens_details": {"reasoning_tokens": 70},
            "cost": 0.0123,
        }
    )
    service = InferenceService(
        ledger,
        rates(input_per_million=10, output_per_million=100),
        transport=lambda *args: HTTPResponse(200, json.dumps(payload).encode()),
    )
    result = service.call(
        route(), [{"role": "user", "content": "reason"}], {"max_tokens": 100},
        call_id="reasoning", stage="x",
    )
    assert result.cost_usd == pytest.approx(0.0123)
    assert result.cost_kind == "reported"
    assert result.usage["completion_tokens"] == 80
    assert ledger.used() == pytest.approx(0.0123)


@pytest.mark.parametrize(
    ("usage", "message"),
    [
        (
            {"prompt_tokens": 50_000, "completion_tokens": 1, "cost": 0.0001},
            "input tokens",
        ),
        (
            {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.9},
            "exceeds reservation",
        ),
    ],
)
def test_reported_token_or_cost_bound_violation_stops_future_spend(
    tmp_path, monkeypatch, usage, message
):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    payload = chat_payload(usage=usage)
    service = InferenceService(
        ledger,
        rates(),
        transport=lambda *args: HTTPResponse(200, json.dumps(payload).encode()),
    )
    with pytest.raises(SafetyStop, match=message):
        service.call(
            route(), [{"role": "user", "content": "bound"}], {"max_tokens": 10},
            call_id=f"bound-{message}", stage="x",
        )
    row = ledger.get(f"bound-{message}")
    assert row["status"] == "safety_stop"
    assert ledger.events(f"bound-{message}")
    with pytest.raises(SafetyStop, match=message):
        service.call(
            route(), [{"role": "user", "content": "bound"}], {"max_tokens": 10},
            call_id=f"bound-{message}", stage="x",
        )


def test_terminal_failed_200_is_charged_but_never_cached_or_resubmitted(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    posts = 0
    payload = chat_payload(status="failed")

    def transport(*args):
        nonlocal posts
        posts += 1
        return HTTPResponse(200, json.dumps(payload).encode())

    service = InferenceService(ledger, rates(), transport=transport)
    with pytest.raises(HTTPCallError, match="ended with status 'failed'"):
        service.call(
            route(), [{"role": "user", "content": "fail"}], {"max_tokens": 10},
            call_id="failed-200", stage="x",
        )
    row = ledger.get("failed-200")
    assert row["status"] == "failed"
    assert ledger.used() == pytest.approx(0.0002)
    with pytest.raises(UnresolvedCall, match="refusing another POST"):
        service.call(
            route(), [{"role": "user", "content": "fail"}], {"max_tokens": 10},
            call_id="failed-200", stage="x",
        )
    assert posts == 1


def test_success_200_without_usage_stays_unknown_and_reserved(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    payload = {"id": "missing-usage", "status": "failed", "error": {"message": "bad"}}
    service = InferenceService(
        ledger,
        rates(),
        transport=lambda *args: HTTPResponse(200, json.dumps(payload).encode()),
    )
    with pytest.raises(UnresolvedCall, match="omitted token usage"):
        service.call(
            route(), [{"role": "user", "content": "fail"}], {"max_tokens": 10},
            call_id="failed-no-usage", stage="x",
        )
    assert ledger.get("failed-no-usage")["status"] == "unknown"
    assert ledger.used() > 0


def test_doubleword_chat_cache_write_and_hit_are_billed_as_prompt_subsets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    payload = chat_payload(
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 600},
            "cache_creation_input_tokens": 200,
        }
    )
    capture = {}

    def transport(method, url, headers, body, timeout):
        capture["body"] = json.loads(body)
        return HTTPResponse(200, json.dumps(payload).encode())

    dw_route = route(
        id="dw-realtime",
        base_url="https://api.doubleword.ai/v1",
    )
    service = InferenceService(
        ledger,
        rates(
            input_per_million=0.15,
            output_per_million=0.6,
            cached_input_per_million=0.003,
            cache_write_per_million=0.1875,
        ),
        transport=transport,
    )
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
            ],
        }
    ]
    result = service.call(
        dw_route, messages, {"max_tokens": 100}, call_id="dw-cache", stage="x"
    )
    expected = (200 * 0.15 + 600 * 0.003 + 200 * 0.1875 + 100 * 0.6) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    assert capture["body"]["messages"] == messages


def test_doubleword_responses_forwards_params_and_stops_on_wrong_tier(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    captured = {}
    payload = {
        "id": "resp-1",
        "status": "completed",
        "service_tier": "realtime",
        "provider": {"name": "doubleword", "deployment": "raw-deploy-id"},
        "output_text": '{"ok":true}',
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }

    def transport(method, url, headers, body, timeout):
        captured["method"] = method
        captured["body"] = json.loads(body)
        return HTTPResponse(200, json.dumps(payload).encode())

    dw_route = route(
        id="dw-flex",
        api="responses",
        base_url="https://api.doubleword.ai/v1",
        service_tier="flex",
        background=True,
        provider={"only": ["doubleword"], "allow_fallbacks": False, "require_parameters": True},
    )
    service = InferenceService(ledger, rates(), transport=transport)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
    }
    with pytest.raises(SafetyStop, match="does not match requested"):
        service.call(
            dw_route,
            [{"role": "user", "content": "hi"}],
            {
                "max_tokens": 321,
                "reasoning_effort": "medium",
                "response_format": response_format,
                "temperature": 0.2,
            },
            call_id="wrong-tier",
            stage="x",
        )

    body = captured["body"]
    assert captured["method"] == "POST"
    assert body["max_output_tokens"] == 321
    assert body["reasoning"] == {"effort": "medium"}
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "schema": {"type": "object"},
        "strict": True,
    }
    assert body["temperature"] == 0.2
    assert body["provider"] == dw_route["provider"]
    assert ledger.get("wrong-tier")["status"] == "safety_stop"


def test_background_response_id_is_resumed_without_second_post(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    posts = 0
    polls = 0

    def first_transport(method, url, headers, body, timeout):
        nonlocal posts, polls
        if method == "POST":
            posts += 1
            return HTTPResponse(200, b'{"id":"resp-resume","status":"in_progress"}')
        polls += 1
        raise TimeoutError("temporary poll failure")

    dw_route = route(
        id="dw-flex",
        api="responses",
        base_url="https://api.doubleword.ai/v1",
        service_tier="flex",
        background=True,
    )
    first = InferenceService(ledger, rates(), transport=first_transport)
    with pytest.raises(UnresolvedCall, match="poll transport failure"):
        first.call(
            dw_route, [{"role": "user", "content": "hi"}], {"max_tokens": 10},
            call_id="resume", stage="x",
        )
    assert ledger.get("resume")["response_id"] == "resp-resume"
    with sqlite3.connect(ledger.path) as conn:
        conn.execute("UPDATE calls SET created_at = ? WHERE call_id = ?", (time.time() - 400, "resume"))

    def resumed_transport(method, url, headers, body, timeout):
        nonlocal posts, polls
        assert method == "GET"
        assert url.endswith("/responses/resp-resume")
        polls += 1
        return HTTPResponse(
            200,
            json.dumps(
                {
                    "id": "resp-resume",
                    "status": "completed",
                    "service_tier": "flex",
                    "output_text": "done",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ).encode(),
        )

    resumed = InferenceService(ledger, rates(), transport=resumed_transport)
    with pytest.raises(UnresolvedCall, match="deadline_exceeded"):
        resumed.call(
            dw_route, [{"role": "user", "content": "hi"}], {"max_tokens": 10},
            call_id="resume", stage="x",
        )
    row = ledger.get("resume")
    assert row["status"] == "deadline_exceeded"
    assert row["response_id"] == "resp-resume"
    assert row["latency_s"] >= 399
    assert posts == 1
    assert polls == 1
    with pytest.raises(UnresolvedCall, match="refusing another POST"):
        resumed.call(
            dw_route, [{"role": "user", "content": "hi"}], {"max_tokens": 10},
            call_id="resume", stage="x",
        )
    assert posts == 1
    assert polls == 1


def test_missing_service_tier_echo_is_unavailable_not_false(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    payload = {
        "id": "resp-no-tier",
        "status": "completed",
        "output_text": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    service = InferenceService(
        ledger, rates(), transport=lambda *args: HTTPResponse(200, json.dumps(payload).encode())
    )
    result = service.call(
        route(
            id="dw-flex",
            api="responses",
            base_url="https://api.doubleword.ai/v1",
            service_tier="flex",
            background=True,
        ),
        [{"role": "user", "content": "hi"}],
        {"max_tokens": 5},
        call_id="no-tier", stage="x",
    )
    assert result.service_tier_verified is None
    assert result.echoed_service_tier is None


def test_responses_extraction_excludes_reasoning_text_and_keeps_it_raw(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    final_json = '{"scores":{"craft":5},"recommendation":"HIRE","rationale":"Grounded."}'
    payload = {
        "id": "resp-reasoning",
        "status": "completed",
        "service_tier": "flex",
        "output": [
            {
                "id": "rs_123",
                "type": "reasoning",
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "We need analyze evidence before returning the JSON.",
                    }
                ],
            },
            {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": final_json, "annotations": []}
                ],
            },
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    service = InferenceService(
        ledger,
        rates(),
        transport=lambda *args: HTTPResponse(200, json.dumps(payload).encode()),
    )
    result = service.call(
        route(
            id="dw-flex",
            api="responses",
            base_url="https://api.doubleword.ai/v1",
            service_tier="flex",
            background=True,
        ),
        [{"role": "user", "content": "grade"}],
        {"max_tokens": 100},
        call_id="visible-only", stage="x",
    )
    assert result.output_text == final_json
    assert "analyze evidence" not in result.output_text
    assert result.raw["output"][0]["content"][0]["type"] == "reasoning_text"
    assert extract_visible_output_text(payload) == final_json


def test_offline_repair_rebuilds_only_stored_normalized_output(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    visible = '{"ok":true}'
    payload = {
        "id": "resp-repair",
        "status": "completed",
        "service_tier": "flex",
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "private analysis"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": visible}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    service = InferenceService(
        ledger,
        rates(),
        transport=lambda *args: HTTPResponse(200, json.dumps(payload).encode()),
    )
    original = service.call(
        route(
            id="dw-flex",
            api="responses",
            base_url="https://api.doubleword.ai/v1",
            service_tier="flex",
            background=True,
        ),
        [{"role": "user", "content": "repair"}],
        {"max_tokens": 10},
        call_id="repair", stage="x",
    )
    row_before = ledger.get("repair")
    contaminated = original.as_dict()
    contaminated["output_text"] = "private analysis" + visible
    ledger.update("repair", result_json=json.dumps(contaminated, sort_keys=True))

    repaired = repair_stored_output_text(ledger, "repair")
    row_after = ledger.get("repair")
    assert repaired.output_text == visible
    assert repaired.raw == payload
    assert row_after["status"] == row_before["status"] == "completed"
    assert row_after["accounted_nano"] == row_before["accounted_nano"]
    assert row_after["response_artifact"] == row_before["response_artifact"]
    assert json.loads(row_after["result_json"])["output_text"] == visible
    assert ledger.events("repair")[-1]["kind"] == "output_text_repaired"


def test_output_token_bound_raises_ceiling_without_changing_the_request(tmp_path, monkeypatch):
    """A reasoning judge can bill beyond the max_tokens it honours for completion.

    The declared ceiling must cover that without altering the request body, because
    the body determines call identity and the measurement instrument.
    """
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    bodies = []

    def transport(method, url, headers, body, timeout):
        bodies.append(json.loads(body))
        return HTTPResponse(200, json.dumps(chat_payload()).encode())

    service = InferenceService(ledger, rates(), transport=transport)
    service.call(
        route(), [{"role": "user", "content": "x"}], {"max_tokens": 20},
        call_id="plain", stage="x",
    )
    service.call(
        route(), [{"role": "user", "content": "x"}], {"max_tokens": 20},
        call_id="raised", stage="x", output_token_bound=200,
    )
    # The wire request is byte-identical; only the reservation ceiling moved.
    assert bodies[0] == bodies[1]
    assert bodies[1]["max_tokens"] == 20
    plain = ledger.get("plain")
    raised = ledger.get("raised")
    assert plain["output_token_bound"] == 20
    assert raised["output_token_bound"] == 200
    assert int(raised["reserved_nano"]) > int(plain["reserved_nano"])


def test_output_token_bound_must_not_lower_the_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})

    def transport(method, url, headers, body, timeout):
        return HTTPResponse(200, json.dumps(chat_payload()).encode())

    service = InferenceService(ledger, rates(), transport=transport)
    with pytest.raises(ValueError, match="cannot be below"):
        service.call(
            route(), [{"role": "user", "content": "x"}], {"max_tokens": 4096},
            call_id="too-low", stage="x", output_token_bound=1024,
        )


def test_responses_prompt_cache_key_is_declared_per_route(tmp_path, monkeypatch):
    """The Responses API rejects a cache_control content block; it takes a key instead."""
    monkeypatch.setenv("TEST_MIGRATION_KEY", "secret")
    ledger = Ledger(tmp_path / "ledger.sqlite", cap_usd=1, stage_caps={"x": 1})
    bodies = []

    def transport(method, url, headers, body, timeout):
        bodies.append(json.loads(body))
        return HTTPResponse(200, json.dumps(chat_payload()).encode())

    service = InferenceService(ledger, rates(), transport=transport)
    chat = route()
    service.call(chat, [{"role": "user", "content": "x"}], {"max_tokens": 20},
                 call_id="chat", stage="x")
    # A chat-completions route never gains the key.
    assert "prompt_cache_key" not in bodies[0]

    with pytest.raises(ValueError, match="prompt_cache_key"):
        service.call({**chat, "api": "responses", "prompt_cache_key": "  "},
                     [{"role": "user", "content": "x"}], {"max_tokens": 20},
                     call_id="blank", stage="x")
