"""The SDK lives in sdk/python as its own package; these tests run it in place."""

import asyncio
import json
import queue
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk/python/src"))

from compound_sdk import Recorder  # noqa: E402
from compound_sdk import shape as sdk_shape  # noqa: E402
from compound_sdk.litellm import record_from_litellm  # noqa: E402
from compound_sdk.recorder import load_or_create_salt  # noqa: E402

from compound.shape import load_shapes  # noqa: E402
from compound.shape import loaders as main_loaders  # noqa: E402
from compound.shape import records as main_records  # noqa: E402

SALT = b"k" * 32
MESSAGES = [
    {"role": "system", "content": "Müşteri faturalarını kontrol eden bir asistansın."},
    {"role": "user", "content": "Kartımdan iki kez çekim yapılmış, 4111 1111 1111 1111."},
]
TOOLS = [{"type": "function", "function": {"name": "lookup_invoice"}}]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _recorder(tmp_path: Path) -> Recorder:
    return Recorder(tmp_path / "shapes.jsonl", task_key="billing", salt=SALT)


def test_sdk_hashing_matches_the_main_package_exactly():
    assert sdk_shape.prefix_chain(MESSAGES, TOOLS, SALT) == main_records.prefix_chain(
        MESSAGES, TOOLS, SALT
    )
    for text in ("Faturamda hata var", "My invoice is wrong", "", "مرحبا"):
        assert sdk_shape.lang_hint(text) == main_records.lang_hint(text)
    usages = [
        {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 512},
         "completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 12}},
        {"input_tokens": 100, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0,
         "output_tokens": 20},
        {"input_tokens": 500, "input_tokens_details": {"cached_tokens": 100}, "output_tokens": 9},
    ]
    for usage in usages:
        assert sdk_shape.usage_counts(usage) == main_loaders._openai_usage(usage)


def test_records_load_as_shapes_and_hold_no_text(tmp_path):
    rec = _recorder(tmp_path)
    rec.record(messages=MESSAGES, tools=TOOLS, model="m", started_at=1.0, ended_at=2.5,
               usage={"prompt_tokens": 50, "completion_tokens": 5})
    rec.flush()
    raw = (tmp_path / "shapes.jsonl").read_text()
    assert "4111" not in raw and "fatura" not in raw
    (shape,) = load_shapes([tmp_path / "shapes.jsonl"], b"other").shapes
    assert shape.latency_ms == pytest.approx(1500)
    assert shape.task_key == "billing" and shape.lang_hint == "tr" and shape.source == "sdk"
    assert shape.prefix_chain == main_records.prefix_chain(MESSAGES, TOOLS, SALT)


def test_salt_is_private_and_stable(tmp_path):
    path = tmp_path / "home" / "salt"
    first = load_or_create_salt(path)
    assert load_or_create_salt(path) == first and len(first) == 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_recording_never_raises(tmp_path, monkeypatch):
    rec = _recorder(tmp_path)
    rec.record(messages=object(), model="m", started_at=1.0)  # type: ignore[arg-type]
    assert rec.errors == 1

    def full(_item):
        raise queue.Full

    monkeypatch.setattr(rec._queue, "put_nowait", full)
    rec.record(messages=MESSAGES, model="m", started_at=1.0)
    assert rec.dropped == 1


class _FakeCompletions:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.seen = result, error, None

    def create(self, **kwargs):
        self.seen = kwargs
        if self.error:
            raise self.error
        return self.result


def _client(completions):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        base_url=SimpleNamespace(host="api.example.com"),
    )


def _response(**usage):
    return SimpleNamespace(
        model="gpt-x-2026",
        usage=SimpleNamespace(model_dump=lambda: usage),
        choices=[SimpleNamespace(finish_reason="stop")],
    )


def test_wrapped_client_returns_the_response_untouched_and_records(tmp_path):
    rec = _recorder(tmp_path)
    response = _response(prompt_tokens=30, completion_tokens=7)
    client = rec.wrap_openai(_client(_FakeCompletions(result=response)))
    assert client.chat.completions.create(model="gpt-x", messages=MESSAGES) is response
    rec.wrap_openai(client)  # wrapping twice is a no-op
    rec.flush()
    (row,) = _read(tmp_path / "shapes.jsonl")
    assert row["model"] == "gpt-x-2026" and row["provider"] == "api.example.com"
    assert (row["input_tokens"], row["output_tokens"]) == (30, 7)
    assert row["finish_reason"] == "stop"


def test_a_failed_call_is_recorded_and_the_error_still_reaches_the_app(tmp_path):
    rec = _recorder(tmp_path)
    client = rec.wrap_openai(_client(_FakeCompletions(error=TimeoutError("slow"))))
    with pytest.raises(TimeoutError):
        client.chat.completions.create(model="gpt-x", messages=MESSAGES)
    rec.flush()
    (row,) = _read(tmp_path / "shapes.jsonl")
    assert row["status"] == "error"


def _chunks():
    yield SimpleNamespace(model="gpt-x", usage=None, choices=[SimpleNamespace(finish_reason=None)])
    yield SimpleNamespace(model="gpt-x", usage=None, choices=[SimpleNamespace(finish_reason="stop")])
    yield SimpleNamespace(model="gpt-x", usage={"prompt_tokens": 12, "completion_tokens": 3},
                          choices=[])


def test_streams_pass_through_and_record_usage_from_the_last_chunk(tmp_path):
    rec = _recorder(tmp_path)
    client = rec.wrap_openai(_client(_FakeCompletions(result=_chunks())))
    stream = client.chat.completions.create(model="gpt-x", messages=MESSAGES, stream=True)
    assert len(list(stream)) == 3
    rec.flush()
    (row,) = _read(tmp_path / "shapes.jsonl")
    assert (row["input_tokens"], row["output_tokens"], row["finish_reason"]) == (12, 3, "stop")


def test_async_clients_are_wrapped_too(tmp_path):
    rec = _recorder(tmp_path)

    class AsyncCompletions:
        async def create(self, **kwargs):
            if kwargs.get("stream"):
                async def gen():
                    for chunk in _chunks():
                        yield chunk
                return gen()
            return _response(prompt_tokens=8, completion_tokens=2)

    client = rec.wrap_openai(_client(AsyncCompletions()))

    async def run():
        await client.chat.completions.create(model="gpt-x", messages=MESSAGES)
        stream = await client.chat.completions.create(model="gpt-x", messages=MESSAGES,
                                                      stream=True)
        return [c async for c in stream]

    assert len(asyncio.run(run())) == 3
    rec.flush()
    rows = _read(tmp_path / "shapes.jsonl")
    assert [r["input_tokens"] for r in rows] == [8, 12]


def test_the_real_openai_client_is_wrapped_end_to_end(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from openai import OpenAI

    body = json.dumps({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "local-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "tamam"}}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 2, "total_tokens": 23,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rec = _recorder(tmp_path)
        client = rec.wrap_openai(
            OpenAI(api_key="test", base_url=f"http://127.0.0.1:{server.server_port}/v1")
        )
        reply = client.chat.completions.create(model="local-model", messages=MESSAGES)
        assert reply.choices[0].message.content == "tamam"
        rec.flush()
    finally:
        server.shutdown()
    (row,) = _read(tmp_path / "shapes.jsonl")
    assert (row["input_tokens"], row["cached_input_tokens"], row["output_tokens"]) == (21, 0, 2)
    assert row["provider"] == "127.0.0.1"


def test_litellm_events_map_onto_shapes(tmp_path):
    rec = _recorder(tmp_path)
    start = datetime(2026, 10, 31, 9, 0, 0, tzinfo=UTC)
    end = datetime(2026, 10, 31, 9, 0, 2, tzinfo=UTC)
    kwargs = {
        "model": "deepseek/deepseek-v4.1-flash",
        "messages": MESSAGES,
        "optional_params": {"tools": TOOLS},
        "litellm_params": {"custom_llm_provider": "openrouter",
                           "metadata": {"task_key": "faturalar"}},
        "litellm_call_id": "call-1",
    }
    response = {"model": "deepseek-v4.1-flash", "usage": {"prompt_tokens": 70,
                "completion_tokens": 9}, "choices": [{"finish_reason": "stop"}]}
    record_from_litellm(rec, kwargs, response, start, end)
    rec.flush()
    (row,) = _read(tmp_path / "shapes.jsonl")
    assert row["task_key"] == "faturalar" and row["trace_id"] == "call-1"
    assert row["provider"] == "openrouter" and row["tool_count"] == 1
    assert row["latency_ms"] == pytest.approx(2000)
