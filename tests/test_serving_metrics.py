from __future__ import annotations

import json

import pytest

from compound import serving_metrics as sm
from compound.bench import main as bench_main  # noqa: F401 — ensures the module imports
from compound.providers_registry import parse_provider

# A tiny shape with a json_schema response_format, mirroring the real agent payload.
SHAPE = {
    "messages": [
        {"role": "user", "content": "solve this"},
        {"role": "assistant", "content": "ok"},
    ],
    "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
}
SHAPE_NO_RF = {"messages": [{"role": "user", "content": "hi"}]}
PIN = {"only": ["deepinfra"], "allow_fallbacks": False, "require_parameters": True}


# --- body construction per route kind and mode ------------------------------


def test_build_body_openrouter_reasoning_on_pins_provider_and_reasoning():
    spec = parse_provider("openrouter/deepinfra")
    body = sm.build_body(spec, "deepseek/deepseek-v4-flash-0731", sm.REASONING_ON, SHAPE)
    assert body["model"] == "deepseek/deepseek-v4-flash-0731"
    assert body["reasoning"] == {"enabled": True}
    assert body["usage"] == {"include": True}
    assert body["provider"] == PIN  # require_parameters pinning preserved
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == sm.DEFAULT_MAX_TOKENS
    assert body["response_format"] == SHAPE["response_format"]
    assert "reasoning_effort" not in body


def test_build_body_openrouter_reasoning_off():
    spec = parse_provider("openrouter/deepinfra")
    body = sm.build_body(spec, "m", sm.REASONING_OFF, SHAPE_NO_RF)
    assert body["reasoning"] == {"enabled": False}
    assert "response_format" not in body  # shape without a response_format


def test_build_body_openrouter_auto_is_unpinned():
    spec = parse_provider("openrouter/auto")
    body = sm.build_body(spec, "m", sm.REASONING_ON, SHAPE_NO_RF)
    assert "provider" not in body  # auto carries no provider.only block
    assert body["reasoning"] == {"enabled": True}


def test_build_body_openrouter_quant_tag_pins_base_slug():
    spec = parse_provider("openrouter/baseten/fp8")
    body = sm.build_body(spec, "m", sm.REASONING_ON, SHAPE_NO_RF)
    assert body["provider"]["only"] == ["baseten"]  # base slug, not the fp8 tag
    assert body["provider"]["require_parameters"] is True


def test_build_body_doubleword_flex_reasoning_on():
    spec = parse_provider("doubleword/flex")
    body = sm.build_body(spec, "deepseek-ai/DeepSeek-V4-Flash-0731", sm.REASONING_ON, SHAPE)
    assert body["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert body["reasoning_effort"] == "medium"
    assert body["service_tier"] == "flex"
    assert "reasoning" not in body  # doubleword rejects the reasoning block
    assert "provider" not in body
    assert body["response_format"] == SHAPE["response_format"]


def test_build_body_doubleword_realtime_reasoning_off_has_no_tier():
    spec = parse_provider("doubleword/realtime")
    body = sm.build_body(spec, "m", sm.REASONING_OFF, SHAPE_NO_RF)
    assert body["reasoning_effort"] == "none"
    assert "service_tier" not in body


# --- nonce cache-busting -----------------------------------------------------


def test_prepend_nonce_string_content_does_not_mutate_original():
    messages = [{"role": "user", "content": "original"}]
    out = sm.prepend_nonce(messages, "NONCE ")
    assert out[0]["content"] == "NONCE original"
    assert messages[0]["content"] == "original"  # untouched


def test_prepend_nonce_block_content():
    messages = [{"role": "user", "content": [{"type": "text", "text": "a"}]}]
    out = sm.prepend_nonce(messages, "NONCE")
    assert out[0]["content"][0] == {"type": "text", "text": "NONCE"}
    assert out[0]["content"][1] == {"type": "text", "text": "a"}


def test_make_nonce_is_unique_and_each_body_gets_a_fresh_one():
    spec = parse_provider("openrouter/deepinfra")
    b1 = sm.build_body(spec, "m", sm.REASONING_ON, SHAPE_NO_RF)
    b2 = sm.build_body(spec, "m", sm.REASONING_ON, SHAPE_NO_RF)
    c1 = b1["messages"][0]["content"]
    c2 = b2["messages"][0]["content"]
    assert c1.startswith("[bench-nonce ") and c2.startswith("[bench-nonce ")
    assert c1 != c2  # cache-busting: no two calls share a prefix
    assert c1.endswith("hi") and c2.endswith("hi")


def test_build_body_accepts_an_explicit_nonce(monkeypatch):
    monkeypatch.delenv("COMPOUND_DW_CACHE", raising=False)
    # OpenRouter caches implicitly, so its body carries the nonce and nothing else.
    spec = parse_provider("openrouter/novita")
    body = sm.build_body(spec, "m", sm.REASONING_OFF, SHAPE_NO_RF, nonce="FIXED\n")
    assert body["messages"][0]["content"] == "FIXED\nhi"

    # Doubleword's cache is opt-in, so the same nonce arrives inside the marked
    # content block the proxy would have added.
    dw = sm.build_body(
        parse_provider("doubleword/realtime"), "m", sm.REASONING_OFF, SHAPE_NO_RF, nonce="FIXED\n"
    )
    assert dw["messages"][0]["content"][-1]["text"] == "FIXED\nhi"


# --- model resolution --------------------------------------------------------


def test_model_for_picks_slug_per_kind():
    orr = parse_provider("openrouter/deepinfra")
    dw = parse_provider("doubleword/flex")
    assert sm.model_for(orr, "or-slug", "dw-slug") == "or-slug"
    assert sm.model_for(dw, "or-slug", "dw-slug") == "dw-slug"


def test_model_for_raises_when_slug_missing():
    orr = parse_provider("openrouter/deepinfra")
    dw = parse_provider("doubleword/flex")
    with pytest.raises(ValueError):
        sm.model_for(orr, None, "dw-slug")
    with pytest.raises(ValueError):
        sm.model_for(dw, "or-slug", None)


# --- streamed measurement ----------------------------------------------------


def test_measure_stream_ttft_decode_usage_from_chunks():
    # A fake monotonic clock: one tick per parsed data line.
    ticks = iter([1.0, 2.0, 3.0, 5.0])
    lines = [
        b'data: {"provider": "deepinfra"}',
        b'data: {"choices": [{"delta": {"reasoning": "thinking"}}]}',  # first delta at t=2
        b'data: {"choices": [{"delta": {"content": "hello"}}]}',  # content chunk
        b'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}], '
        b'"usage": {"completion_tokens": 6, "cost": 0.01}}',  # last delta at t=5
        b"data: [DONE]",
    ]
    m = sm.measure_stream(lines, lambda: next(ticks))
    assert m["first_delta"] == 2.0  # reasoning counts as first token
    assert m["last_delta"] == 5.0
    assert m["n_content_chunks"] == 2
    assert m["provider"] == "deepinfra"
    assert m["finish"] == "stop"
    assert m["usage"]["completion_tokens"] == 6


class _FakeResponse:
    """A urlopen-like streaming response usable as a context manager."""

    status = 200

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_one_call_assembles_record_from_stream(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sekret")
    lines = [
        b'data: {"provider": "deepinfra"}',
        b'data: {"choices": [{"delta": {"content": "hi"}}]}',
        b'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}], '
        b'"usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.002, '
        b'"completion_tokens_details": {"reasoning_tokens": 0}, '
        b'"prompt_tokens_details": {"cached_tokens": 0}}}',
        b"data: [DONE]",
    ]
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data)
        return _FakeResponse(lines)

    monkeypatch.setattr(sm.urlrequest, "urlopen", fake_urlopen)
    spec = parse_provider("openrouter/deepinfra")
    rec = sm.one_call(spec, "m", sm.REASONING_ON, "S", SHAPE_NO_RF, 1, 0)

    assert rec["route"] == "deepinfra"
    assert rec["mode"] == sm.REASONING_ON
    assert rec["shape"] == "S"
    assert rec["status"] == 200
    assert rec["provider_echo"] == "deepinfra"
    assert rec["finish_reason"] == "stop"
    assert rec["completion_tokens"] == 2
    assert rec["cost_usd"] == 0.002
    assert rec["ttft_s"] is not None
    assert "error" not in rec
    # the request was pinned and authenticated, never printing the key
    assert captured["auth"] == "Bearer sekret"
    assert captured["body"]["provider"] == PIN
    assert captured["url"].endswith("/chat/completions")


def test_one_call_records_http_error_without_retry(monkeypatch):
    from urllib import error as urlerror

    monkeypatch.setenv("DOUBLEWORD_API_KEY", "sekret")

    def boom(req, timeout=None):
        raise urlerror.HTTPError(
            req.full_url, 429, "Too Many Requests", {}, io_body("rate limited")
        )

    monkeypatch.setattr(sm.urlrequest, "urlopen", boom)
    spec = parse_provider("doubleword/flex")
    rec = sm.one_call(spec, "m", sm.REASONING_OFF, "S", SHAPE_NO_RF, 1, 0)
    assert rec["status"] == 429
    assert "rate limited" in rec["error"]
    assert rec["ttft_s"] is None


def io_body(text: str):
    import io

    return io.BytesIO(text.encode())


# --- summary aggregation -----------------------------------------------------


def _rec(route, mode, **kw):
    base = {"route": route, "mode": mode}
    base.update(kw)
    return base


def test_summarize_aggregates_per_route_mode():
    records = [
        _rec("deepinfra", "reasoning-on", ttft_s=1.0, decode_tps=100.0, total_s=5.0,
             completion_tokens=200, cost_usd=0.01),
        _rec("deepinfra", "reasoning-on", ttft_s=3.0, decode_tps=50.0, total_s=7.0,
             completion_tokens=400, cost_usd=0.02),
        _rec("deepinfra", "reasoning-on", status=429, error="rate limited"),
        _rec("deepinfra", "reasoning-off", ttft_s=0.5, decode_tps=120.0, total_s=2.0,
             completion_tokens=100),  # no cost reported
    ]
    rows = summarize_by_key(records)
    on = rows[("deepinfra", "reasoning-on")]
    assert on["n"] == 3
    assert on["errors"] == 1
    assert on["ttft_p50"] == 2.0  # median of [1.0, 3.0]; the errored row has no ttft
    assert on["completion_p50"] == 300.0
    assert on["cost_usd"] == pytest.approx(0.03)  # summed where present

    off = rows[("deepinfra", "reasoning-off")]
    assert off["n"] == 1
    assert off["errors"] == 0
    assert off["cost_usd"] is None  # no billed cost anywhere -> None, not 0


def summarize_by_key(records):
    return {(r["route"], r["mode"]): r for r in sm.summarize(records)}


def test_summarize_ttft_p90():
    records = [
        _rec("h", "m", ttft_s=float(i)) for i in range(1, 11)
    ]  # 1..10
    row = sm.summarize(records)[0]
    assert row["ttft_p90"] == pytest.approx(9.1)  # linear-interpolated p90


def test_percentile_handles_empty_and_single():
    assert sm.percentile([], 50) is None
    assert sm.percentile([None, None], 50) is None
    assert sm.percentile([4.0], 90) == 4.0


def test_format_summary_shows_dashes_for_missing_metrics():
    rows = sm.summarize([_rec("h", "m", status=None, error="boom")])
    table = sm.format_summary(rows)
    assert "route" in table and "cost$" in table
    assert "h" in table
    # every numeric metric was null -> rendered as a dash, cost too
    assert "-" in table


# --- CLI arg parsing ---------------------------------------------------------


def _parse(argv):
    import argparse

    # Rebuild just the serving subparser exactly as bench.main wires it, by
    # invoking the real parser through a monkeypatch-free path: parse_known via
    # the module's main is awkward, so we assert through the public parser.
    from compound import bench

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    # Mirror bench.main's serving subparser wiring for an isolated parse test.
    sp = sub.add_parser("serving")
    sp.add_argument("--providers", required=True)
    sp.add_argument("--shapes", required=True)
    sp.add_argument("--model-or", dest="model_or")
    sp.add_argument("--model")
    sp.add_argument("--rounds", type=int, default=1)
    sp.add_argument("--interval", type=float, default=3600.0)
    sp.add_argument("--reps", type=int, default=2)
    sp.add_argument("--out", default="artifacts/bench/serving-metrics")
    del bench  # imported only to assert the module is wired
    return parser.parse_args(argv)


def test_cli_serving_is_registered_on_the_real_parser(monkeypatch, tmp_path, capsys):
    # Drive the real bench.main dispatch, stubbing run_serving so no network runs.
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps({"S": SHAPE_NO_RF}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "present")
    seen = {}

    def fake_run_serving(specs, model_or, model, shapes_arg, **kw):
        seen["routes"] = [s.label for s in specs]
        seen["model_or"] = model_or
        seen["model"] = model
        seen["rounds"] = kw["rounds"]
        seen["interval"] = kw["interval"]
        seen["reps"] = kw["reps"]
        from pathlib import Path

        return Path(kw["out_dir"]) / "results.jsonl"

    monkeypatch.setattr(sm, "run_serving", fake_run_serving)
    monkeypatch.setattr(
        "sys.argv",
        [
            "compound-bench",
            "serving",
            "--providers",
            "openrouter/deepinfra,doubleword/flex",
            "--shapes",
            str(shapes),
            "--model-or",
            "or-slug",
            "--model",
            "dw-slug",
            "--rounds",
            "3",
            "--interval",
            "60",
            "--reps",
            "2",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    from compound.bench import main

    rc = main()
    assert rc == 0
    assert seen["routes"] == ["deepinfra", "doubleword-flex"]
    assert seen["model_or"] == "or-slug"
    assert seen["model"] == "dw-slug"
    assert seen["rounds"] == 3
    assert seen["interval"] == 60.0
    assert seen["reps"] == 2


def test_cli_serving_fails_fast_on_missing_model(monkeypatch, tmp_path):
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps({"S": SHAPE_NO_RF}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    monkeypatch.setattr(
        "sys.argv",
        [
            "compound-bench",
            "serving",
            "--providers",
            "openrouter/deepinfra",
            "--shapes",
            str(shapes),
            # no --model-or given for an OpenRouter route
        ],
    )
    from compound.bench import main

    with pytest.raises(SystemExit):
        main()


def test_parse_defaults():
    args = _parse(["serving", "--providers", "openrouter/auto", "--shapes", "s.json"])
    assert args.rounds == 1
    assert args.interval == 3600.0
    assert args.reps == 2
    assert args.out == "artifacts/bench/serving-metrics"
    assert args.model_or is None and args.model is None


def test_warm_cell_sends_the_prompt_unchanged_and_cold_nonces_it():
    # The cold/warm distinction is the cache measurement: a nonce makes a warm
    # hit impossible, so a warm cell must send byte-identical bytes every rep.
    from compound.providers_registry import parse_provider
    from compound.serving_metrics import CACHE_COLD, CACHE_WARM, REASONING_OFF, build_body

    spec = parse_provider("openrouter/novita")
    shape = {"messages": [{"role": "user", "content": "hello"}]}

    warm_a = build_body(spec, "m", REASONING_OFF, shape, nonce="")
    warm_b = build_body(spec, "m", REASONING_OFF, shape, nonce="")
    assert warm_a["messages"] == warm_b["messages"] == shape["messages"]

    cold_a = build_body(spec, "m", REASONING_OFF, shape)
    cold_b = build_body(spec, "m", REASONING_OFF, shape)
    assert cold_a["messages"] != cold_b["messages"]
    assert cold_a["messages"][0]["content"].endswith("hello")
    assert CACHE_COLD != CACHE_WARM  # the labels the records carry


def test_temperature_and_per_shape_max_tokens_reach_the_body():
    from compound.providers_registry import parse_provider
    from compound.serving_metrics import REASONING_OFF, build_body

    spec = parse_provider("openrouter/novita")
    shape = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 100}
    body = build_body(spec, "m", REASONING_OFF, shape, max_tokens=8192, temperature=0.0)
    assert body["temperature"] == 0.0
    # The shape's own budget wins: a profile grid varies output length per cell.
    assert body["max_tokens"] == 100

    no_shape_cap = {"messages": [{"role": "user", "content": "x"}]}
    assert build_body(spec, "m", REASONING_OFF, no_shape_cap, max_tokens=64)["max_tokens"] == 64


def test_measure_stream_accumulates_the_generated_text():
    from compound.serving_metrics import measure_stream

    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    ticks = iter([1.0, 2.0, 3.0, 4.0])
    m = measure_stream(lines, lambda: next(ticks))
    assert m["text"] == "Hello"
    assert m["n_content_chunks"] == 2


def test_marker_hosts_get_the_marker_the_proxy_would_inject(monkeypatch):
    # A serving comparison that skips the marker measures an opt-in cache host at
    # its worst case while every implicit-cache host is measured at its best.
    from compound.providers_registry import parse_provider
    from compound.serving_metrics import REASONING_OFF, build_body

    monkeypatch.delenv("COMPOUND_DW_CACHE", raising=False)  # markers default on
    shape = {"messages": [{"role": "user", "content": "hi"}]}

    dw = build_body(parse_provider("doubleword/realtime"), "m", REASONING_OFF, shape, nonce="")
    last = dw["messages"][-1]["content"]
    assert isinstance(last, list) and last[-1]["cache_control"]["type"] == "ephemeral"

    # OpenRouter caches prefixes on its own; marking it would be a no-op at best.
    orr = build_body(parse_provider("openrouter/novita"), "m", REASONING_OFF, shape, nonce="")
    assert orr["messages"][-1]["content"] == "hi"


def test_marker_can_be_turned_off_for_the_unmarked_path(monkeypatch):
    from compound.providers_registry import parse_provider
    from compound.serving_metrics import REASONING_OFF, build_body

    monkeypatch.setenv("COMPOUND_DW_CACHE", "0")
    shape = {"messages": [{"role": "user", "content": "hi"}]}
    dw = build_body(parse_provider("doubleword/realtime"), "m", REASONING_OFF, shape, nonce="")
    assert dw["messages"][-1]["content"] == "hi"


# --- Anthropic Messages dialect ----------------------------------------------

ANTHROPIC_CFG = {
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "type": "anthropic",
        "cache_strategy": "explicit_marker",
    },
    "openai-flex": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "service_tier": "flex",
        "timeout_s": 900,
        "max_tokens_field": "max_completion_tokens",
    },
}

SHAPE_WITH_SYSTEM = {
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ],
    "max_tokens": 64,
}


def test_messages_body_hoists_system_and_pins_thinking(monkeypatch):
    monkeypatch.delenv("COMPOUND_DW_CACHE", raising=False)
    spec = parse_provider("direct/anthropic", providers_config=ANTHROPIC_CFG)
    on = sm.build_body(spec, "claude-sonnet-5", sm.REASONING_ON, SHAPE_WITH_SYSTEM, nonce="")
    assert on["system"] == "be brief"
    assert [m["role"] for m in on["messages"]] == ["user"]
    assert on["thinking"] == {"type": "adaptive"}
    assert on["max_tokens"] == 64
    assert on["stream"] is True
    # current Claude models reject sampling parameters, so none is sent
    assert "temperature" not in on
    assert "reasoning_effort" not in on
    # The marker rides on the last message, exactly as it does for Doubleword.
    last = on["messages"][-1]["content"]
    assert last[-1]["cache_control"]["type"] == "ephemeral"
    off = sm.build_body(spec, "claude-sonnet-5", sm.REASONING_OFF, SHAPE_WITH_SYSTEM, nonce="")
    assert off["thinking"] == {"type": "disabled"}


def test_messages_body_translates_json_schema_and_refuses_json_object():
    spec = parse_provider("direct/anthropic", providers_config=ANTHROPIC_CFG)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    shape = {"messages": [{"role": "user", "content": "x"}],
             "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}}}
    body = sm.build_body(spec, "m", sm.REASONING_OFF, shape, nonce="")
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    loose = {"messages": [{"role": "user", "content": "x"}],
             "response_format": {"type": "json_object"}}
    with pytest.raises(ValueError, match="no Messages API equivalent"):
        sm.build_body(spec, "m", sm.REASONING_OFF, loose, nonce="")


def test_messages_body_cold_cell_still_nonces_the_first_turn():
    spec = parse_provider("direct/anthropic", providers_config=ANTHROPIC_CFG)
    a = sm.build_body(spec, "m", sm.REASONING_OFF, SHAPE_WITH_SYSTEM)
    b = sm.build_body(spec, "m", sm.REASONING_OFF, SHAPE_WITH_SYSTEM)
    # the nonce lands on the first turn, here the hoisted system text, which is
    # still the start of the cacheable prefix; the user turn stays byte-stable
    assert a["system"] != b["system"]
    assert a["system"].endswith("be brief") and b["system"].endswith("be brief")
    assert a["messages"] == b["messages"]


def test_measure_messages_stream_normalises_usage_and_times_first_delta():
    lines = [
        b"event: message_start",
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,'
        b'"cache_read_input_tokens":90,"cache_creation_input_tokens":5,'
        b'"output_tokens":1,"service_tier":"standard"}}}',
        b"event: content_block_start",
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}',
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hm"}}',
        b"event: ping",
        b'data: {"type":"ping"}',
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hel"}}',
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"lo"}}',
        b"event: message_delta",
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}',
        b"event: message_stop",
        b'data: {"type":"message_stop"}',
    ]
    # one clock sample per data line: message_start, content_block_start,
    # thinking delta, ping, text, text, message_delta, message_stop
    clock = iter([1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0])
    m = sm.measure_messages_stream(lines, lambda: next(clock))
    # first delta is the thinking delta (the model started answering), the
    # last is the final text delta
    assert m["first_delta"] == 3.0
    assert m["last_delta"] == 5.0
    assert m["n_content_chunks"] == 2
    assert m["text"] == "Hello"
    assert m["finish"] == "end_turn"
    assert m["usage"]["prompt_tokens"] == 105
    assert m["usage"]["prompt_tokens_details"] == {"cached_tokens": 90}
    assert m["usage"]["cache_write_tokens"] == 5
    assert m["usage"]["completion_tokens"] == 15
    assert m["usage"]["service_tier"] == "standard"
    assert m["error"] is None


def test_measure_messages_stream_records_a_mid_stream_error():
    lines = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}',
        b'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
    ]
    m = sm.measure_messages_stream(lines, lambda: 0.0)
    assert "overloaded_error" in m["error"]


def test_measure_messages_stream_treats_eof_without_message_stop_as_failure():
    lines = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"half"}}',
    ]
    m = sm.measure_messages_stream(lines, lambda: 0.0)
    assert m["error"] == "stream ended before message_stop"
    assert m["usage"] is None  # half a call is never priced as a whole one
    assert m["text"] == "half"


def test_measure_messages_stream_ignores_empty_deltas_for_timing():
    lines = [
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}',
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"tok"}}',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
        b'data: {"type":"message_stop"}',
    ]
    clock = iter([1.0, 2.0, 10.0, 11.0, 12.0])
    m = sm.measure_messages_stream(lines, lambda: next(clock))
    assert m["first_delta"] == 10.0 and m["last_delta"] == 10.0
    assert m["n_content_chunks"] == 1 and m["error"] is None


def test_one_call_anthropic_uses_messages_endpoint_and_key_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sekret")
    lines = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,'
        b'"cache_read_input_tokens":90,"cache_creation_input_tokens":0,"service_tier":"standard"}}}',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
        b'data: {"type":"message_stop"}',
    ]
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["timeout"] = timeout
        return _FakeResponse(lines)

    monkeypatch.setattr(sm.urlrequest, "urlopen", fake_urlopen)
    spec = parse_provider("direct/anthropic", providers_config=ANTHROPIC_CFG)
    rec = sm.one_call(spec, "claude-sonnet-5", sm.REASONING_OFF, "S", SHAPE_NO_RF, 1, 0,
                      temperature=0.0)

    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["X-api-key"] == "sekret"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert "Authorization" not in captured["headers"]
    assert rec["route"] == "anthropic"
    assert rec["model"] == "claude-sonnet-5"
    assert rec["prompt_tokens"] == 100
    assert rec["cached_tokens"] == 90
    assert rec["cache_write_tokens"] == 0
    assert rec["completion_tokens"] == 2
    assert rec["finish_reason"] == "end_turn"
    assert rec["service_tier_echo"] == "standard"
    assert rec["cost_usd"] is None  # Anthropic bills no per-call cost
    assert rec["temperature"] is None  # what was sent, not what was asked
    assert "error" not in rec


def test_openai_tier_echo_and_declared_timeout_reach_the_record(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sekret")
    lines = [
        b'data: {"service_tier":"flex","choices":[{"delta":{"content":"hi"}}]}',
        b'data: {"service_tier":"flex","choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":1,'
        b'"prompt_tokens_details":{"cached_tokens":0}}}',
        b"data: [DONE]",
    ]
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data)
        return _FakeResponse(lines)

    monkeypatch.setattr(sm.urlrequest, "urlopen", fake_urlopen)
    spec = parse_provider("direct/openai-flex", providers_config=ANTHROPIC_CFG)
    rec = sm.one_call(spec, "gpt-5.4-mini", sm.REASONING_OFF, "S", SHAPE_NO_RF, 1, 0)
    assert captured["body"]["service_tier"] == "flex"
    assert captured["timeout"] == 900  # the host's declared need beats the default
    # OpenAI's own field name for the output cap; max_tokens is a 400 there
    assert "max_tokens" not in captured["body"]
    assert captured["body"]["max_completion_tokens"] == sm.DEFAULT_MAX_TOKENS
    assert rec["service_tier_requested"] == "flex"
    assert rec["service_tier_echo"] == "flex"
    assert rec["cost_usd"] is None


def test_model_for_prefers_a_per_host_wire_model():
    from compound.providers_registry import apply_host_models, parse_providers

    specs = parse_providers("openrouter/deepinfra,direct/anthropic,direct/openai-flex",
                            providers_config=ANTHROPIC_CFG)
    specs = apply_host_models(
        specs, {"anthropic": "claude-sonnet-5", "openai-flex": "gpt-5.4-mini"},
        known_names=ANTHROPIC_CFG,
    )
    assert sm.model_for(specs[0], "or-slug", None) == "or-slug"
    assert sm.model_for(specs[1], "or-slug", None) == "claude-sonnet-5"
    assert sm.model_for(specs[2], "or-slug", None) == "gpt-5.4-mini"


def test_cli_serving_accepts_host_model(monkeypatch, tmp_path):
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps({"S": SHAPE_NO_RF}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.chdir(tmp_path)
    import yaml

    (tmp_path / "compound.yaml").write_text(yaml.safe_dump({"providers": ANTHROPIC_CFG}))
    seen = {}

    def fake_run_serving(specs, model_or, model, shapes_arg, **kw):
        seen["wire"] = [(s.label, s.wire_model) for s in specs]
        from pathlib import Path

        return Path(kw["out_dir"]) / "results.jsonl"

    monkeypatch.setattr(sm, "run_serving", fake_run_serving)
    monkeypatch.setattr("sys.argv", [
        "compound-bench", "serving",
        "--providers", "openrouter/deepinfra,direct/anthropic",
        "--shapes", str(shapes), "--model-or", "or-slug",
        "--host-model", "anthropic=claude-sonnet-5",
        "--out", str(tmp_path / "out"),
    ])
    from compound.bench import main

    assert main() == 0
    assert seen["wire"] == [("deepinfra", None), ("anthropic", "claude-sonnet-5")]


# --- derived cost ------------------------------------------------------------

RATES = {
    "anthropic": {"claude-sonnet-5": {"input": 2.0, "cached_input": 0.2,
                                      "cache_write": 2.5, "output": 10.0}},
    "telnyx": {"deepseek-v4-flash": {"input": 0.13, "cached_input": 0.03, "output": 0.26}},
}


def test_derived_cost_prices_each_token_class():
    rec = {"route": "anthropic", "model": "claude-sonnet-5", "prompt_tokens": 1_000_000,
           "cached_tokens": 500_000, "cache_write_tokens": 100_000,
           "completion_tokens": 10_000}
    # 400k uncached @2 + 500k read @0.2 + 100k write @2.5 + 10k out @10
    assert sm.derived_cost_usd(rec, RATES) == pytest.approx(0.8 + 0.1 + 0.25 + 0.1)


def test_derived_cost_defers_to_a_measured_cost_and_to_missing_data():
    measured = {"route": "anthropic", "model": "claude-sonnet-5", "prompt_tokens": 10,
                "completion_tokens": 1, "cost_usd": 0.002}
    assert sm.derived_cost_usd(measured, RATES) is None
    no_tokens = {"route": "anthropic", "model": "claude-sonnet-5"}
    assert sm.derived_cost_usd(no_tokens, RATES) is None
    unknown = {"route": "novita", "model": "x", "prompt_tokens": 10, "completion_tokens": 1}
    assert sm.derived_cost_usd(unknown, RATES) is None


def test_derived_cost_uses_the_only_card_when_a_ledger_has_no_model():
    old_ledger = {"route": "telnyx", "prompt_tokens": 1_000_000, "cached_tokens": 0,
                  "completion_tokens": 0}
    assert sm.derived_cost_usd(old_ledger, RATES) == pytest.approx(0.13)
    # a record that names a model the cards do not know is never priced at
    # another model's rate
    other = dict(old_ledger, model="some-other-model")
    assert sm.derived_cost_usd(other, RATES) is None
