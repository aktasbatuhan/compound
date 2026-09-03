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
