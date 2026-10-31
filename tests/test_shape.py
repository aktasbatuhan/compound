import json
import sys
from pathlib import Path

import pytest

from compound.shape import CacheAssumptions, CallShape, build_profile, load_shapes
from compound.shape.loaders import shape_from_request_log
from compound.shape.profile import peak_concurrency
from compound.shape.records import lang_hint, prefix_chain

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "packages/contract/fixtures"
SALT = b"s" * 32
SYSTEM = {"role": "system", "content": "You answer billing questions. " * 400}


def _log(ts, user, *, system=SYSTEM, usage=None, latency_ms=800):
    return {
        "ts": ts,
        "latency_ms": latency_ms,
        "request": {"model": "gpt-x", "messages": [system, {"role": "user", "content": user}]},
        "response": {"model": "gpt-x", "usage": usage or {"prompt_tokens": 3000}},
    }


def test_shared_prefixes_share_hashes_and_the_salt_changes_them():
    a = prefix_chain([SYSTEM, {"role": "user", "content": "one"}], None, SALT)
    b = prefix_chain([SYSTEM, {"role": "user", "content": "two"}], None, SALT)
    other_salt = prefix_chain([SYSTEM, {"role": "user", "content": "one"}], None, b"t" * 32)
    assert a[:2] == b[:2]
    assert a[2][0] != b[2][0]
    assert all(x[0] != y[0] for x, y in zip(a, other_salt, strict=True))


def test_a_changed_tool_list_breaks_the_whole_chain():
    with_tool = prefix_chain([SYSTEM], [{"name": "lookup"}], SALT)
    without = prefix_chain([SYSTEM], None, SALT)
    assert with_tool[0][0] != without[0][0]
    assert with_tool[1][0] != without[1][0]


def test_shape_records_carry_no_prompt_text():
    shape = shape_from_request_log(_log(1.0, "my card 4111 1111 1111 1111 was charged twice"), SALT)
    dumped = json.dumps(shape.to_json())
    assert "4111" not in dumped
    assert "billing" not in dumped
    assert shape.prompt_chars and shape.message_count == 2


def test_shape_records_round_trip_and_reject_other_schemas():
    shape = shape_from_request_log(_log(1.0, "hello"), SALT)
    assert CallShape.from_json(shape.to_json()) == shape
    with pytest.raises(ValueError):
        CallShape.from_json({"schema": "something.else"})


@pytest.mark.parametrize(
    ("text", "hint"),
    [
        ("Faturamda iki kez ücret alınmış, iade istiyorum.", "tr"),
        ("I was charged twice on my invoice.", "latin"),
        ("تم خصم المبلغ مرتين", "other"),
        ("1234 ---", None),
    ],
)
def test_lang_hint(text, hint):
    assert lang_hint(text) == hint


def test_trace_contract_calls_keep_usage_and_timing():
    report = load_shapes([FIXTURES / "eval-ready-full.json"], SALT)
    assert report.by_source["trace"] == 2
    first, second = report.shapes
    assert (first.input_tokens, first.cached_input_tokens, first.output_tokens) == (412, 256, 21)
    assert second.message_count == 4 and second.tool_count == 1
    assert second.latency_ms == pytest.approx(6039, abs=1)
    assert second.task_key == "support.invoice_triage"
    # The second call extends the first one's prompt, so their chains agree up
    # to the first call's last message.
    assert second.prefix_chain[:3] == first.prefix_chain[:3]


def test_usage_dialects_count_cached_input_the_same_way():
    anthropic = shape_from_request_log(
        _log(1.0, "x", usage={"input_tokens": 100, "cache_read_input_tokens": 900,
                              "cache_creation_input_tokens": 50, "output_tokens": 20}),
        SALT,
    )
    assert (anthropic.input_tokens, anthropic.cached_input_tokens) == (1050, 900)
    responses = shape_from_request_log(
        _log(1.0, "x", usage={"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600},
                              "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 30}}),
        SALT,
    )
    assert (responses.input_tokens, responses.cached_input_tokens) == (1000, 600)
    assert responses.reasoning_tokens == 30
    chat = shape_from_request_log(
        _log(1.0, "x", usage={"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 0},
                              "completion_tokens": 5}),
        SALT,
    )
    assert chat.cached_input_tokens == 0  # a measured zero, not unknown


def test_ledger_rows_load_without_a_prefix_chain(tmp_path):
    row = {"ts": 1_000.0, "route": "dw", "latency_ms": 2000.0, "status": 200, "upstream": "dw",
           "prompt_tokens": 500, "cached_tokens": None, "completion_tokens": 10, "abandoned": False}
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps(row) + "\n")
    (shape,) = load_shapes([path], SALT).shapes
    assert shape.started_at == pytest.approx(998.0)
    assert shape.cached_input_tokens is None and shape.prefix_chain is None
    assert shape.status == "ok"


def test_unrecognised_records_are_counted_not_guessed(tmp_path):
    path = tmp_path / "mixed.jsonl"
    lines = [json.dumps(_log(1.0, "a")), json.dumps({"hello": "world"}), "not json"]
    path.write_text("\n".join(lines) + "\n")
    report = load_shapes([path], SALT)
    assert len(report.shapes) == 1
    assert report.skipped["unrecognised"] == 2


def test_reuse_counts_repeated_prefixes_inside_the_ttl_only():
    shapes = [
        shape_from_request_log(_log(t, f"question {t}"), SALT) for t in (0.0, 60.0, 1_000.0)
    ]
    reuse = build_profile(shapes)["cache"]["reuse"]
    # Call 2 reuses call 1's system prompt; call 3 arrives after the TTL.
    per_call_chars = shapes[1].prefix_chain[-1][1]
    shared_chars = shapes[1].prefix_chain[1][1]
    expected = 3000 * shared_chars / per_call_chars
    assert reuse["reusable_tokens"] == pytest.approx(expected)
    assert reuse["input_tokens"] == 9000


def test_reuse_below_the_minimum_cacheable_prefix_is_zero():
    short = {"role": "system", "content": "Be brief."}
    usage = {"prompt_tokens": 40}
    shapes = [
        shape_from_request_log(_log(t, "q", system=short, usage=usage), SALT) for t in (0.0, 1.0)
    ]
    reuse = build_profile(shapes, CacheAssumptions(min_prefix_tokens=1024))["cache"]["reuse"]
    assert reuse["reusable_tokens"] == 0


def test_peak_concurrency_counts_overlap_but_not_back_to_back_calls():
    def call(start, end):
        return CallShape(started_at=start, ended_at=end)

    assert peak_concurrency([call(0, 10), call(5, 12), call(6, 7)]) == 3
    assert peak_concurrency([call(0, 10), call(10, 20)]) == 1
    assert peak_concurrency([CallShape(started_at=None)]) is None


def test_projection_and_burstiness_wait_for_enough_data():
    few = [shape_from_request_log(_log(t * 60.0, "q"), SALT) for t in range(5)]
    profile = build_profile(few)
    assert profile["monthly_projection"] is None
    assert profile["arrivals"]["burstiness"] is None

    day = 86_400.0
    many = [shape_from_request_log(_log(i * day / 40, "q"), SALT) for i in range(41)]
    profile = build_profile(many)
    assert profile["monthly_projection"]["calls"] == pytest.approx(41 * 30)
    assert profile["arrivals"]["burstiness"] == pytest.approx(1440 / 41 * 1, rel=0.05)


def test_cli_writes_a_profile_and_text_free_shapes(tmp_path, monkeypatch, capsys):
    from compound.cli import main

    out, calls = tmp_path / "profile.json", tmp_path / "shapes.jsonl"
    monkeypatch.setattr(
        sys, "argv",
        ["compound", "shape", str(FIXTURES / "eval-ready-full.json"),
         "--out", str(out), "--calls-out", str(calls)],
    )
    assert main() == 0
    profile = json.loads(out.read_text())
    assert profile["calls"] == 2
    assert profile["cache"]["observed"]["hit_share"] == pytest.approx(668 / 995)
    text = calls.read_text()
    assert "double-charged" not in text and "INV-2291" not in text
    # A shape file loads back as shapes.
    assert len(load_shapes([calls], SALT).shapes) == 2
    assert "2 calls (2 from trace)" in capsys.readouterr().out
