import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from compound.core.providers_registry import parse_providers
from compound.serving import replay
from compound.serving.filler import FillerCorpus
from compound.shape.records import shape_from_messages

SALT = b"r" * 32
CORPUS = FillerCorpus({
    "en": [f"English sentence number {i} about invoices and payments." for i in range(50)],
    "tr": [f"Fatura ve ödeme hakkında {i} numaralı Türkçe cümle." for i in range(50)],
})
SYSTEM = {"role": "system", "content": "You check invoices for customers. " * 200}


def _shape(ts, user, *, extra=None, tokens=3000, out=120, system=SYSTEM):
    messages = [system, {"role": "user", "content": user}] + (extra or [])
    return shape_from_messages(messages, tools=None, salt=SALT, started_at=ts,
                               latency_ms=900, input_tokens=tokens, output_tokens=out)


def test_shared_prefixes_get_identical_filler_and_runs_do_not_share():
    a = replay.synthesize(_shape(0, "first question"), CORPUS, run_nonce="n1")
    b = replay.synthesize(_shape(1, "second question"), CORPUS, run_nonce="n1")
    c = replay.synthesize(_shape(0, "first question"), CORPUS, run_nonce="n2")
    assert a["messages"][0] == b["messages"][0]
    assert a["messages"][1] != b["messages"][1]
    assert a["messages"][0] != c["messages"][0]


def test_filler_keeps_roles_sizes_and_output_budget():
    turkish = {"role": "system", "content": "Faturaları kontrol eden bir asistansın. " * 200}
    extra = [{"role": "assistant", "content": "x"}, {"role": "tool", "content": "{}"}]
    shape = _shape(0, "Ödeme iki kez çekilmiş", extra=extra, system=turkish)
    body = replay.synthesize(shape, CORPUS, run_nonce="n")
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert body["lang"] == "tr" and "Reply in Turkish" in body["messages"][-1]["content"]
    assert body["max_tokens"] == 120 and body["target_input_tokens"] == 3000
    system_tokens = len(body["messages"][0]["content"]) * replay.DEFAULT_DENSITY["tr"] / 1000
    assert system_tokens == pytest.approx(3000 * 0.95, rel=0.1)


def test_plan_scales_by_load_and_picks_the_busiest_window():
    shapes = [_shape(t, f"q{t}") for t in (0, 10, 20, 3600, 3601, 3602, 3603)]
    at_1x = replay.plan(shapes)
    at_2x = replay.plan(shapes, load=2)
    assert [p.offset_s for p in at_2x] == [p.offset_s / 2 for p in at_1x]
    window = replay.plan(shapes, window_minutes=1)
    assert [p.shape.started_at for p in window] == [3600, 3601, 3602, 3603]
    assert len(replay.plan(shapes, max_calls=2)) == 2
    with pytest.raises(ValueError):
        replay.plan([shape_from_messages([], tools=None, salt=SALT, started_at=None)])


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _specs():
    config = {"a": {"base_url": "http://a/v1", "api_key_env": "A_KEY"},
              "b": {"base_url": "http://b/v1", "api_key_env": "B_KEY"}}
    return [(s, "m") for s in parse_providers("direct/a,direct/b", providers_config=config)]


def test_every_call_goes_to_every_route_on_schedule(tmp_path):
    clock, seen = FakeClock(), []

    def call(spec, model, mode, name, body, *_, **kw):
        seen.append((spec.label, name, clock.now, kw["cache_mode"]))
        return {"route": spec.label, "status": 200, "cost_usd": 0.001, "text": "secret answer"}

    planned = replay.plan([_shape(t, f"q{t}") for t in (0, 5, 9)])
    out = tmp_path / "r.jsonl"
    result = replay.run_replay(planned, _specs(), CORPUS, load=1, out_path=out, call=call,
                               clock=clock, sleep=clock.sleep, max_workers=1)
    assert result["sent"] == 6 and result["spent_usd"] == pytest.approx(0.006)
    assert {c for *_, c in seen} == {"warm"}
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["replay_index"] for r in rows} == {0, 1, 2}
    assert sorted({r["scheduled_offset_s"] for r in rows}) == [0, 5, 9]
    assert {r["send_lag_s"] for r in rows} == {0}
    assert sorted(r["route"] for r in rows if r["replay_index"] == 1) == ["a", "b"]
    assert all("text" not in r for r in rows)
    assert rows[0]["target_input_tokens"] == 3000


def test_spend_cap_counts_calls_still_in_flight(tmp_path):
    clock, release = FakeClock(), threading.Event()

    def call(spec, *_a, **_k):
        release.wait(5)  # nothing settles until every dispatch decision is made
        return {"route": spec.label, "status": 200, "cost_usd": 0.2}

    # Upper bound per call: 3,000 input + 120 output tokens at $100/$1,000 per M.
    prices = {"direct/a": (100.0, 1000.0), "direct/b": (100.0, 1000.0)}
    per_call = (3000 * 100 + 120 * 1000) / 1e6
    planned = replay.plan([_shape(t, f"q{t}") for t in range(5)])
    budget_for_three = per_call * 3 + 0.01
    result = {}

    def run():
        result.update(replay.run_replay(
            planned, _specs(), CORPUS, load=1, out_path=tmp_path / "r.jsonl", call=call,
            clock=clock, sleep=clock.sleep, max_usd=budget_for_three, prices=prices))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(1)
    release.set()
    worker.join(5)
    assert result["sent"] == 3 and result["skipped_over_budget"] == 7
    # Settled at the reported cost, not the reservation.
    assert result["spent_usd"] == pytest.approx(0.6)


def test_summary_separates_rate_limits_and_reports_cache_and_size():
    records = [
        {"route": "x", "replay_load": 2, "status": 200, "ttft_s": 0.5, "decode_tps": 80,
         "total_s": 3, "prompt_tokens": 1000, "cached_tokens": 600, "target_input_tokens": 1000,
         "cost_usd": 0.01, "send_lag_s": 0.01},
        {"route": "x", "replay_load": 2, "status": 200, "ttft_s": 0.7, "decode_tps": 60,
         "total_s": 4, "prompt_tokens": 1100, "cached_tokens": 0, "target_input_tokens": 1000,
         "cost_usd": 0.02, "send_lag_s": 0.02},
        {"route": "x", "replay_load": 2, "status": 429, "error": "slow down"},
        {"route": "x", "replay_load": 2, "status": None, "error": "TimeoutError"},
    ]
    (row,) = replay.summarize(records)
    assert (row["calls"], row["ok"], row["rate_limited"], row["other_errors"]) == (4, 2, 1, 1)
    assert row["cache_share"] == pytest.approx(600 / 2100)
    assert row["cost_per_call_usd"] == pytest.approx(0.015)
    assert "429" in replay.format_summary([row], 0.5)


def test_replay_end_to_end_through_the_real_streaming_client(tmp_path, monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            prompt = sum(len(m["content"]) for m in body["messages"]) // 4
            chunks = [
                {"choices": [{"delta": {"content": "Tamam, "}}]},
                {"choices": [{"delta": {"content": "kontrol ediyorum."}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": 5,
                                          "prompt_tokens_details": {"cached_tokens": 0}}},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LOCAL_KEY", "test")
    config = {"local": {"base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        "api_key_env": "LOCAL_KEY"}}
    routes = [(s, "local-model") for s in parse_providers("direct/local", providers_config=config)]
    try:
        planned = replay.plan([_shape(t * 0.05, f"q{t}") for t in range(3)])
        out = tmp_path / "r.jsonl"
        replay.run_replay(planned, routes, CORPUS, load=1, out_path=out)
    finally:
        server.shutdown()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3 and all(r["status"] == 200 for r in rows)
    assert all(r["ttft_s"] is not None and r["completion_tokens"] == 5 for r in rows)
    assert all(b["stream"] and b["max_tokens"] == 120 for b in seen)
    assert seen[0]["messages"][0] == seen[1]["messages"][0]  # shared prefix, same bytes


def test_cli_dry_run_sends_nothing(tmp_path, monkeypatch, capsys):
    from compound.cli import main

    path = tmp_path / "shapes.jsonl"
    path.write_text("\n".join(json.dumps(_shape(t, f"q{t}").to_json()) for t in range(4)))
    monkeypatch.setattr(sys, "argv", ["compound", "replay", str(path), "--providers",
                                      "openrouter/auto", "--model-or", "x/y", "--load", "1,2"])
    assert main() == 0
    out = capsys.readouterr().out
    assert "load 2x: 4 calls" in out and "dry run: nothing was sent" in out
