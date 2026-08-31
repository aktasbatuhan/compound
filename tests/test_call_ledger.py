"""Tests for the per-call ledger.

The ledger exists so cost, cache and routing claims rest on calls rather than
episodes, so these tests are mostly about the honesty of a single row: an
unreported field stays null, a failed call is still recorded, and a pin is
compared on a form the two dialects share.
"""

from __future__ import annotations

import json
import threading

import pytest

from compound.call_ledger import (
    CallLedger,
    build_record,
    format_summary,
    load_records,
    normalize_host,
    parse_response_payload,
    request_fields,
    summarize,
    usage_fields,
)


def sse(*chunks: dict) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


class TestParseResponsePayload:
    def test_plain_json_object(self):
        raw = json.dumps({"provider": "DeepInfra", "usage": {"prompt_tokens": 10}}).encode()
        assert parse_response_payload(raw) == {
            "provider": "DeepInfra",
            "usage": {"prompt_tokens": 10},
        }

    def test_streamed_chunks_merge_across_the_stream(self):
        # The provider echo arrives on the first chunk and usage on the last;
        # a row needs both, so the merge has to span the whole stream.
        raw = sse(
            {"provider": "Parasail", "choices": [{"delta": {"content": "hi"}}]},
            {"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 7}},
        )
        merged = parse_response_payload(raw, "text/event-stream")
        assert merged["provider"] == "Parasail"
        assert merged["usage"] == {"prompt_tokens": 7}

    def test_partial_leading_chunk_from_a_truncated_tail_is_skipped(self):
        raw = b'ider": "X"}\n\ndata: {"usage": {"prompt_tokens": 3}}\n\ndata: [DONE]\n\n'
        assert parse_response_payload(raw)["usage"] == {"prompt_tokens": 3}

    def test_unparseable_and_empty_bodies_are_none(self):
        assert parse_response_payload(b"") is None
        assert parse_response_payload(b"{not json") is None
        assert parse_response_payload(b'"a bare string"') is None


class TestUsageFields:
    def test_full_usage_is_extracted(self):
        payload = {
            "provider": "Fireworks",
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{"finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 2593,
                "completion_tokens": 523,
                "cost": 0.00091564,
                "prompt_tokens_details": {"cached_tokens": 2048},
                "completion_tokens_details": {"reasoning_tokens": 207},
            },
        }
        fields = usage_fields(payload)
        assert fields["provider_echo"] == "Fireworks"
        assert fields["prompt_tokens"] == 2593
        assert fields["cached_tokens"] == 2048
        assert fields["reasoning_tokens"] == 207
        assert fields["cost_usd"] == 0.00091564
        assert fields["finish_reason"] == "stop"

    def test_unreported_cache_is_null_not_zero(self):
        # A host that never reports cached tokens must not read as a measured
        # 0% hit rate: one is missing data, the other is a finding.
        fields = usage_fields({"usage": {"prompt_tokens": 100}})
        assert fields["cached_tokens"] is None
        assert fields["prompt_tokens"] == 100

    def test_missing_payload_yields_all_nulls(self):
        assert set(usage_fields(None).values()) == {None}


class TestRequestFields:
    def test_cache_marker_is_detected(self):
        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi",
                                              "cache_control": {"type": "ephemeral"}}]}
            ],
        }
        fields = request_fields(body)
        assert fields["cache_marked"] is True
        assert fields["messages"] == 1

    def test_plain_string_content_is_unmarked(self):
        fields = request_fields({"messages": [{"role": "user", "content": "hi"}]})
        assert fields["cache_marked"] is False

    @pytest.mark.parametrize(
        "body,expected",
        [
            ({"reasoning": {"enabled": True}}, "on"),
            ({"reasoning": {"enabled": False}}, "off"),
            ({"reasoning_effort": "medium"}, "on"),
            ({"reasoning_effort": "none"}, "off"),
            ({}, None),
        ],
    )
    def test_reasoning_pin_reads_both_dialects(self, body, expected):
        assert request_fields(body)["reasoning_pin"] == expected


class TestNormalizeHost:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("DeepInfra", "deepinfra"),
            ("deepinfra/fp4", "deepinfra"),
            ("Together", "together"),
            (None, None),
        ],
    )
    def test_dialects_fold_together(self, raw, expected):
        assert normalize_host(raw) == expected


class TestBuildRecord:
    def test_honored_pin_across_slug_and_display_name(self):
        # The pin is a quant-tagged slug and the echo a display name. Comparing
        # them raw would call every honored pin a violation.
        record = build_record(
            route="deepinfra-fp4",
            upstream="deepinfra/fp4",
            status=200,
            latency_ms=1234.5,
            request_body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            response_raw=json.dumps({"provider": "DeepInfra", "usage": {}}).encode(),
        )
        assert record["pin_honored"] is True
        assert record["route"] == "deepinfra-fp4"
        assert record["latency_ms"] == 1234.5

    def test_escaped_pin_is_flagged(self):
        record = build_record(
            route="deepinfra-fp4",
            upstream="deepinfra/fp4",
            status=200,
            latency_ms=1.0,
            request_body=None,
            response_raw=json.dumps({"provider": "Together"}).encode(),
        )
        assert record["pin_honored"] is False

    def test_unpinned_auto_route_records_the_served_host_without_a_verdict(self):
        # openrouter/auto has no pin to honor. The echo is the measurement:
        # it is how routing hops between upstreams become visible.
        record = build_record(
            route="openrouter-auto",
            upstream=None,
            status=200,
            latency_ms=1.0,
            request_body=None,
            response_raw=json.dumps({"provider": "Parasail"}).encode(),
        )
        assert record["provider_echo"] == "Parasail"
        assert record["pin_honored"] is None

    def test_failed_call_is_still_a_row(self):
        # A 429 that killed an episode is reliability data, not a gap.
        record = build_record(
            route="fireworks",
            upstream="fireworks",
            status=429,
            latency_ms=12.0,
            request_body=None,
            response_raw=b'{"error": {"message": "rate limited"}}',
            error="http_429",
        )
        assert record["status"] == 429
        assert record["error"] == "http_429"
        assert record["prompt_tokens"] is None


class TestCallLedger:
    def test_rows_append_as_jsonl(self, tmp_path):
        ledger = CallLedger(tmp_path / "nested" / "calls.jsonl")
        ledger.write({"route": "a"})
        ledger.write({"route": "b"})
        lines = (tmp_path / "nested" / "calls.jsonl").read_text().splitlines()
        assert [json.loads(line)["route"] for line in lines] == ["a", "b"]

    def test_concurrent_writes_do_not_interleave(self, tmp_path):
        # The proxy is a ThreadingHTTPServer: without the lock, partial lines
        # from two threads would corrupt the file.
        ledger = CallLedger(tmp_path / "calls.jsonl")
        payload = "x" * 2000

        def writer(i: int) -> None:
            for _ in range(20):
                ledger.write({"i": i, "pad": payload})

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = (tmp_path / "calls.jsonl").read_text().splitlines()
        assert len(lines) == 160
        assert all(json.loads(line)["pad"] == payload for line in lines)


class TestSummarize:
    def rows(self, records):
        return {row["route"]: row for row in summarize(records)}

    def test_cache_hit_rate_ignores_hosts_that_never_reported(self):
        # Two calls: one host reports the split, one stays silent. Counting the
        # silent one as a zero would invent a cache finding out of missing data.
        records = [
            {"route": "a", "status": 200, "prompt_tokens": 100, "cached_tokens": 80},
            {"route": "a", "status": 200, "prompt_tokens": 100, "cached_tokens": None},
        ]
        row = self.rows(records)["a"]
        assert row["cache_reported"] == 1
        assert row["cached_tokens"] == 80
        assert row["calls"] == 2

    def test_cache_hit_rate_is_null_when_nobody_reported(self):
        records = [{"route": "dw", "status": 200, "prompt_tokens": 500, "cached_tokens": None}]
        assert self.rows(records)["dw"]["cache_hit_rate"] is None

    def test_routing_spread_counts_distinct_upstreams(self):
        # The auto route's whole finding: identical work, three different hosts.
        records = [
            {"route": "auto", "status": 200, "provider_echo": h}
            for h in ("DeepInfra", "Parasail", "DeepInfra", "Together")
        ]
        row = self.rows(records)["auto"]
        assert row["distinct_upstreams"] == 3
        assert row["upstreams"]["DeepInfra"] == 2

    def test_errors_and_pin_violations_are_counted(self):
        records = [
            {"route": "f", "status": 200, "pin_honored": True},
            {"route": "f", "status": 429, "error": "http_429", "pin_honored": None},
            {"route": "f", "status": 200, "pin_honored": False},
        ]
        row = self.rows(records)["f"]
        assert row["calls"] == 3
        assert row["errors"] == 1
        assert row["pin_violations"] == 1

    def test_cost_is_null_unless_a_provider_billed_it(self):
        records = [{"route": "dw", "status": 200, "cost_usd": None}]
        assert self.rows(records)["dw"]["cost_usd"] is None

    def test_format_summary_renders_every_route(self):
        records = [
            {"route": "auto", "status": 200, "provider_echo": "DeepInfra",
             "prompt_tokens": 100, "cached_tokens": 50, "cost_usd": 0.001, "latency_ms": 900},
            {"route": "deepinfra/fp4", "status": 200, "provider_echo": "DeepInfra",
             "prompt_tokens": 100, "cached_tokens": None, "latency_ms": 800},
        ]
        text = format_summary(summarize(records))
        assert "auto" in text and "deepinfra/fp4" in text
        assert "50.0" in text          # the reported hit rate
        assert "—" in text             # the unreported one stays a dash


class TestLoadRecords:
    def test_half_written_final_line_is_skipped(self, tmp_path):
        # A run killed mid-write must still yield every complete call.
        path = tmp_path / "calls.jsonl"
        path.write_text('{"route": "a"}\n{"route": "b"}\n{"route": "trunc"')
        assert [r["route"] for r in load_records(path)] == ["a", "b"]
