import io
import json
import sys
import urllib.error

import pytest

from compound.ingest import gateways
from compound.shape import load_shapes

SALT = b"g" * 32
RECORD = {
    "id": "gen-abc", "created_at": "2026-10-30T09:00:00.000Z", "model": "deepseek/deepseek-v4.1-flash",
    "provider_name": "DeepInfra", "latency": 2400, "generation_time": 1900, "streamed": True,
    "finish_reason": "stop", "tokens_prompt": 1000, "tokens_completion": 80,
    "native_tokens_prompt": 1210, "native_tokens_completion": 95, "native_tokens_reasoning": 30,
    "native_tokens_cached": 1024, "total_cost": 0.00042, "session_id": "sess-1",
}
CONTENT = {"input": {"messages": [
    {"role": "system", "content": "Faturaları kontrol eden bir asistansın."},
    {"role": "user", "content": "Kartımdan iki kez çekim yapılmış."},
]}, "output": {"content": "Kontrol ediyorum."}, "error": None}


def fake_get(records, contents=None):
    contents = contents or {}

    def get(url):
        gen_id = url.split("id=")[1]
        table = contents if "/generation/content" in url else records
        if gen_id not in table:
            raise gateways.NotFound(url)
        return {"data": table[gen_id]}

    return get


def test_openrouter_latency_is_total_and_native_tokens_win():
    shape = gateways.shape_from_generation("openrouter", RECORD, salt=SALT)
    assert shape.latency_ms == 2400 and shape.extra.get("ttft_ms") is None
    assert (shape.input_tokens, shape.output_tokens) == (1210, 95)
    assert (shape.reasoning_tokens, shape.cached_input_tokens) == (30, 1024)
    assert shape.trace_id == "sess-1" and shape.extra["cost_usd"] == 0.00042
    assert shape.prefix_chain is None


def test_vercel_latency_is_time_to_first_token():
    shape = gateways.shape_from_generation("vercel", RECORD, salt=SALT)
    assert shape.latency_ms == 1900 and shape.extra["ttft_ms"] == 2400
    assert shape.ended_at - shape.started_at == pytest.approx(1.9)


def test_stored_prompts_become_hashes_and_roles_without_text():
    shape = gateways.shape_from_generation("openrouter", RECORD, content=CONTENT, salt=SALT)
    assert shape.roles == ["system", "user"] and shape.lang_hint == "tr"
    assert len(shape.prefix_chain) == 3  # tool list, then two messages
    dumped = json.dumps(shape.to_json())
    assert "Fatura" not in dumped and "Kartım" not in dumped


def test_plain_string_prompts_are_one_user_message():
    content = {"input": "Merhaba, faturam hatalı.", "output": {}}
    shape = gateways.shape_from_generation("openrouter", RECORD, content=content, salt=SALT)
    assert shape.roles == ["user"]


def test_import_reports_unknown_ids_and_keeps_metadata_when_content_is_missing():
    records = {"gen-abc": RECORD, "gen-def": {**RECORD, "id": "gen-def",
                                               "created_at": "2026-10-30T08:00:00Z"}}
    report = gateways.import_generations(
        "openrouter", ["gen-abc", "gen-def", "gen-missing"],
        fake_get(records, {"gen-abc": CONTENT}), salt=SALT, content=True,
    )
    assert report.not_found == ["gen-missing"]
    assert report.with_content == 1 and len(report.shapes) == 2
    assert report.shapes[0].extra["generation_id"] == "gen-def"  # sorted by start time


def test_vercel_has_no_stored_prompts():
    with pytest.raises(ValueError, match="stored prompts"):
        gateways.import_generations("vercel", ["gen_1"], fake_get({}), salt=SALT, content=True)


def test_ids_are_read_from_lists_csv_and_request_logs(tmp_path):
    plain = tmp_path / "ids.txt"
    plain.write_text("gen-1\ngen-2,2026-10-30\n\nnot-an-id\ngen-1\n")
    logs = tmp_path / "log.jsonl"
    logs.write_text(json.dumps({"response": {"id": "gen_01ARZ"}}) + "\n"
                    + json.dumps({"generation_id": "gen-3"}) + "\n{bad json\n")
    assert gateways.read_ids([plain, logs]) == ["gen-1", "gen-2", "gen_01ARZ", "gen-3"]


def test_activity_becomes_monthly_volume_per_model():
    rows = [
        {"date": "2026-10-29", "model": "m/a", "provider_name": "X", "requests": 100,
         "prompt_tokens": 200_000, "completion_tokens": 30_000, "reasoning_tokens": 0,
         "cached_tokens": 50_000, "usage": 1.5, "byok_usage_inference": 0},
        {"date": "2026-10-30", "model": "m/a", "provider_name": "Y", "requests": 300,
         "prompt_tokens": 600_000, "completion_tokens": 90_000, "reasoning_tokens": 0,
         "cached_tokens": 150_000, "usage": 4.5, "byok_usage_inference": 0},
    ]
    (m,) = gateways.summarize_activity(rows)
    assert m["monthly_calls"] == pytest.approx(400 / 2 * 30)
    assert m["avg_input_tokens"] == pytest.approx(2000) and m["avg_output_tokens"] == 300
    assert m["cached_share"] == pytest.approx(0.25)
    assert m["monthly_usd"] == pytest.approx(6.0 / 2 * 30)
    assert m["providers"] == {"X": 100, "Y": 300}


def test_activity_without_a_management_key_says_how_to_fix_it(monkeypatch):
    from compound.cli import main

    def forbidden(_key):
        def get(url):
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(b""))
        return get

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(gateways, "http_get", forbidden)
    monkeypatch.setattr(sys, "argv", ["compound", "gateway-activity"])
    with pytest.raises(SystemExit, match="management key"):
        main()


def test_cli_imports_ids_into_loadable_shapes(tmp_path, monkeypatch, capsys):
    from compound.cli import main

    ids = tmp_path / "ids.txt"
    ids.write_text("gen-abc\ngen-gone\n")
    out = tmp_path / "shapes.jsonl"
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(gateways, "http_get", lambda _key: fake_get({"gen-abc": RECORD},
                                                                     {"gen-abc": CONTENT}))
    monkeypatch.setattr(sys, "argv", ["compound", "gateway-import", str(ids), "--gateway",
                                      "openrouter", "--content", "--out", str(out)])
    assert main() == 0
    printed = capsys.readouterr().out
    assert "1 of 2 generations found, 1 with stored prompts; not found: 1" in printed
    (shape,) = load_shapes([out], b"other").shapes
    assert shape.cached_input_tokens == 1024 and shape.roles == ["system", "user"]
