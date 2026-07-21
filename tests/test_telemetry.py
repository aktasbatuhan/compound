from __future__ import annotations

import json

from compound.telemetry import ingest_tau_results, summarize_model_calls


def test_telemetry_summary_groups_provider_and_model(tmp_path) -> None:
    records = [
        {
            "provider": "doubleword",
            "requested_model": "candidate",
            "status": "ok",
            "latency_ms": 1000,
            "input_tokens": 20,
            "output_tokens": 10,
            "reasoning_tokens": 4,
            "e2e_output_tps": 10.0,
        },
        {
            "provider": "doubleword",
            "requested_model": "candidate",
            "status": "error",
            "latency_ms": 200,
        },
    ]
    source = tmp_path / "calls.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))

    model = summarize_model_calls(source)["models"][0]

    assert model["calls"] == 2
    assert model["api_surface"] == "chat.completions"
    assert model["service_tier"] == "default"
    assert model["successful_calls"] == 1
    assert model["error_calls"] == 1
    assert model["latency_ms_p50"] == 1000
    assert model["e2e_output_tps_mean"] == 10.0
    assert model["e2e_output_tps_aggregate"] == 10.0


def test_telemetry_summary_separates_flex_from_default_route(tmp_path) -> None:
    records = [
        {
            "provider": "doubleword",
            "requested_model": "candidate",
            "status": "ok",
            "latency_ms": 1000,
            "output_tokens": 10,
            "e2e_output_tps": 10.0,
        },
        {
            "provider": "doubleword",
            "requested_model": "candidate",
            "status": "ok",
            "api_surface": "responses",
            "requested_service_tier": "flex",
            "latency_ms": 5000,
            "output_tokens": 20,
            "e2e_output_tps": 4.0,
        },
    ]
    source = tmp_path / "calls.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))

    models = summarize_model_calls(source)["models"]

    assert len(models) == 2
    assert {(item["api_surface"], item["service_tier"]) for item in models} == {
        ("chat.completions", "default"),
        ("responses", "flex"),
    }


def test_aggregate_tps_excludes_untimed_tokens(tmp_path) -> None:
    records = [
        {
            "provider": "openrouter",
            "requested_model": "model",
            "status": "ok",
            "latency_ms": 1000,
            "output_tokens": 10,
            "e2e_output_tps": 10.0,
        },
        {
            "provider": "openrouter",
            "requested_model": "model",
            "status": "ok",
            "latency_ms": None,
            "output_tokens": 90,
            "e2e_output_tps": None,
        },
    ]
    source = tmp_path / "calls.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))

    model = summarize_model_calls(source)["models"][0]

    assert model["calls"] == 2
    assert model["timed_calls"] == 1
    assert model["output_tokens"] == 100
    assert model["timed_output_tokens"] == 10
    assert model["e2e_output_tps_aggregate"] == 10.0


def test_tau_ingestion_is_idempotent(tmp_path) -> None:
    results = {
        "simulations": [
            {
                "id": "sim-1",
                "task_id": "0",
                "messages": [
                    {
                        "role": "assistant",
                        "turn_idx": 2,
                        "timestamp": "2026-01-01T00:00:00Z",
                        "generation_time_seconds": 2.0,
                        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                        "raw_data": {
                            "model": "resolved",
                            "choices": [{"finish_reason": "tool_calls"}],
                            "usage": {"completion_tokens_details": {"reasoning_tokens": 5}},
                        },
                    }
                ],
            }
        ]
    }
    source = tmp_path / "results.json"
    source.write_text(json.dumps(results))
    destination = tmp_path / "calls.jsonl"
    kwargs = {
        "telemetry_path": destination,
        "agent_provider": "doubleword",
        "agent_model": "candidate",
        "user_provider": "openrouter",
        "user_model": "user",
    }

    assert ingest_tau_results(source, **kwargs) == 1
    assert ingest_tau_results(source, **kwargs) == 0
    record = json.loads(destination.read_text())
    assert record["latency_ms"] == 2000
    assert record["e2e_output_tps"] == 10.0
