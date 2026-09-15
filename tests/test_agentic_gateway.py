import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from compound.agentic_gateway import (
    Gateway,
    ProviderResponseError,
    SpendGuard,
    chat_response,
    check_response_errors,
    responses_body,
)
from compound.agentic_study import plan


def test_parallel_reservations_cannot_each_spend_full_allowance(tmp_path):
    guard = SpendGuard(tmp_path / "spend.json", 1)

    def reserve(i):
        try:
            return guard.reserve(0.4, str(i), "agent")
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(8)))
    assert sum(x is not None for x in accepted) == 2
    assert sum(x["charged_or_reserved_usd"] for x in json.loads(guard.path.read_text())) == 0.8
    guard.settle(0, 0.1)
    reloaded = SpendGuard(guard.path, 1)
    reloaded.reserve(0.4, "next", "auxiliary")
    with pytest.raises(ValueError):
        reloaded.reserve(0.4, "over", "agent")


def test_episode_cap_is_atomic_and_separate_from_other_episodes(tmp_path):
    guard = SpendGuard(tmp_path / "spend.json", 1)

    def attempt(_):
        try:
            guard.reserve(0.03, "same", "agent", 0.1)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(8))) == 3
    guard.reserve(0.03, "different", "agent", 0.1)
    assert guard.remaining("same", "agent", 0.1) == pytest.approx(0.01)


def test_budget_adapts_output_then_stops_without_paid_call(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/budget-ten-dollar.json").read_text())
    study = plan(spec)
    e = next(e for e in study["episodes"] if e["route"] == "gpt-astra" and e["tier"] == "standard")
    gateway = Gateway(spec, study, tmp_path, limit=1)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    wires = []

    def respond(request, timeout):
        wires.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "service_tier": "default",
                    "usage": {"cost": 0.39},
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", respond)
    gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hello"}]})
    assert 0 < wires[0]["max_tokens"] < 8192
    with pytest.raises(ValueError, match="episode budget exhausted"):
        gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hello"}]})
    assert len(wires) == 1
    assert json.loads((tmp_path / "budget-stops.jsonl").read_text())["role"] == "agent"


def test_observed_token_bound_keeps_changed_bytes_and_framing_reserve(tmp_path):
    spec = json.loads(Path("benchmarks/flex-agentic/budget-ten-dollar.json").read_text())
    spec["controls"]["reservation_policy"] = "observed_tokens_plus_changed_bytes"
    gateway = Gateway(spec, plan(spec), tmp_path)
    first = {"messages": [{"role": "user", "content": "a" * 20000}], "max_tokens": 8192}
    original_bound, canonical = gateway.input_bound("e", "agent", first)
    gateway.input_evidence["e", "agent"] = (canonical, 5000)
    changed = {
        "messages": [*first["messages"], {"role": "assistant", "content": "b" * 500}],
        "max_tokens": 1,
    }
    bound, _ = gateway.input_bound("e", "agent", changed)
    assert 5000 + 500 + 4096 <= bound < original_bound
    assert gateway.input_bound("e", "auxiliary", changed)[0] > bound
    spec["controls"].pop("reservation_policy")
    assert gateway.input_bound("e", "agent", changed)[0] > bound


def test_responses_preserve_tools_and_require_cache_marker():
    body = {
        "model": "test",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "Compute this"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "calc", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "4"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}
        ],
    }
    wire = responses_body(body, "flex")
    assert "cache_control" not in wire["input"][0]["content"][0]
    assert wire["input"][1]["call_id"] == wire["input"][2]["call_id"] == "c1"
    assert wire["input"][2]["output"] == "4"
    assert wire["tools"][0]["name"] == "calc"
    assert wire["service_tier"] == "flex"


def test_responses_tool_reply_preserves_cached_usage():
    reply = chat_response(
        {
            "output": [
                {"type": "function_call", "call_id": "abc", "name": "calc", "arguments": "{}"}
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 80},
            },
            "service_tier": "flex",
        }
    )
    assert reply["choices"][0]["finish_reason"] == "tool_calls"
    assert reply["choices"][0]["message"]["tool_calls"][0]["id"] == "abc"
    assert reply["usage"]["prompt_tokens_details"]["cached_tokens"] == 80
    assert reply["service_tier"] == "flex"


def test_http_200_embedded_error_is_not_a_model_answer():
    raw = {
        "choices": [
            {
                "finish_reason": "error",
                "error": {"code": 429, "message": "upstream capacity exhausted"},
                "message": {"role": "assistant", "content": None},
            }
        ],
        "usage": {"cost": 0},
    }
    with pytest.raises(ProviderResponseError) as caught:
        check_response_errors(raw, "auxiliary")
    assert caught.value.code == 429
    assert caught.value.role == "auxiliary"


def test_gateway_preserves_signature_and_rejects_wrong_tier(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    study = plan(spec)
    e = next(e for e in study["episodes"] if e["route"] == "gemini-vertex" and e["tier"] == "flex")
    gateway = Gateway(spec, study, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    wires = []
    detail = [{"type": "reasoning.encrypted", "data": "opaque", "id": "call1"}]
    reply = {
        "service_tier": "flex",
        "provider": "Google",
        "usage": {"cost": 0.001},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call1", "function": {"name": "echo", "arguments": "{}"}}
                    ],
                    "reasoning_details": detail,
                }
            }
        ],
    }

    def respond(request, timeout):
        wires.append(json.loads(request.data))
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hi"}]})
    stripped = dict(reply["choices"][0]["message"])
    stripped.pop("reasoning_details")
    gateway.call(e["episode_id"], "agent", {"messages": [stripped]})
    assert wires[-1]["messages"][0]["reasoning_details"] == detail
    assert wires[-1]["provider"] == {
        "only": ["google-vertex/global/flex"],
        "allow_fallbacks": False,
    }
    reply["service_tier"] = "default"
    with pytest.raises(ValueError, match="served a different tier"):
        gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hi"}]})
    assert len(wires) == 3  # no retry or unpinned fallback
    assert sum(x["charged_or_reserved_usd"] for x in gateway.guard.entries) == pytest.approx(0.003)
    assert json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[-1])["status"] == 502


def test_missing_rates_names_every_unpriced_route_before_a_run_starts():
    from compound.agentic_gateway import missing_rates

    spec = {
        "models": [{"id": "deepseek-dw"}, {"id": "not-priced"}],
        "auxiliary_models": [{"id": "ds-sim"}],
    }
    assert missing_rates(spec) == ["not-priced/flex", "not-priced/standard"]
    assert missing_rates({"models": [{"id": "deepseek-dw"}]}) == []


def test_a_contradicting_tier_is_fatal_and_a_missing_one_is_recorded_unconfirmed():
    """A wrong tier means we measured the wrong thing; a silent one must stay visible."""
    from compound.agentic_gateway import RATES

    assert ("ds-sim", "standard") in RATES
    source = (Path(__file__).resolve().parents[1] / "src/compound/agentic_gateway.py").read_text()
    assert "if served is not None and served not in expected:" in source
    assert "provider served a different tier than requested" in source
    assert 'row["tier_confirmed"] = served is not None' in source
    assert '"response_echo" if served is not None else "billing_meter"' in source
