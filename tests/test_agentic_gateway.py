import io
import json
import threading
import urllib.error
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


def test_budget_keeps_fixed_output_cap_and_stops_without_paid_call(tmp_path, monkeypatch):
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
    with pytest.raises(ValueError, match="episode budget exhausted"):
        gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hello"}]})
    assert wires == []
    assert json.loads((tmp_path / "budget-stops.jsonl").read_text())["role"] == "agent"


def test_legacy_budget_policy_must_be_explicit_to_shrink_output(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/budget-ten-dollar.json").read_text())
    spec["controls"]["budget_output_policy"] = "shrink_legacy"
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


def test_observed_token_evidence_is_isolated_by_attempt(tmp_path):
    spec = json.loads(Path("benchmarks/flex-agentic/budget-ten-dollar.json").read_text())
    spec["controls"]["reservation_policy"] = "observed_tokens_plus_changed_bytes"
    gateway = Gateway(spec, plan(spec), tmp_path)
    payload = {"messages": [{"role": "user", "content": "a" * 20000}]}
    original, canonical = gateway.input_bound("e", "agent", payload, attempt_id="a")
    gateway.input_evidence["e", "a", "agent"] = (canonical, 100)
    same_attempt, _ = gateway.input_bound("e", "agent", payload, attempt_id="a")
    other_attempt, _ = gateway.input_bound("e", "agent", payload, attempt_id="b")
    assert same_attempt < original
    assert other_attempt == original


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
    spec["controls"]["reasoning_effort"] = "low"
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
        "require_parameters": True,
    }
    assert wires[-1]["reasoning"] == {"effort": "low"}
    reply["service_tier"] = "default"
    with pytest.raises(ValueError, match="served a different tier"):
        gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hi"}]})
    assert gateway.stop_event.is_set()
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


def test_changed_model_identity_requires_its_own_declared_rate_card():
    from compound.agentic_gateway import missing_rates, rates_for

    replacement = {
        "id": "deepseek-dw",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
    }
    assert missing_rates({"models": [replacement]}) == [
        "deepseek-dw/flex",
        "deepseek-dw/standard",
    ]
    replacement["rates"] = {
        "model": replacement["model"],
        "standard": {
            "input_per_million": 0.1,
            "cache_read_per_million": 0.02,
            "output_per_million": 0.2,
        },
        "flex": {
            "input_per_million": 0.08,
            "cache_read_per_million": 0.01,
            "output_per_million": 0.15,
        },
    }
    assert missing_rates({"models": [replacement]}) == []
    assert rates_for(replacement, "deepseek-dw", "flex") == (
        (0.08, 0.01, 0.15),
        "model_declared",
    )


def test_402_sets_shared_stop_and_refuses_later_call(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    study = plan(spec)
    e = next(e for e in study["episodes"] if e["route"] == "gemini-vertex")
    stopped = threading.Event()
    gateway = Gateway(spec, study, tmp_path, stop_event=stopped)
    gateway.active_attempts[e["episode_id"]] = "attempt-7"
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    calls = 0

    def payment_required(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url, 402, "payment required", {}, io.BytesIO(b'{"error":"credit"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", payment_required)
    with pytest.raises(urllib.error.HTTPError):
        gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hi"}]})
    assert stopped.is_set()
    with pytest.raises(ValueError, match="study stop requested"):
        gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "again"}]})
    assert calls == 1
    assert gateway.guard.entries[0]["attempt_id"] == "attempt-7"
    row = json.loads((tmp_path / "calls.jsonl").read_text())
    assert row["attempt_id"] == "attempt-7"
    assert row["status"] == 402


def test_doubleword_honors_reasoning_ttl_and_leaves_missing_tier_unverified(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    spec["controls"].update(reasoning_effort="high", cache_ttl="1h")
    study = plan(spec)
    e = next(
        e for e in study["episodes"] if e["route"] == "deepseek-dw" and e["tier"] == "standard"
    )
    gateway = Gateway(spec, study, tmp_path)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "fake")
    wires = []

    def respond(request, timeout):
        wires.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "id": "dw-1",
                    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "cache_creation_input_tokens": 100,
                    },
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", respond)
    gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hi"}]})
    assert wires[0]["reasoning_effort"] == "high"
    assert wires[0]["messages"][-1]["content"][-1]["cache_control"]["ttl"] == "1h"
    row = json.loads((tmp_path / "calls.jsonl").read_text())
    assert row["tier_confirmed"] is False
    assert row["tier_evidence"] == "unverified"
    assert row["requested_endpoint"] == "realtime"
    expected = (100 * 0.09 * 2 + 10 * 0.18) / 1e6
    assert row["cost_usd"] == pytest.approx(expected)


def test_malformed_derived_token_counts_keep_reservation_unsettled(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    study = plan(spec)
    e = next(
        e for e in study["episodes"] if e["route"] == "deepseek-dw" and e["tier"] == "standard"
    )
    gateway = Gateway(spec, study, tmp_path)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "fake")

    def respond(request, timeout):
        return io.BytesIO(
            json.dumps(
                {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 1,
                        "cache_read_input_tokens": 11,
                    },
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", respond)
    with pytest.raises(ValueError, match="invalid provider token counts"):
        gateway.call(e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hi"}]})
    assert gateway.guard.entries[0]["settled"] is False


def test_missing_usage_counts_return_answer_but_keep_reservation(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    study = plan(spec)
    e = next(
        e for e in study["episodes"] if e["route"] == "deepseek-dw" and e["tier"] == "standard"
    )
    gateway = Gateway(spec, study, tmp_path)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "fake")

    def respond(request, timeout):
        return io.BytesIO(
            json.dumps(
                {
                    "usage": {"prompt_tokens": 10},
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", respond)
    reply = gateway.call(
        e["episode_id"], "agent", {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert reply["choices"][0]["message"]["content"] == "done"
    assert gateway.guard.entries[0]["settled"] is False
    row = json.loads((tmp_path / "calls.jsonl").read_text())
    assert row["cost_usd"] is None


def test_a_contradicting_tier_is_fatal_and_a_missing_one_is_recorded_unconfirmed():
    """A wrong tier means we measured the wrong thing; a silent one must stay visible."""
    from compound.agentic_gateway import RATES

    assert ("ds-sim", "standard") in RATES
    source = (Path(__file__).resolve().parents[1] / "src/compound/agentic_gateway.py").read_text()
    assert "if served is not None and served not in expected:" in source
    assert "provider served a different tier than requested" in source
    assert 'row["tier_confirmed"] = served is not None' in source
    assert '"response_echo" if served is not None else "unverified"' in source


def test_output_cap_and_truncation_are_recorded(tmp_path, monkeypatch):
    """A reply cut short by the study's own token ceiling must be auditable.

    Without this the gateway silently clamps the harness's requested max_tokens
    down to the spec control and records nothing, so a truncated completion is
    indistinguishable from a model that answered badly and the task is scored a
    failure.
    """
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    spec["controls"]["max_output_tokens"] = 4096
    study = plan(spec)
    e = next(e for e in study["episodes"] if e["route"] == "gemini-vertex" and e["tier"] == "flex")
    gateway = Gateway(spec, study, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    reply = {
        "service_tier": "flex",
        "usage": {"cost": 0.001},
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "cut off mid-"},
            }
        ],
    }
    sent = []

    def respond(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    gateway.call(
        e["episode_id"],
        "agent",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8192},
    )

    assert sent[-1]["max_tokens"] == 4096
    row = json.loads((tmp_path / "calls.jsonl").read_text().strip().splitlines()[-1])
    assert row["requested_output_tokens"] == 8192
    assert row["max_output_tokens"] == 4096
    assert row["output_cap_applied"] is True
    assert row["finish_reason"] == "length"
    assert row["output_truncated"] is True


def test_untruncated_reply_under_the_cap_is_not_flagged(tmp_path, monkeypatch):
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    spec["controls"]["max_output_tokens"] = 8192
    study = plan(spec)
    e = next(e for e in study["episodes"] if e["route"] == "gemini-vertex" and e["tier"] == "flex")
    gateway = Gateway(spec, study, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    reply = {
        "service_tier": "flex",
        "usage": {"cost": 0.001},
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(reply).encode()),
    )
    gateway.call(
        e["episode_id"],
        "agent",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4096},
    )
    row = json.loads((tmp_path / "calls.jsonl").read_text().strip().splitlines()[-1])
    assert row["requested_output_tokens"] == 4096
    assert row["output_cap_applied"] is False
    assert row["output_truncated"] is False
