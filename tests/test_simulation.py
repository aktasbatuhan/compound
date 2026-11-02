import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from compound.core.providers_registry import parse_provider
from compound.eval.simulation import (
    ChatResult,
    Environment,
    LLMUser,
    ScriptedUser,
    SpendCap,
    decide,
    fidelity,
    grade,
    provider_chat,
    run_episode,
    scenario_from_trace,
)
from compound.eval.simulation.run import Arm, simulate
from compound.eval.simulation.scenario import Action

TOOLS = [
    {"name": "lookup_invoice", "description": "Fetch an invoice",
     "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}}}},
    {"name": "refund", "description": "Refund one charge",
     "parameters": {"type": "object", "properties": {"charge_id": {"type": "string"}}}},
]
HISTORY = [
    {"role": "system", "content": "You are a billing assistant."},
    {"role": "user", "content": "Faturam INV-1 iki kez çekilmiş."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "name": "lookup_invoice", "arguments": {"invoice_id": "INV-1"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": '{"charges": ["ch_1", "ch_2"]}'},
    {"role": "assistant", "content": "İki çekim görüyorum. ch_2'yi iade edeyim mi?"},
    {"role": "user", "content": "Evet lütfen."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c2", "name": "refund", "arguments": {"charge_id": "ch_2"}}]},
    {"role": "tool", "tool_call_id": "c2", "content": '{"ok": true}'},
]
TRACE = {
    "schema": "compound.trace", "schema_version": 1, "trace_id": "t1", "task_key": "billing",
    "steps": [{"type": "model_call", "step_id": "g", "model": "gpt-x", "input": HISTORY,
               "tools_available": TOOLS, "output": {"role": "assistant",
                                                    "content": "İade edildi."}}],
}


def scenario():
    return scenario_from_trace(TRACE)


def _call(i, name, args):
    return {"id": f"x{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def faithful_model(refund_charge="ch_2", cost=0.001):
    """Behaves like the real assistant, reading the transcript to decide."""
    def chat(messages, tools):
        last = messages[-1]
        tools_done = [m for m in messages if m["role"] == "tool"]
        if last["role"] == "user" and len([m for m in messages if m["role"] == "user"]) == 1:
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [_call(1, "lookup_invoice", {"invoice_id": "INV-1"})]}
        elif last["role"] == "tool" and len(tools_done) == 1:
            msg = {"role": "assistant", "content": "ch_2'yi iade edeyim mi?"}
        elif last["role"] == "user":
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [_call(2, "refund", {"charge_id": refund_charge})]}
        else:
            msg = {"role": "assistant", "content": "İade edildi."}
        return ChatResult(msg, {"prompt_tokens": 100, "completion_tokens": 10}, cost, 50)
    return chat


def test_scenario_keeps_turns_actions_and_recordings():
    s = scenario()
    assert s.user_turns == ["Faturam INV-1 iki kez çekilmiş.", "Evet lütfen."]
    assert s.reference_actions == [Action("lookup_invoice", '{"invoice_id":"INV-1"}'),
                                   Action("refund", '{"charge_id":"ch_2"}')]
    assert s.recorded[Action("refund", '{"charge_id":"ch_2"}')] == '{"ok": true}'
    assert s.policies == {"lookup_invoice": "recorded", "refund": "recorded"}
    assert s.incumbent_model == "gpt-x" and s.lang == "tr" and not s.notes
    assert s.tools[0]["function"]["name"] == "lookup_invoice"


def test_environment_follows_each_policy():
    s = scenario()
    env = Environment(recorded=s.recorded, policies={
        "lookup_invoice": "recorded", "refund": "blocked", "ping": "mocked",
        "search": "live_read_only", "other": "live_read_only"},
        mock_results={"ping": "pong"}, live={"search": lambda args: f"found {args['q']}"})
    assert env.call("lookup_invoice", '{"invoice_id": "INV-1"}').content.startswith('{"charges"')
    assert env.call("lookup_invoice", {"invoice_id": "INV-9"}).stop == "missing_recorded_result"
    assert env.call("refund", {"charge_id": "ch_2"}).stop == "tool_blocked"
    assert env.call("ping", {}).content == "pong"
    assert env.call("search", '{"q": "fatura"}').content == "found fatura"
    assert env.call("other", {}).stop == "unsupported_policy"


def test_a_faithful_model_reproduces_the_conversation_and_passes():
    s = scenario()
    ep = run_episode(s, faithful_model(), ScriptedUser(), Environment.for_scenario(s), model="m")
    assert ep.stop == "user_done" and ep.user_turns == 2 and ep.model_calls == 4
    assert grade(ep, s).passed and ep.cost_usd == pytest.approx(0.004)
    assert [m["role"] for m in ep.transcript] == [
        "system", "user", "assistant", "tool", "assistant", "user", "assistant", "tool",
        "assistant"]


def test_refunding_the_wrong_charge_stops_at_an_unrecorded_call_and_fails():
    s = scenario()
    ep = run_episode(s, faithful_model("ch_1"), ScriptedUser(), Environment.for_scenario(s),
                     model="m")
    assert ep.stop == "missing_recorded_result" and ep.stop_tool == "refund"
    g = grade(ep, s)
    assert not g.passed and g.reason == "stopped: missing_recorded_result"
    assert g.recall == pytest.approx(0.5) and g.precision == pytest.approx(0.5)


def test_a_model_that_never_answers_is_truncated():
    s = scenario()

    def looping(messages, tools):
        return ChatResult({"role": "assistant", "content": None, "tool_calls": [
            _call(1, "lookup_invoice", {"invoice_id": "INV-1"})]}, {}, None, 1)

    ep = run_episode(s, looping, ScriptedUser(), Environment.for_scenario(s), model="m",
                     max_model_calls=5)
    assert ep.stop == "truncated" and ep.model_calls == 5 and not ep.cost_complete


def test_spend_cap_stops_before_the_next_call():
    s = scenario()
    cap = SpendCap(1.5)
    ep = run_episode(s, faithful_model(cost=1.0), ScriptedUser(), Environment.for_scenario(s),
                     model="m", cap=cap)
    assert ep.stop == "budget" and ep.model_calls == 2


def test_llm_user_sees_roles_flipped_and_can_end_the_conversation():
    seen = []

    def simulator(messages, tools):
        seen.append(messages)
        text = "Evet, ch_2." if len(seen) == 1 else "###STOP###"
        return ChatResult({"role": "assistant", "content": text}, {}, 0.0001, 5)

    s = scenario()
    user = LLMUser(simulator)
    ep = run_episode(s, faithful_model(), user, Environment.for_scenario(s), model="m")
    assert ep.stop == "user_done" and grade(ep, s).passed
    brief, first_user, first_reply = seen[0][0], seen[0][1], seen[0][-1]
    assert brief["role"] == "system" and "Evet lütfen." in brief["content"]
    assert "Turkish" in brief["content"]
    assert first_user == {"role": "assistant", "content": "Faturam INV-1 iki kez çekilmiş."}
    assert first_reply["role"] == "user"  # the assistant's reply, from the simulator's side
    assert len(user.usage) == 2


def test_verdicts_cover_every_outcome():
    faithful = fidelity([True] * 9 + [False], threshold=0.8, min_n=5)
    assert faithful["faithful"] and faithful["rate"] == pytest.approx(0.9)
    shaky = fidelity([True, False, False, True, False], threshold=0.8, min_n=5)
    assert decide([True] * 5, [True] * 5, margin=0.05, fidelity_report=shaky,
                  min_n=5)["verdict"] == "simulator_unfaithful"
    same = [True] * 40
    assert decide(same, same, margin=0.05, fidelity_report=faithful,
                  min_n=10)["verdict"] == "meets_gate"
    worse = [True] * 10 + [False] * 30
    assert decide(worse, same, margin=0.05, fidelity_report=faithful,
                  min_n=10)["verdict"] == "fails_gate"
    assert decide([True] * 3, [True] * 3, margin=0.05, fidelity_report=faithful,
                  min_n=10)["verdict"] == "insufficient_data"


def test_simulate_checks_fidelity_first_and_grades_candidates(tmp_path):
    scenarios = [scenario() for _ in range(12)]
    good = Arm("prod", "gpt-x", faithful_model())
    bad = Arm("cheap", "small", faithful_model("ch_1"))
    result = simulate(scenarios, good, [bad], ScriptedUser, out_dir=tmp_path, min_n=10)
    assert result["fidelity"]["faithful"]
    verdict = result["candidates"]["cheap"]
    assert verdict["verdict"] == "fails_gate" and verdict["candidate_pass_rate"] == 0
    assert verdict["stops"] == {"missing_recorded_result": 12}
    rows = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert len(rows) == 24 and json.loads((tmp_path / "verdict.json").read_text())


def test_an_unfaithful_simulator_means_no_candidate_is_called(tmp_path):
    called = []

    def candidate(messages, tools):
        called.append(1)
        return faithful_model()(messages, tools)

    scenarios = [scenario() for _ in range(12)]
    result = simulate(scenarios, Arm("prod", "gpt-x", faithful_model("ch_1")),
                      [Arm("cand", "m", candidate)], ScriptedUser, out_dir=tmp_path, min_n=10)
    assert result["candidates"]["cand"]["verdict"] == "simulator_unfaithful"
    assert called == []


def test_cli_dry_run_describes_the_plan(tmp_path, monkeypatch, capsys):
    from compound.cli import main

    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(TRACE) + "\n")
    monkeypatch.setattr(sys, "argv", ["compound", "simulate", str(path), "--incumbent",
                                      "openrouter/auto=openai/gpt-x", "--candidates",
                                      "openrouter/auto=z-ai/glm-5.3"])
    assert main() == 0
    out = capsys.readouterr().out
    assert "1 scenarios, 2.0 user turns and 2.0 tool calls" in out
    assert "dry run: nothing was sent" in out


def test_provider_chat_round_trips_tool_calls_with_the_real_client(monkeypatch):
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            reply = {"id": "r", "object": "chat.completion", "created": 0, "model": "m",
                     "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                         "role": "assistant", "content": None,
                         "tool_calls": [_call(1, "lookup_invoice", {"invoice_id": "INV-1"})]}}],
                     "usage": {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48,
                               "cost": 0.0002}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LOCAL_KEY", "test")
    spec = parse_provider("direct/local", providers_config={"local": {
        "base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key_env": "LOCAL_KEY"}})
    try:
        s = scenario()
        result = provider_chat(spec, "m")(s.system + [{"role": "user", "content": "hi"}], s.tools)
    finally:
        server.shutdown()
    assert result.message["tool_calls"][0]["function"]["name"] == "lookup_invoice"
    assert result.cost_usd == pytest.approx(0.0002)
    assert bodies[0]["tools"][0]["function"]["name"] == "lookup_invoice"
    assert bodies[0]["tool_choice"] == "auto"
