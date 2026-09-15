import json
import runpy
from pathlib import Path

from compound.agentic_gateway import is_context_rejection

MODULE = Path("scripts/supervise_flex_gcp.py").resolve()


def rejection():
    inner = {
        "error": {
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "message": "The input token count exceeds the maximum number of tokens allowed 1048576.",
        }
    }
    return json.dumps({"error": {"code": 400, "metadata": {"raw": json.dumps(inner)}}})


def test_validation_waiver_does_not_cover_ambiguous_failures():
    assert is_context_rejection(400, rejection())
    assert not is_context_rejection(429, rejection())
    assert not is_context_rejection(400, '{"error":{"message":"unknown"}}')


def test_supervisor_waits_for_active_work_and_limits_restarts():
    choose = runpy.run_path(str(MODULE))["choose_action"]
    state = {"active": [123], "recorded": 10, "planned": 150, "guard_usd": 12, "infra": []}
    assert choose(state, checkpoint_due=True, boots=2) == "wait"
    state.update(active=[], guard_usd=1)
    assert choose(state, checkpoint_due=True, boots=1) == "restart"
    assert choose(state, checkpoint_due=True, boots=2) == "runtime_stop"
    assert choose(state, checkpoint_due=False, boots=1) == "unexpected_stop"
    assert choose(state, checkpoint_due=False, boots=1, released_usd=3) == "continue"
    state["infra"] = ["failed-sandbox"]
    assert choose(state, checkpoint_due=True, boots=1) == "infrastructure_stop"
    state.update(infra=[], guard_usd=11.7)
    assert choose(state, checkpoint_due=False, boots=1) == "budget_stop"
    state.update(recorded=150, guard_usd=5)
    assert choose(state, checkpoint_due=False, boots=1) == "complete"


def test_reconciliation_is_idempotent_and_keeps_uncertain_reservations(tmp_path):
    module = runpy.run_path(str(MODULE))
    directory = tmp_path / "results"
    directory.mkdir()
    (directory / "study-spec.json").write_text(
        json.dumps({"models": [{"id": "gemini-studio", "provider": "openrouter"}]})
    )
    entries = [
        {
            "episode_id": "rejected",
            "role": "agent",
            "reservation_usd": 3,
            "charged_or_reserved_usd": 3,
            "settled": False,
        },
        {
            "episode_id": "uncertain",
            "role": "agent",
            "reservation_usd": 0.1,
            "charged_or_reserved_usd": 0.1,
            "settled": False,
        },
    ]
    (directory / "spend.json").write_text(json.dumps(entries))
    call = {
        "episode_id": "rejected",
        "role": "agent",
        "route": "gemini-studio",
        "reservation_usd": 3,
        "status": 400,
        "error_body": rejection(),
        "request_sha256": "proof",
    }
    (directory / "calls.jsonl").write_text(json.dumps(call) + "\n")
    script = module["RECONCILE_SCRIPT"].replace(
        "/opt/compound/artifacts/flex-agentic-gcp", str(directory)
    )
    exec(script, {})
    exec(script, {})
    settled = json.loads((directory / "spend.json").read_text())
    assert settled[0]["settled"] and settled[0]["charged_or_reserved_usd"] == 0
    assert settled[1] == entries[1]
    assert len((directory / "reconciliations.jsonl").read_text().splitlines()) == 1
    assert (
        json.loads((directory / "calls.jsonl").read_text())["cost_kind"]
        == "documented_validation_waiver"
    )
