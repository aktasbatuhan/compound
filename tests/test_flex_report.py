import importlib
from pathlib import Path


def test_merge_retains_interrupted_cost_without_charging_completed_attempt_twice(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    report = importlib.import_module("report_flex_complete")
    baseline = [{"episode_id": report.INTERRUPTED, "charged_or_reserved_usd": 0.3}]
    continuation = [{"episode_id": report.INTERRUPTED, "charged_or_reserved_usd": 0.4}]
    combined = report.combine_accounting(baseline, continuation)
    assert sum(c["charged_or_reserved_usd"] for c in combined) == 0.7
    assert combined[0]["episode_id"] == report.INTERRUPTED + "-interrupted"
    assert combined[1]["episode_id"] == report.INTERRUPTED
    assert baseline[0]["episode_id"] == report.INTERRUPTED


def test_responses_cost_correction_removes_only_unapplicable_write_markup(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    report = importlib.import_module("report_flex_complete")
    call = {
        "phase": "pilot",
        "episode_id": "a",
        "role": "agent",
        "reservation_usd": 1,
        "route": "deepseek-dw",
        "requested_tier": "standard",
        "request_sha256": "proof",
        "cost_kind": "derived_with_5m_cache_write_rate",
        "cost_usd": 0.1305,
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 100_000,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    spend = [
        {k: call[k] for k in ("phase", "episode_id", "role", "reservation_usd")}
        | {"settled": True, "charged_or_reserved_usd": 0.1305}
    ]
    corrections = report.correct_doubleword_costs([call], spend)
    assert abs(call["cost_usd"] - 0.108) < 1e-12
    assert spend[0]["charged_or_reserved_usd"] == call["cost_usd"]
    assert corrections[0]["request_sha256"] == "proof"
    assert report.correct_doubleword_costs([call], spend) == []


def test_error_waiver_releases_reservation_but_preserves_failure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    report = importlib.import_module("report_flex_complete")
    call = {
        "route": "gpt-sol",
        "status": 429,
        "error_type": "ProviderResponseError",
        "usage": {},
        "cost_usd": None,
        "phase": "pilot",
        "episode_id": "a",
        "role": "agent",
        "reservation_usd": 0.1,
        "request_sha256": "proof",
    }
    spend = [
        {k: call[k] for k in ("phase", "episode_id", "role", "reservation_usd")}
        | {"settled": False, "charged_or_reserved_usd": 0.1}
    ]
    # Missing usage evidence and transport failures do not qualify for this correction.
    ambiguous = {**call, "usage": None}
    transport = {**call, "error_type": "TimeoutError"}
    assert report.reconcile_error_waivers([ambiguous, transport], spend) == []
    events = report.reconcile_error_waivers([call], spend)
    assert events[0]["released_reservation_usd"] == 0.1
    assert call["cost_usd"] == 0 and call["status"] == 429
    assert spend[0]["settled"] and spend[0]["charged_or_reserved_usd"] == 0
    assert report.reconcile_error_waivers([call], spend) == []
