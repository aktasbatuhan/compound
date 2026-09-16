import copy
import json
from pathlib import Path

import pytest

from compound.agentic_study import finance_success, plan, summarize


@pytest.fixture
def spec():
    return json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())


def outcome(study, episode, **changes):
    return {
        "episode_id": episode["episode_id"],
        "spec_sha256": study["spec_sha256"],
        "status": "graded",
        "success": True,
        "grader": "test-grader@123",
        "duration_s": 120,
        "agent_cost_usd": 0.1,
        "auxiliary_cost_usd": 0,
        "sandbox_cost_usd": 0,
        **changes,
    }


def test_paired_plan_is_frozen_and_contains_no_mini(spec):
    p = plan(spec)
    assert p == plan(copy.deepcopy(spec))
    assert len(p["episodes"]) == len({e["episode_id"] for e in p["episodes"]}) == 150
    assert all("gpt-5.4-mini" not in m["model"] for m in spec["models"])
    for route in spec["models"]:
        for tier in ("standard", "flex"):
            assert (
                len([e for e in p["episodes"] if e["route"] == route["id"] and e["tier"] == tier])
                == 15
            )


def test_unrun_is_unknown_not_zero_success(spec):
    report = summarize(spec, [])
    assert all(g["success_rate"] is None and g["pending"] == 5 for g in report["groups"])


def test_failures_costs_and_deadlines_use_planned_denominator(spec):
    p = plan(spec)
    rows = [outcome(p, e) for e in p["episodes"]]
    target = p["episodes"][0]
    indices = [
        i
        for i, e in enumerate(p["episodes"])
        if all(e[k] == target[k] for k in ("suite", "route", "tier"))
    ]
    rows[indices[0]].update(status="provider_error", success=None)
    rows[indices[1]].update(success=False)
    report = summarize(spec, rows)
    g = next(
        g for g in report["groups"] if all(g[k] == target[k] for k in ("suite", "route", "tier"))
    )
    assert g["success_rate"] == 3 / 5
    assert g["graded_success_rate"] == 3 / 4
    assert g["success_by_deadline"] == {"60": 0, "300": 0.6, "900": 0.6}
    assert g["costs"]["agent_cost_usd"]["per_success"] == pytest.approx(0.5 / 3)


def test_missing_cost_does_not_become_zero(spec):
    p = plan(spec)
    rows = [outcome(p, e, agent_cost_usd=None) for e in p["episodes"]]
    assert all(
        g["costs"]["agent_cost_usd"]["per_success"] is None for g in summarize(spec, rows)["groups"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "provider_error", "success": True},
        {"success": 1},
        {"grader": ""},
        {"duration_s": float("nan")},
        {"agent_cost_usd": -1},
        {"spec_sha256": "other"},
    ],
)
def test_rejects_misleading_outcomes(spec, changes):
    p = plan(spec)
    with pytest.raises(ValueError):
        summarize(spec, [outcome(p, p["episodes"][0], **changes)])


def test_duplicate_episode_rejected(spec):
    p = plan(spec)
    r = outcome(p, p["episodes"][0])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(spec, [r, r])


def test_infrastructure_error_keeps_result_incomplete(spec):
    p = plan(spec)
    rows = [outcome(p, e, status="infrastructure_error", success=None) for e in p["episodes"]]
    assert all(g["success_rate"] is None for g in summarize(spec, rows)["groups"])


def test_finance_requires_citations_and_correct_inconsistency_handling():
    clean = {
        "parse_success": True,
        "expected_inconsistency": False,
        "final_balance_sheet_and_journal_entries_match": True,
        "inconsistency_flag_matches": True,
        "inconsistency_code_matches": True,
    }
    assert finance_success(clean)
    assert not finance_success({**clean, "final_balance_sheet_and_journal_entries_match": False})
    assert not finance_success({**clean, "inconsistency_code_matches": False})
    inconsistent = {
        "parse_success": True,
        "expected_inconsistency": True,
        "inconsistency_flag_matches": True,
        "inconsistency_code_matches": True,
        "inconsistency_empty_answer": True,
    }
    assert finance_success(inconsistent)
    assert not finance_success({**inconsistent, "inconsistency_empty_answer": False})
