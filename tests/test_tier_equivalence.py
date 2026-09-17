"""Integrity and boundary tests for the frozen equivalence analysis."""

import importlib.util
import random
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze_tier_equivalence",
    Path(__file__).resolve().parents[1] / "scripts/analyze_tier_equivalence.py",
)
tier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tier)


def synthetic(tasks=88, trials=7, p_standard=0.30, p_flex=0.30, seed=0, status="graded"):
    """A full one-suite, one-route plan and its outcome rows."""
    rng = random.Random(seed)
    episode_list, rows = [], []
    for task in range(tasks):
        for trial in range(trials):
            for tier_name, probability in (("standard", p_standard), ("flex", p_flex)):
                eid = f"{task}-{trial}-{tier_name}"
                episode_list.append(
                    {
                        "episode_id": eid,
                        "suite": "retail",
                        "route": "doubleword",
                        "task_id": str(task),
                        "trial": trial,
                        "tier": tier_name,
                    }
                )
                rows.append(
                    {
                        "episode_id": eid,
                        "spec_sha256": "frozen-spec",
                        "status": status,
                        "success": rng.random() < probability if status == "graded" else None,
                    }
                )
    plan = {
        "spec_sha256": "frozen-spec",
        "episode_count": len(episode_list),
        "episodes": episode_list,
    }
    return plan, {e["episode_id"]: e for e in episode_list}, rows


def primary(plan, episodes, rows):
    issues = tier.confirmatory_issues(plan, rows)
    return tier.analyse(episodes, rows, False, integrity_issues=issues), issues


def test_no_true_difference_at_full_size_can_read_as_equivalent():
    plan, episodes, rows = synthetic(seed=1)
    out, issues = primary(plan, episodes, rows)
    assert issues == []
    assert out["tasks_paired"] == 88
    assert out["attempts"] == 88 * 7 * 2
    assert abs(out["task_weighted_difference"]) < 0.05
    assert out["confirmatory_eligible"] is True
    assert out["verdict"] == "equivalent within the margin"


def test_a_large_true_difference_is_not_called_equivalent():
    plan, episodes, rows = synthetic(p_standard=0.45, p_flex=0.15, seed=2)
    out, _ = primary(plan, episodes, rows)
    assert out["task_weighted_difference"] < -0.15
    assert out["verdict"] == "difference at least as large as the margin"


def test_four_of_twenty_eight_rows_remain_descriptive_but_never_confirmatory():
    plan, episodes, rows = synthetic(tasks=2, trials=7, seed=3)
    out, issues = primary(plan, episodes, rows[:4])
    assert out["attempts"] == 4
    assert out["task_weighted_difference"] is not None
    assert out["verdict"] == "invalid for confirmatory inference"
    assert out["confirmatory_eligible"] is False
    assert any("recorded 4 outcomes for 28 planned" in issue for issue in issues)
    assert any("requires 88 tasks" in issue for issue in issues)


def test_missing_pair_is_named_and_blocks_confirmation():
    plan, episodes, rows = synthetic(seed=4)
    rows.pop()
    out, issues = primary(plan, episodes, rows)
    assert out["verdict"] == "invalid for confirmatory inference"
    assert any("missing 1 planned outcomes" in issue for issue in issues)


def test_duplicate_and_unknown_outcomes_are_not_silently_counted():
    plan, episodes, rows = synthetic(seed=5)
    rows.extend([dict(rows[0]), {**rows[1], "episode_id": "alien"}])
    out, issues = primary(plan, episodes, rows)
    assert out["verdict"] == "invalid for confirmatory inference"
    assert any("duplicate outcomes" in issue for issue in issues)
    assert any("unknown outcomes" in issue for issue in issues)


def test_mismatched_spec_identity_blocks_confirmation():
    plan, episodes, rows = synthetic(seed=6)
    rows[0]["spec_sha256"] = "another-study"
    out, issues = primary(plan, episodes, rows)
    assert out["verdict"] == "invalid for confirmatory inference"
    assert any("do not match the plan specification" in issue for issue in issues)


def test_mixed_routes_are_never_pooled_into_one_task_effect():
    plan, episodes, rows = synthetic(seed=7)
    for episode in plan["episodes"][:2]:
        episode["route"] = "other-route"
    out, issues = primary(plan, episodes, rows)
    assert out["verdict"] == "invalid for confirmatory inference"
    assert any("requires one route" in issue for issue in issues)
    assert out["tasks_paired"] == 89


def test_mixed_suites_are_never_pooled_into_one_task_effect():
    plan, episodes, rows = synthetic(seed=8)
    for episode in plan["episodes"][:2]:
        episode["suite"] = "coding"
    out, issues = primary(plan, episodes, rows)
    assert out["verdict"] == "invalid for confirmatory inference"
    assert any("requires one suite" in issue for issue in issues)
    assert out["tasks_paired"] == 89


def test_budget_grid_is_never_pooled_into_one_equivalence_effect():
    plan, episodes, rows = synthetic(seed=18)
    for episode in plan["episodes"][:2]:
        episode["budget_usd"] = 0.25
    out, issues = primary(plan, episodes, rows)
    assert out["verdict"] == "invalid for confirmatory inference"
    assert any("requires one budget level" in issue for issue in issues)
    assert out["tasks_paired"] == 89


def test_duplicate_plan_ids_and_broken_tier_pairs_are_detected():
    plan, _, rows = synthetic(seed=9)
    plan["episodes"][1]["episode_id"] = plan["episodes"][0]["episode_id"]
    plan["episodes"][1]["tier"] = "standard"
    issues = tier.confirmatory_issues(plan, rows)
    assert any("duplicate episode IDs" in issue for issue in issues)
    assert any("incomplete or duplicate tier pairs" in issue for issue in issues)


def test_infrastructure_or_budget_invalidates_the_study():
    for invalid_status in ("infrastructure_error", "budget_exhausted"):
        plan, episodes, rows = synthetic(seed=10)
        rows[0].update(status=invalid_status, success=None)
        out, issues = primary(plan, episodes, rows)
        assert out["verdict"] == "invalid for confirmatory inference"
        assert any("study-invalidating outcomes" in issue for issue in issues)


def test_provider_timeout_is_an_itt_failure_without_invalidating_study():
    plan, episodes, rows = synthetic(p_standard=1.0, p_flex=1.0, seed=11)
    target = next(
        row
        for row in rows
        if episodes[row["episode_id"]]["tier"] == "flex"
        and episodes[row["episode_id"]]["task_id"] == "0"
    )
    target.update(status="timeout", success=None)
    out, issues = primary(plan, episodes, rows)
    assert issues == []
    assert out["attempts"] == 88 * 7 * 2
    assert out["task_weighted_difference"] < 0


def test_tasks_remain_equally_weighted_in_descriptive_estimate():
    _, episodes, rows = synthetic(seed=16)
    # Add many duplicate observations to task 0. This malformed study must be
    # blocked by integrity checks in normal use, while the estimand still gives
    # task 0 only one of 88 task-level weights.
    for trial in range(100):
        for tier_name, success in (("standard", False), ("flex", True)):
            eid = f"extra-{trial}-{tier_name}"
            episodes[eid] = {
                "episode_id": eid,
                "suite": "retail",
                "route": "doubleword",
                "task_id": "0",
                "trial": 100 + trial,
                "tier": tier_name,
            }
            rows.append(
                {
                    "episode_id": eid,
                    "spec_sha256": "frozen-spec",
                    "status": "graded",
                    "success": success,
                }
            )
    out = tier.analyse(episodes, rows, False, ["synthetic malformed study"])
    assert out["tasks_paired"] == 88
    assert out["task_weighted_difference"] < 0.08


def test_graded_only_is_explicitly_descriptive():
    _, episodes, rows = synthetic(seed=12)
    out = tier.analyse(episodes, rows, True, integrity_issues=[], confirmatory=False)
    assert out["analysis_role"] == "descriptive secondary"
    assert out["confirmatory_eligible"] is False
    assert out["verdict"] == "descriptive only"


def test_degenerate_bootstrap_cannot_establish_equivalence():
    plan, episodes, rows = synthetic(p_standard=1.0, p_flex=1.0, seed=13)
    out, issues = primary(plan, episodes, rows)
    assert issues == []
    assert out["descriptive_statistical_assessment"] == "equivalent within the margin"
    assert out["ci_low"] == out["ci_high"] == 0
    assert out["verdict"] == "invalid for confirmatory inference"
    assert any("degenerate" in reason for reason in out["confirmatory_blockers"])


def test_a_difference_sitting_on_the_margin_is_not_declared_equivalent():
    plan, episodes, rows = synthetic(p_standard=0.30, p_flex=0.30 - tier.MARGIN, seed=14)
    out, _ = primary(plan, episodes, rows)
    assert out["verdict"] != "equivalent within the margin"


def test_the_interval_is_deterministic_for_a_given_dataset():
    plan, episodes, rows = synthetic(seed=15)
    issues = tier.confirmatory_issues(plan, rows)
    first = tier.analyse(episodes, rows, False, issues)
    second = tier.analyse(episodes, rows, False, issues)
    assert (first["ci_low"], first["ci_high"]) == (second["ci_low"], second["ci_high"])
