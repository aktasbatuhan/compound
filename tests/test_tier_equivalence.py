"""Boundary cases for the frozen equivalence analysis, checked before paid collection."""

import importlib.util
import random
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze_tier_equivalence",
    Path(__file__).resolve().parents[1] / "scripts/analyze_tier_equivalence.py",
)
tier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tier)


def synthetic(tasks, trials, p_standard, p_flex, seed=0, status="graded"):
    """Episodes and outcomes for a two-arm, equally replicated study."""
    rng = random.Random(seed)
    episodes, rows = {}, []
    for task in range(tasks):
        for trial in range(trials):
            for tier_name, p in (("standard", p_standard), ("flex", p_flex)):
                eid = f"{task}-{trial}-{tier_name}"
                episodes[eid] = {"episode_id": eid, "task_id": str(task), "tier": tier_name}
                rows.append(
                    {
                        "episode_id": eid,
                        "status": status,
                        "success": rng.random() < p if status == "graded" else None,
                    }
                )
    return episodes, rows


def test_no_true_difference_at_full_size_reads_as_equivalent():
    episodes, rows = synthetic(88, 7, 0.30, 0.30, seed=1)
    out = tier.analyse(episodes, rows, graded_only=False)
    assert out["tasks_paired"] == 88
    assert out["attempts"] == 88 * 7 * 2
    assert abs(out["task_weighted_difference"]) < 0.05
    assert out["verdict"] == "equivalent within the margin"


def test_a_large_true_difference_is_not_called_equivalent():
    episodes, rows = synthetic(88, 7, 0.45, 0.15, seed=2)
    out = tier.analyse(episodes, rows, graded_only=False)
    assert out["task_weighted_difference"] < -0.15
    assert out["verdict"] == "difference at least as large as the margin"


def test_a_small_sample_is_inconclusive_rather_than_equivalent():
    """The failure mode this study exists to avoid: reading thin data as 'no change'."""
    episodes, rows = synthetic(6, 2, 0.30, 0.30, seed=3)
    out = tier.analyse(episodes, rows, graded_only=False)
    assert out["verdict"] == "inconclusive at this sample size"


def test_a_difference_sitting_on_the_margin_is_not_declared_equivalent():
    episodes, rows = synthetic(88, 7, 0.30, 0.30 - tier.MARGIN, seed=4)
    out = tier.analyse(episodes, rows, graded_only=False)
    assert out["verdict"] != "equivalent within the margin"


def test_intention_to_treat_counts_a_timeout_as_a_failure_but_graded_only_drops_it():
    episodes, rows = synthetic(40, 4, 1.0, 1.0, seed=5)
    # Every flex attempt on half the tasks times out instead of returning.
    for eid, meta in episodes.items():
        if meta["tier"] == "flex" and int(meta["task_id"]) < 20:
            row = next(r for r in rows if r["episode_id"] == eid)
            row.update(status="timeout", success=None)
    itt = tier.analyse(episodes, rows, graded_only=False)
    graded = tier.analyse(episodes, rows, graded_only=True)
    # Half the tasks lose their entire flex arm, so ITT shows a large penalty.
    assert itt["task_weighted_difference"] < -0.4
    # Conditioning on graded episodes hides it: those tasks drop out entirely.
    assert graded["tasks_paired"] == 20
    assert abs(graded["task_weighted_difference"]) < 1e-9


def test_tasks_carry_equal_weight_regardless_of_how_many_trials_they_got():
    """One heavily replicated task must not outvote the other 87."""
    episodes, rows = synthetic(88, 3, 0.30, 0.30, seed=6)
    extra_episodes, extra_rows = synthetic(1, 200, 0.0, 1.0, seed=7)
    for eid, meta in extra_episodes.items():
        meta = dict(meta, task_id="0")
        episodes[eid + "-x"] = meta
    for row in extra_rows:
        rows.append(dict(row, episode_id=row["episode_id"] + "-x"))
    out = tier.analyse(episodes, rows, graded_only=False)
    # Task 0 is driven to +1.0, contributing at most 1/88 of the mean.
    assert out["tasks_paired"] == 88
    assert out["task_weighted_difference"] < 1.0 / 88 + 0.05


def test_a_task_missing_one_arm_is_reported_rather_than_silently_dropped():
    episodes, rows = synthetic(10, 2, 0.3, 0.3, seed=8)
    for eid, meta in list(episodes.items()):
        if meta["task_id"] == "4" and meta["tier"] == "flex":
            rows = [r for r in rows if r["episode_id"] != eid]
    out = tier.analyse(episodes, rows, graded_only=False)
    assert out["tasks_paired"] == 9
    assert out["tasks_unpaired"] == ["4"]


def test_the_interval_is_deterministic_for_a_given_dataset():
    episodes, rows = synthetic(30, 3, 0.3, 0.3, seed=9)
    first = tier.analyse(episodes, rows, graded_only=False)
    second = tier.analyse(episodes, rows, graded_only=False)
    assert (first["ci_low"], first["ci_high"]) == (second["ci_low"], second["ci_high"])
