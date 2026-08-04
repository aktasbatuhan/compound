"""Offline checks for the tau GEPA adapter: no network, no tau episodes."""

import json

import pytest

pytest.importorskip("gepa")

from compound.tau_gepa import (  # noqa: E402
    SEED_CANDIDATE,
    TauEpisodeTrace,
    TauGEPAAdapter,
    TauTask,
    _sim_feedback,
    load_partition_tasks,
)


def test_partitions_are_disjoint_and_complete() -> None:
    train = {t.case_id for t in load_partition_tasks("optimizer_train")}
    val = {t.case_id for t in load_partition_tasks("optimizer_validation")}
    decision = {t.case_id for t in load_partition_tasks("decision_test")}
    assert len(train) == 10 and len(val) == 4 and len(decision) == 6
    assert not (train & val) and not (train & decision) and not (val & decision)


def test_sim_feedback_extracts_signal() -> None:
    sim = {
        "termination_reason": "max_steps",
        "reward_info": {
            "reward": 0.5,
            "action_checks": [{"action": "cancel_flight", "met": False}],
            "info": {"note": "terminated prematurely"},
        },
        "messages": [
            {"role": "user", "content": "I need to cancel my flight to Boston"},
            {"role": "assistant", "content": "Let me check your reservation."},
        ],
    }
    reward, termination, feedback, first_user = _sim_feedback(sim)
    assert reward == 0.5
    assert termination == "max_steps"
    assert "cancel_flight" in feedback and "terminated prematurely" in feedback
    assert first_user.startswith("I need to cancel")


def test_gepa_optimize_integration_offline(tmp_path) -> None:
    """Drive the REAL gepa engine with a stubbed episode runner: proves the
    loader/adapter/reflection wiring end-to-end without spending a cent."""
    import gepa

    from compound.gepa_v2 import NamespacedTrialLoader
    from compound.tau_gepa import REFLECTION_TEMPLATE

    train = [TauTask("airline", str(i)) for i in range(4)]
    val = [TauTask("retail", str(i)) for i in range(2)]

    class StubAdapter(TauGEPAAdapter):
        def __init__(self) -> None:  # no runner
            pass

        def evaluate(self, batch, candidate, capture_traces=False):
            from gepa import EvaluationBatch

            # Longer instructions score higher: gives gepa a gradient to climb.
            score = min(1.0, 0.25 + 0.01 * len(candidate.get("agent_instruction", "")))
            traces = [
                TauEpisodeTrace(t, score, "done", f"reward={score}", "hi") for t in batch
            ]
            return EvaluationBatch(
                outputs=[{"case_id": t.case_id, "reward": score} for t in batch],
                scores=[score] * len(batch),
                trajectories=traces if capture_traces else None,
                objective_scores=[{"task_success": score}] * len(batch),
            )

    def stub_reflection(prompt):
        return "```text\nAlways confirm the customer's goal before acting, then act decisively.\n```"

    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=NamespacedTrialLoader(train, "train"),
        valset=NamespacedTrialLoader(val, "validation"),
        adapter=StubAdapter(),
        reflection_lm=stub_reflection,
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=2,
        reflection_prompt_template={"agent_instruction": REFLECTION_TEMPLATE},
        perfect_score=1.0,
        max_metric_calls=14,
        run_dir=str(tmp_path / "gepa"),
        cache_evaluation=False,
        seed=7,
        display_progress_bar=False,
    )
    assert result.best_candidate is not None
    # The stub rewards longer instructions, so the seed ("") must be beaten.
    assert len(result.best_candidate["agent_instruction"]) > 0
