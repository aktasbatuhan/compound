from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import gepa
import pytest

from compound.budget import BudgetLedger
from compound.contracts import Usage
from compound.costs import TokenPrices
from compound.ds1000_optimizer import DS1000Example, completion_fingerprint
from compound.gepa_v2 import (
    MAX_CANDIDATE_WORDS,
    SEED_COMPONENTS,
    DS1000GEPAAdapterV2,
    ExperimentCap,
    NamespacedTrialLoader,
    TrialExample,
    _target_satisfied,
    compose_system_prompt,
    select_library_components,
)
from compound.gepa_v2_runner import (
    _decision_manifest_cases,
    _load_teacher_traces,
    _paired_comparison,
)
from compound.providers import ProviderResponse


def _response(text: str) -> ProviderResponse:
    return ProviderResponse(
        output={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
        usage=Usage(input_tokens=100, output_tokens=20),
        latency_ms=10,
    )


def _adapter(
    tmp_path: Path, provider: Mock, *, flex_runner: Mock | None = None
) -> DS1000GEPAAdapterV2:
    ledger = BudgetLedger.load(tmp_path / "budget.json", 25)
    return DS1000GEPAAdapterV2(
        provider=provider,
        model="candidate",
        prices=TokenPrices(1, 2),
        ledger=ledger,
        experiment_cap=ExperimentCap(ledger, ledger.spent_usd, 4),
        cache_dir=tmp_path / "cache",
        docker_image="test-image",
        max_tokens=4096,
        reasoning_effort=None,
        critic=None,
        flex_runner=flex_runner,
    )


def _example(trial_id: int = 0) -> TrialExample:
    base = DS1000Example(
        "ds1000_1",
        "result = ... # put solution in this variable\nBEGIN SOLUTION\n<code>",
        "def test_execution(code): pass",
        {"library": "Numpy"},
    )
    return TrialExample(base, trial_id)


def test_trial_fingerprint_preserves_trial_zero_and_separates_repeats() -> None:
    base = completion_fingerprint(
        case_id="case",
        model="model",
        system_prompt="prompt",
        max_tokens=10,
        reasoning_effort=None,
    )
    explicit_zero = completion_fingerprint(
        case_id="case",
        model="model",
        system_prompt="prompt",
        max_tokens=10,
        reasoning_effort=None,
        trial_id=0,
    )
    repeat = completion_fingerprint(
        case_id="case",
        model="model",
        system_prompt="prompt",
        max_tokens=10,
        reasoning_effort=None,
        trial_id=1,
    )

    assert base == explicit_zero
    assert repeat != base


def test_prompt_composition_uses_only_the_relevant_library_strategy() -> None:
    numpy_prompt = compose_system_prompt(SEED_COMPONENTS, "NumPy")
    pandas_prompt = compose_system_prompt(SEED_COMPONENTS, "pandas")

    assert SEED_COMPONENTS["response_contract"] in numpy_prompt
    assert SEED_COMPONENTS["numpy_strategy"] in numpy_prompt
    assert SEED_COMPONENTS["pandas_strategy"] not in numpy_prompt
    assert SEED_COMPONENTS["pandas_strategy"] in pandas_prompt
    assert SEED_COMPONENTS["numpy_strategy"] not in pandas_prompt


def test_target_check_requires_an_assignment_or_return() -> None:
    prompt = "result = ... # put solution in this variable\nBEGIN SOLUTION"
    assert _target_satisfied(prompt, "result + 1") == 0.0
    assert _target_satisfied(prompt, "result = values + 1") == 1.0
    assert _target_satisfied(prompt, "return values + 1") == 1.0


def test_gepa_loaders_use_disjoint_train_and_validation_ids() -> None:
    train = NamespacedTrialLoader([_example()], "train")
    validation = NamespacedTrialLoader([_example()], "validation")

    assert set(train.all_ids()).isdisjoint(validation.all_ids())
    assert train.fetch(["train:0"]) == [_example()]


def test_library_only_selector_never_mutates_the_shared_contract() -> None:
    numpy_failure = Mock()
    numpy_failure.library = "Numpy"
    numpy_failure.metrics = {"task_success": 0.0}
    pandas_success = Mock()
    pandas_success.library = "Pandas"
    pandas_success.metrics = {"task_success": 1.0}

    selected = select_library_components(None, [numpy_failure, pandas_success], [], 0, {})

    assert selected == ["numpy_strategy"]
    assert "response_contract" not in selected


def test_teacher_loader_prefers_first_passing_model_and_filters_cases(tmp_path: Path) -> None:
    artifact = tmp_path / "teachers.json"
    artifact.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "gpt",
                        "cases": [
                            {"case_id": "a", "trial_id": 0, "passed": False, "completion": "bad"},
                            {"case_id": "b", "trial_id": 0, "passed": True, "completion": "gpt-b"},
                        ],
                    },
                    {
                        "model": "opus",
                        "cases": [
                            {"case_id": "a", "trial_id": 0, "passed": True, "completion": "opus-a"},
                            {"case_id": "b", "trial_id": 0, "passed": True, "completion": "opus-b"},
                            {"case_id": "c", "trial_id": 0, "passed": True, "completion": "opus-c"},
                        ],
                    },
                ]
            }
        )
    )

    teachers = _load_teacher_traces(
        artifact, preferred_models=["gpt", "opus"], allowed_case_ids={"a", "b"}
    )

    assert teachers == {
        "a": {"model": "opus", "completion": "opus-a"},
        "b": {"model": "gpt", "completion": "gpt-b"},
    }


def test_external_decision_manifest_rejects_origin_overlap(tmp_path: Path) -> None:
    optimization = tmp_path / "optimization.json"
    optimization.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "train",
                        "partition": "optimizer_train",
                        "stratum": "numpy",
                        "metadata": {"perturbation_origin_id": 1},
                    }
                ]
            }
        )
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "cohort_role": "sealed_decision",
                "cases": [
                    {
                        "case_id": "decision",
                        "partition": "decision_test",
                        "stratum": "numpy",
                        "metadata": {"perturbation_origin_id": 1},
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="origin groups overlap"):
        _decision_manifest_cases(
            run_manifest={"benchmark_manifest": str(optimization)},
            decision_manifest_path=decision,
        )


@patch("compound.gepa_v2.subprocess.run")
def test_adapter_returns_correctness_dominant_objectives_and_caches(
    run: Mock, tmp_path: Path
) -> None:
    run.return_value = Mock(
        returncode=0,
        stdout=json.dumps({"passed": True, "feedback": "passed"}),
        stderr="",
    )
    provider = Mock()
    provider.complete.return_value = _response("result = values + 1")
    adapter = _adapter(tmp_path, provider)

    first = adapter.evaluate([_example()], dict(SEED_COMPONENTS), capture_traces=True)
    second = adapter.evaluate([_example()], dict(SEED_COMPONENTS), capture_traces=True)

    assert first.scores[0] >= 1.09
    assert first.objective_scores is not None
    assert first.objective_scores[0]["task_success"] == 1.0
    assert second.scores == first.scores
    assert _paired_comparison(first, second)["trial_transitions"]["both_pass"] == 1
    provider.complete.assert_called_once()
    run.assert_called_once()


def test_oversized_candidate_is_rejected_without_a_model_call(tmp_path: Path) -> None:
    provider = Mock()
    adapter = _adapter(tmp_path, provider)
    oversized = dict(SEED_COMPONENTS)
    oversized["response_contract"] = "word " * (MAX_CANDIDATE_WORDS + 1)

    batch = adapter.evaluate([_example()], oversized, capture_traces=True)

    assert batch.scores == [0.0]
    assert batch.trajectories is not None
    assert "prompt_budget_exceeded" in batch.trajectories[0].feedback
    provider.complete.assert_not_called()


def test_cache_only_adapter_refuses_a_missing_paid_completion(tmp_path: Path) -> None:
    provider = Mock()
    adapter = _adapter(tmp_path, provider)
    adapter.cache_only = True

    try:
        adapter.evaluate([_example()], dict(SEED_COMPONENTS))
    except FileNotFoundError as error:
        assert "required cached completion" in str(error)
    else:
        raise AssertionError("cache-only evaluation unexpectedly allowed a model call")
    provider.complete.assert_not_called()


@patch("compound.gepa_v2.subprocess.run")
def test_repeated_trials_use_distinct_completion_cache_entries(run: Mock, tmp_path: Path) -> None:
    run.return_value = Mock(
        returncode=0,
        stdout=json.dumps({"passed": True, "feedback": "passed"}),
        stderr="",
    )
    provider = Mock()
    provider.complete.return_value = _response("result = values + 1")
    adapter = _adapter(tmp_path, provider)

    adapter.evaluate([_example(0), _example(1)], dict(SEED_COMPONENTS))

    assert provider.complete.call_count == 2
    assert len(list((tmp_path / "cache" / "completions").glob("*.json"))) == 2


@patch("compound.gepa_v2.subprocess.run")
def test_flex_adapter_batches_missing_trials_and_separates_request_identity(
    run: Mock, tmp_path: Path
) -> None:
    run.return_value = Mock(
        returncode=0,
        stdout=json.dumps({"passed": True, "feedback": "passed"}),
        stderr="",
    )
    provider = Mock()
    flex_runner = Mock()

    def results(requests):
        return [
            {
                "output_text": "result = values + 1",
                "e2e_latency_ms": 10,
                "input_tokens": 100,
                "output_tokens": 20,
                "reasoning_tokens": 5,
                "finish_reason": "completed",
                "e2e_output_tps": 2.0,
                "estimated_cost_usd": 0.001,
                "response_id": f"resp_{index}",
                "requested_service_tier": "flex",
            }
            for index, _request in enumerate(requests)
        ]

    flex_runner.run_many.side_effect = results
    adapter = _adapter(tmp_path, provider, flex_runner=flex_runner)

    adapter.evaluate([_example(0), _example(1)], dict(SEED_COMPONENTS))

    flex_runner.run_many.assert_called_once()
    requests = flex_runner.run_many.call_args.args[0]
    assert requests[0].cache_discriminator is None
    assert requests[1].cache_discriminator is not None
    assert len(list((tmp_path / "cache" / "completions").glob("*.json"))) == 2
    provider.complete.assert_not_called()


@patch("compound.gepa_v2.subprocess.run")
def test_flex_adapter_deduplicates_repeated_trial_identity(
    run: Mock, tmp_path: Path
) -> None:
    run.return_value = Mock(
        returncode=0,
        stdout=json.dumps({"passed": True, "feedback": "passed"}),
        stderr="",
    )
    provider = Mock()
    flex_runner = Mock()
    flex_runner.run_many.return_value = [
        {
            "output_text": "result = values + 1",
            "e2e_latency_ms": 10,
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 0,
            "finish_reason": "completed",
            "e2e_output_tps": 2.0,
            "estimated_cost_usd": 0.001,
            "response_id": "resp_once",
            "requested_service_tier": "flex",
        }
    ]
    adapter = _adapter(tmp_path, provider, flex_runner=flex_runner)

    batch = adapter.evaluate([_example(), _example()], dict(SEED_COMPONENTS))

    assert len(batch.scores) == 2
    assert len(flex_runner.run_many.call_args.args[0]) == 1


@patch("compound.gepa_v2.subprocess.run")
def test_adapter_runs_through_pinned_gepa_multiobjective_flow(run: Mock, tmp_path: Path) -> None:
    run.return_value = Mock(
        returncode=0,
        stdout=json.dumps({"passed": True, "feedback": "passed"}),
        stderr="",
    )
    provider = Mock()
    provider.complete.return_value = _response("result = values + 1")
    adapter = _adapter(tmp_path, provider)
    dataset = [_example(0), _example(1)]

    result = gepa.optimize(
        seed_candidate=dict(SEED_COMPONENTS),
        trainset=NamespacedTrialLoader(dataset, "train"),
        valset=NamespacedTrialLoader(dataset, "validation"),
        adapter=adapter,
        reflection_lm=lambda _prompt: "```Use existing variables and assign result.```",
        frontier_type="hybrid",
        module_selector=lambda *_args: ["response_contract"],
        reflection_minibatch_size=2,
        perfect_score=1.10,
        max_metric_calls=8,
        run_dir=str(tmp_path / "gepa"),
        cache_evaluation=True,
        seed=7,
    )

    assert result.total_metric_calls is not None
    assert result.total_metric_calls <= 8
    assert result.val_aggregate_subscores is not None
    assert "task_success" in result.val_aggregate_subscores[0]
