from dataclasses import dataclass

import pytest

from compound.contracts import (
    BenchmarkCase,
    Candidate,
    Evaluation,
    Partition,
    Trajectory,
)
from compound.runner import DecisionSetAccessError, ExperimentRunner


@dataclass
class FakeAdapter:
    name: str = "fake"

    def load_cases(self):
        return []

    def run(self, case, candidate, model):
        return Trajectory(
            benchmark=case.benchmark,
            case_id=case.case_id,
            candidate_id=candidate.candidate_id,
            provider="fake",
            model="fake",
            messages=(),
            output="ok",
        )

    def evaluate(self, case, trajectory):
        return Evaluation(score=1.0, feedback="passed deterministic check")


def _case(partition: Partition) -> BenchmarkCase:
    return BenchmarkCase("fake", "1", "fake_task", partition, {"prompt": "hello"})


def test_optimizer_cannot_access_decision_set() -> None:
    runner = ExperimentRunner()
    with pytest.raises(DecisionSetAccessError):
        runner.run_case(
            FakeAdapter(),
            _case(Partition.DECISION_TEST),
            Candidate("c1", {"system_prompt": "x"}),
            object(),
        )


def test_final_evaluation_can_access_decision_set_explicitly() -> None:
    runner = ExperimentRunner(allow_decision_test=True)
    result = runner.run_case(
        FakeAdapter(),
        _case(Partition.DECISION_TEST),
        Candidate("c1", {"system_prompt": "x"}),
        object(),
    )
    assert result.evaluation.score == 1.0
