from __future__ import annotations

from dataclasses import dataclass

from compound.contracts import BenchmarkAdapter, BenchmarkCase, Candidate, Evaluation, Partition


class DecisionSetAccessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunResult:
    case: BenchmarkCase
    evaluation: Evaluation


class ExperimentRunner:
    def __init__(self, *, allow_decision_test: bool = False) -> None:
        self.allow_decision_test = allow_decision_test

    def run_case(
        self,
        adapter: BenchmarkAdapter,
        case: BenchmarkCase,
        candidate: Candidate,
        model: object,
    ) -> RunResult:
        if case.partition is Partition.DECISION_TEST and not self.allow_decision_test:
            raise DecisionSetAccessError(
                "decision-test access is disabled during optimization and selection"
            )
        trajectory = adapter.run(case, candidate, model)
        return RunResult(case=case, evaluation=adapter.evaluate(case, trajectory))

