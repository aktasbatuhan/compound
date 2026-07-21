from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Partition(StrEnum):
    OPTIMIZER_TRAIN = "optimizer_train"
    OPTIMIZER_VALIDATION = "optimizer_validation"
    DECISION_TEST = "decision_test"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    benchmark: str
    case_id: str
    task_key: str
    partition: Partition
    input: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.benchmark or not self.case_id or not self.task_key:
            raise ValueError("benchmark, case_id, and task_key are required")


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    components: dict[str, str]
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.components:
            raise ValueError("candidate_id and at least one component are required")


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Trajectory:
    benchmark: str
    case_id: str
    candidate_id: str
    provider: str
    model: str
    messages: tuple[dict[str, Any], ...]
    output: Any
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Evaluation:
    score: float
    feedback: str
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if not self.feedback.strip():
            raise ValueError("textual feedback is required for reflective optimization")


class BenchmarkAdapter(Protocol):
    name: str

    def load_cases(self) -> list[BenchmarkCase]: ...

    def run(self, case: BenchmarkCase, candidate: Candidate, model: Any) -> Trajectory: ...

    def evaluate(self, case: BenchmarkCase, trajectory: Trajectory) -> Evaluation: ...

