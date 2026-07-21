from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa import EvaluationBatch

from compound.adapters.ds1000 import build_test_program, postprocess_completion
from compound.budget import BudgetExceededError, BudgetLedger
from compound.costs import TokenPrices, estimate_cost
from compound.ds1000_optimizer import (
    DS1000Example,
    completion_fingerprint,
    grade_fingerprint,
    response_text,
)
from compound.flex import CachedFlexRunner, FlexRequest
from compound.providers import OpenAICompatibleProvider

SEED_COMPONENTS = {
    "response_contract": (
        "Complete the supplied Python scaffold. Return only nonempty executable Python for the "
        "BEGIN SOLUTION block, without Markdown, explanations, tests, or recreated example data. "
        "Use the existing imports and variables. Assign or return the exact requested result. "
        "Produce a general solution for unseen inputs and preserve the required type and shape."
    ),
    "numpy_strategy": (
        "For NumPy tasks, reason from axes, broadcasting, shapes, and dtypes. Prefer vectorized "
        "operations and derive dimensions from the inputs rather than displayed examples."
    ),
    "pandas_strategy": (
        "For pandas tasks, preserve indexes, column order, labels, missing values, and requested "
        "dtypes. Prefer vectorized Series and DataFrame operations over example-specific code."
    ),
}

MAX_CANDIDATE_WORDS = 600
PERFECT_COMPOSITE_SCORE = 1.10


@dataclass(frozen=True, slots=True)
class TrialExample:
    base: DS1000Example
    trial_id: int = 0

    @property
    def case_id(self) -> str:
        return self.base.case_id

    @property
    def library(self) -> str:
        return str(self.base.metadata["library"])


class NamespacedTrialLoader:
    """Give train and validation examples disjoint GEPA cache identities."""

    def __init__(self, examples: Sequence[TrialExample], namespace: str) -> None:
        if not namespace or ":" in namespace:
            raise ValueError("namespace must be a nonempty token without ':'")
        self.examples = list(examples)
        self.namespace = namespace

    def all_ids(self) -> list[str]:
        return [f"{self.namespace}:{index}" for index in range(len(self.examples))]

    def fetch(self, ids: Sequence[str]) -> list[TrialExample]:
        prefix = f"{self.namespace}:"
        if any(not data_id.startswith(prefix) for data_id in ids):
            raise KeyError(f"data id does not belong to {self.namespace!r}")
        return [self.examples[int(data_id.removeprefix(prefix))] for data_id in ids]

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(frozen=True, slots=True)
class V2Trace:
    case_id: str
    trial_id: int
    library: str
    prompt: str
    completion: str
    feedback: str
    metrics: dict[str, float]
    composite_score: float
    model_latency_ms: int
    grader_latency_ms: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    finish_reason: str | None
    e2e_output_tps: float | None
    cost_usd: float


@dataclass(slots=True)
class ExperimentCap:
    ledger: BudgetLedger
    start_spend_usd: float
    limit_usd: float

    @property
    def spent_usd(self) -> float:
        return self.ledger.spent_usd - self.start_spend_usd

    def require_headroom(self, estimated_max_usd: float) -> None:
        self.ledger.require_headroom(estimated_max_usd)
        if self.spent_usd + estimated_max_usd > self.limit_usd:
            raise BudgetExceededError(
                f"estimated call would exceed ${self.limit_usd:.2f} experiment cap"
            )


def expand_trials(examples: list[DS1000Example], trials: int) -> list[TrialExample]:
    if trials < 1:
        raise ValueError("trials must be positive")
    return [TrialExample(example, trial) for example in examples for trial in range(trials)]


def candidate_word_count(candidate: Mapping[str, str]) -> int:
    return sum(len(text.split()) for text in candidate.values())


def compose_system_prompt(candidate: Mapping[str, str], library: str) -> str:
    strategy_key = "numpy_strategy" if library.lower() == "numpy" else "pandas_strategy"
    return f"{candidate['response_contract']}\n\n{candidate[strategy_key]}"


def _syntax_valid(completion: str) -> tuple[float, str | None]:
    if not completion.strip():
        return 0.0, "empty_completion"
    try:
        ast.parse(postprocess_completion(completion))
    except SyntaxError as error:
        return 0.0, f"syntax_error: {error.msg} at line {error.lineno}"
    return 1.0, None


def _target_satisfied(prompt: str, completion: str) -> float:
    cleaned = postprocess_completion(completion)
    before_solution = prompt.split("BEGIN SOLUTION", 1)[0]
    targets = re.findall(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*[^=].*$", before_solution)
    target = targets[-1] if targets else None
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return 0.0
    if any(isinstance(node, ast.Return) for node in ast.walk(tree)):
        return 1.0
    if target is None:
        return 1.0
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    return float(target in assigned)


def _avoids_scaffold_recreation(prompt: str, completion: str) -> float:
    setup_match = re.search(r"<code>(.*?)</code>", prompt, flags=re.DOTALL)
    if not setup_match:
        return 1.0
    setup_assignments = [
        line.strip()
        for line in setup_match.group(1).splitlines()
        if re.match(r"^\s*[A-Za-z_]\w*\s*=", line)
    ]
    normalized_completion = "\n".join(line.strip() for line in completion.splitlines())
    return float(not any(line in normalized_completion for line in setup_assignments))


def _failure_category(metrics: Mapping[str, float], grader_feedback: str) -> str:
    if not metrics["completion_present"]:
        return "empty_completion"
    if not metrics["syntax_valid"]:
        return "syntax_error"
    if not metrics["format_valid"]:
        return "format_error"
    if not metrics["target_satisfied"]:
        return "missing_target"
    if "timeout" in grader_feedback.lower():
        return "timeout"
    if "AssertionError" in grader_feedback:
        return "assertion_failure"
    if metrics["task_success"] < 1.0:
        return "execution_failure"
    return "passed"


class MeteredFailureCritic:
    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        model: str,
        prices: TokenPrices,
        ledger: BudgetLedger,
        experiment_cap: ExperimentCap,
        cache_dir: str | Path,
        teacher_traces: Mapping[str, Mapping[str, str]] | None = None,
        max_tokens: int = 400,
    ) -> None:
        self.provider = provider
        self.model = model
        self.prices = prices
        self.ledger = ledger
        self.experiment_cap = experiment_cap
        self.cache_dir = Path(cache_dir)
        self.teacher_traces = dict(teacher_traces or {})
        self.max_tokens = max_tokens

    def __call__(self, trace: V2Trace) -> str:
        payload = {
            "task": trace.prompt,
            "completion": trace.completion,
            "feedback": trace.feedback,
        }
        teacher = self.teacher_traces.get(trace.case_id)
        if teacher is not None:
            payload["teacher_model"] = teacher["model"]
            payload["teacher_completion"] = teacher["completion"]
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{fingerprint}.json"
        if path.exists():
            return str(json.loads(path.read_text())["diagnosis"])
        self.experiment_cap.require_headroom(0.20)
        response = self.provider.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Diagnose failed data-science code. When a passing teacher completion is "
                        "provided, contrast its approach with the failed completion. Give the root "
                        "cause and one reusable prompt rule in at most 120 words. Never repeat a "
                        "full solution, case-specific values, or variable names from the teacher."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            max_tokens=self.max_tokens,
            telemetry_context={
                "benchmark": "ds1000",
                "case_id": trace.case_id,
                "trial_id": trace.trial_id,
                "call_type": "failure_critic",
                "teacher_model": teacher["model"] if teacher is not None else None,
                "fingerprint": fingerprint,
            },
        )
        diagnosis = response_text(response.output)
        cost = estimate_cost(response.usage, self.prices)
        self.ledger.record(cost, label=f"critic:{fingerprint}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"diagnosis": diagnosis}, indent=2) + "\n")
        return diagnosis


class MeteredV2ReflectionLM:
    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        model: str,
        prices: TokenPrices,
        ledger: BudgetLedger,
        experiment_cap: ExperimentCap,
        cache_dir: str | Path,
        max_tokens: int = 1200,
    ) -> None:
        self.provider = provider
        self.model = model
        self.prices = prices
        self.ledger = ledger
        self.experiment_cap = experiment_cap
        self.cache_dir = Path(cache_dir)
        self.max_tokens = max_tokens

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        fingerprint = hashlib.sha256(
            json.dumps(
                {"model": self.model, "messages": messages, "max_tokens": self.max_tokens},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        path = self.cache_dir / f"{fingerprint}.json"
        if path.exists():
            return str(json.loads(path.read_text())["text"])
        self.experiment_cap.require_headroom(0.40)
        response = self.provider.complete(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            telemetry_context={
                "benchmark": "ds1000",
                "call_type": "reflection_v2",
                "fingerprint": fingerprint,
            },
        )
        text = response_text(response.output)
        cost = estimate_cost(response.usage, self.prices)
        self.ledger.record(cost, label=f"reflection-v2:{fingerprint}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}, indent=2) + "\n")
        return text


class DS1000GEPAAdapterV2:
    propose_new_texts = None

    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        model: str,
        prices: TokenPrices,
        ledger: BudgetLedger,
        experiment_cap: ExperimentCap,
        cache_dir: str | Path,
        docker_image: str,
        max_tokens: int,
        reasoning_effort: str | None,
        critic: MeteredFailureCritic | None,
        cache_only: bool = False,
        telemetry_call_type: str = "candidate_v2",
        flex_runner: CachedFlexRunner | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.prices = prices
        self.ledger = ledger
        self.experiment_cap = experiment_cap
        self.cache_dir = Path(cache_dir)
        self.completion_cache_dir = self.cache_dir / "completions"
        self.grade_cache_dir = self.cache_dir / "grades"
        self.docker_image = docker_image
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.critic = critic
        self.cache_only = cache_only
        self.telemetry_call_type = telemetry_call_type
        self.flex_runner = flex_runner

    def _completion_identity(self, example: TrialExample, system_prompt: str) -> tuple[str, Path]:
        fingerprint = completion_fingerprint(
            case_id=example.case_id,
            model=self.model,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            trial_id=example.trial_id,
        )
        return fingerprint, self.completion_cache_dir / f"{fingerprint}.json"

    def _prefetch_flex(self, entries: list[tuple[TrialExample, str]]) -> None:
        if self.flex_runner is None:
            return
        pending: list[tuple[TrialExample, str, str, Path]] = []
        requests: list[FlexRequest] = []
        pending_fingerprints: set[str] = set()
        for example, system_prompt in entries:
            fingerprint, path = self._completion_identity(example, system_prompt)
            if path.exists() or fingerprint in pending_fingerprints:
                continue
            if self.cache_only:
                raise FileNotFoundError(f"required cached completion is missing: {fingerprint}")
            pending_fingerprints.add(fingerprint)
            pending.append((example, system_prompt, fingerprint, path))
            requests.append(
                FlexRequest(
                    model=self.model,
                    input=example.base.prompt,
                    instructions=system_prompt,
                    max_output_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    # Trial zero deliberately reuses the original request identity;
                    # repeated trials must be independent paid generations.
                    cache_discriminator=(fingerprint if example.trial_id else None),
                    telemetry_context={
                        "benchmark": "ds1000",
                        "case_id": example.case_id,
                        "trial_id": example.trial_id,
                        "call_type": self.telemetry_call_type,
                        "completion_fingerprint": fingerprint,
                    },
                )
            )
        if not requests:
            return
        responses = self.flex_runner.run_many(requests)
        for (_, _, _fingerprint, path), response in zip(pending, responses, strict=True):
            record = {
                "completion": response["output_text"],
                "model_latency_ms": response["e2e_latency_ms"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "reasoning_tokens": response["reasoning_tokens"],
                "finish_reason": response["finish_reason"],
                "e2e_output_tps": response["e2e_output_tps"],
                "cost_usd": response["estimated_cost_usd"],
                "response_id": response["response_id"],
                "api_surface": "responses",
                "service_tier": response["requested_service_tier"],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    def _completion(self, example: TrialExample, system_prompt: str) -> dict[str, Any]:
        fingerprint, path = self._completion_identity(example, system_prompt)
        if path.exists():
            return json.loads(path.read_text())
        if self.cache_only:
            raise FileNotFoundError(f"required cached completion is missing: {fingerprint}")
        if self.flex_runner is not None:
            self._prefetch_flex([(example, system_prompt)])
            return json.loads(path.read_text())
        self.experiment_cap.require_headroom(0.30)
        extra_body = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None
        response = self.provider.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example.base.prompt},
            ],
            extra_body=extra_body,
            max_tokens=self.max_tokens,
            telemetry_context={
                "benchmark": "ds1000",
                "case_id": example.case_id,
                "trial_id": example.trial_id,
                "call_type": self.telemetry_call_type,
                "fingerprint": fingerprint,
            },
        )
        choices = response.output.get("choices") or []
        record = {
            "completion": response_text(response.output),
            "model_latency_ms": response.latency_ms,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "reasoning_tokens": response.usage.reasoning_tokens,
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "e2e_output_tps": (
                response.usage.output_tokens / (response.latency_ms / 1000)
                if response.latency_ms > 0
                else None
            ),
            "cost_usd": estimate_cost(response.usage, self.prices),
        }
        self.ledger.record(record["cost_usd"], label=f"ds1000-v2:{fingerprint}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record

    def _docker_grade(self, example: TrialExample, completion: str) -> dict[str, Any]:
        fingerprint = grade_fingerprint(completion=completion, evaluator_image=self.docker_image)
        path = self.grade_cache_dir / f"{fingerprint}.json"
        if path.exists():
            return json.loads(path.read_text())
        if not completion.strip():
            grade = {
                "score": 0.0,
                "feedback": "Model returned no final completion.",
                "grader_latency_ms": 0,
            }
        else:
            cleaned = postprocess_completion(completion)
            try:
                ast.parse(cleaned)
            except SyntaxError as error:
                grade = {
                    "score": 0.0,
                    "feedback": f"SyntaxError: {error.msg} at line {error.lineno}",
                    "grader_latency_ms": 0,
                }
            else:
                program = build_test_program({"code_context": example.base.code_context}, cleaned)
                started = time.perf_counter()
                process = subprocess.run(
                    ["docker", "run", "--rm", "-i", self.docker_image],
                    input=json.dumps({"program": program}),
                    text=True,
                    capture_output=True,
                    timeout=90,
                    check=False,
                )
                latency_ms = round((time.perf_counter() - started) * 1000)
                try:
                    result = json.loads(process.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    result = {
                        "passed": False,
                        "feedback": process.stderr.strip() or "invalid evaluator output",
                    }
                grade = {
                    "score": float(bool(result.get("passed"))),
                    "feedback": str(result.get("feedback") or "grader provided no feedback"),
                    "grader_latency_ms": latency_ms,
                }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(grade, indent=2, sort_keys=True) + "\n")
        return grade

    def _run_one(self, example: TrialExample, candidate: Mapping[str, str]) -> V2Trace:
        word_count = candidate_word_count(candidate)
        if word_count > MAX_CANDIDATE_WORDS:
            metrics = {
                "task_success": 0.0,
                "completion_present": 0.0,
                "syntax_valid": 0.0,
                "format_valid": 0.0,
                "target_satisfied": 0.0,
                "avoids_scaffold_recreation": 0.0,
                "token_efficiency": 0.0,
                "prompt_compactness": 0.0,
            }
            return V2Trace(
                case_id=example.case_id,
                trial_id=example.trial_id,
                library=example.library,
                prompt=example.base.prompt,
                completion="",
                feedback=f"prompt_budget_exceeded: {word_count}>{MAX_CANDIDATE_WORDS} words",
                metrics=metrics,
                composite_score=0.0,
                model_latency_ms=0,
                grader_latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                finish_reason=None,
                e2e_output_tps=None,
                cost_usd=0.0,
            )

        system_prompt = compose_system_prompt(candidate, example.library)
        completion_record = self._completion(example, system_prompt)
        completion = str(completion_record["completion"])
        grade = self._docker_grade(example, completion)
        syntax_valid, syntax_feedback = _syntax_valid(completion)
        metrics = {
            "task_success": float(grade["score"]),
            "completion_present": float(bool(completion.strip())),
            "syntax_valid": syntax_valid,
            "format_valid": float("```" not in completion and "<code>" not in completion),
            "target_satisfied": _target_satisfied(example.base.prompt, completion),
            "avoids_scaffold_recreation": _avoids_scaffold_recreation(
                example.base.prompt, completion
            ),
            "token_efficiency": (
                min(1.0, 256 / max(256, int(completion_record["output_tokens"])))
                if completion.strip()
                else 0.0
            ),
            "prompt_compactness": max(0.0, 1.0 - word_count / MAX_CANDIDATE_WORDS),
        }
        secondary = (
            0.03 * metrics["syntax_valid"]
            + 0.02 * metrics["completion_present"]
            + 0.01 * metrics["format_valid"]
            + 0.02 * metrics["target_satisfied"]
            + 0.01 * metrics["avoids_scaffold_recreation"]
            + 0.005 * metrics["token_efficiency"]
            + 0.005 * metrics["prompt_compactness"]
        )
        composite_score = metrics["task_success"] + secondary
        category = _failure_category(metrics, str(grade["feedback"]))
        feedback = json.dumps(
            {
                "category": category,
                "grader": grade["feedback"],
                "syntax": syntax_feedback,
                "metrics": metrics,
            },
            sort_keys=True,
        )
        return V2Trace(
            case_id=example.case_id,
            trial_id=example.trial_id,
            library=example.library,
            prompt=example.base.prompt,
            completion=completion,
            feedback=feedback,
            metrics=metrics,
            composite_score=composite_score,
            model_latency_ms=int(completion_record["model_latency_ms"]),
            grader_latency_ms=int(grade["grader_latency_ms"]),
            input_tokens=int(completion_record["input_tokens"]),
            output_tokens=int(completion_record["output_tokens"]),
            reasoning_tokens=int(completion_record["reasoning_tokens"]),
            finish_reason=completion_record["finish_reason"],
            e2e_output_tps=completion_record["e2e_output_tps"],
            cost_usd=float(completion_record["cost_usd"]),
        )

    def evaluate(
        self,
        batch: list[TrialExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[V2Trace, dict[str, Any]]:
        if self.flex_runner is not None and candidate_word_count(candidate) <= MAX_CANDIDATE_WORDS:
            self._prefetch_flex(
                [(example, compose_system_prompt(candidate, example.library)) for example in batch]
            )
        traces = [self._run_one(example, candidate) for example in batch]
        return EvaluationBatch(
            outputs=[
                {
                    "case_id": trace.case_id,
                    "trial_id": trace.trial_id,
                    "completion": trace.completion,
                }
                for trace in traces
            ],
            scores=[trace.composite_score for trace in traces],
            trajectories=traces if capture_traces else None,
            objective_scores=[trace.metrics for trace in traces],
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[V2Trace, dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        del candidate
        traces = eval_batch.trajectories
        if traces is None:
            raise ValueError("captured trajectories are required for reflection")
        result: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            library = None
            if component == "numpy_strategy":
                library = "numpy"
            elif component == "pandas_strategy":
                library = "pandas"
            relevant = [
                trace for trace in traces if library is None or trace.library.lower() == library
            ]
            records = []
            for trace in relevant:
                diagnosis = (
                    self.critic(trace)
                    if self.critic and trace.metrics["task_success"] < 1.0
                    else ""
                )
                records.append(
                    {
                        "case_id": trace.case_id,
                        "trial_id": trace.trial_id,
                        "Inputs": {"task": trace.prompt},
                        "Generated Outputs": trace.completion,
                        "Feedback": trace.feedback,
                        "Frontier diagnosis": diagnosis,
                    }
                )
            if records:
                result[component] = records
        return result


def select_relevant_components(
    state: Any,
    trajectories: list[V2Trace],
    subsample_scores: list[float],
    candidate_idx: int,
    candidate: dict[str, str],
) -> list[str]:
    del state, subsample_scores, candidate_idx, candidate
    components: list[str] = []
    categories = {
        json.loads(trace.feedback).get("category")
        for trace in trajectories
        if trace.feedback.startswith("{")
    }
    if categories & {"empty_completion", "syntax_error", "format_error", "missing_target"}:
        components.append("response_contract")
    failed_libraries = {
        trace.library.lower() for trace in trajectories if trace.metrics["task_success"] < 1.0
    }
    if "numpy" in failed_libraries:
        components.append("numpy_strategy")
    if "pandas" in failed_libraries:
        components.append("pandas_strategy")
    return components or ["response_contract"]


def select_library_components(
    state: Any,
    trajectories: list[V2Trace],
    subsample_scores: list[float],
    candidate_idx: int,
    candidate: dict[str, str],
) -> list[str]:
    """Keep the shared output contract fixed and mutate only task-family guidance."""
    del state, subsample_scores, candidate_idx, candidate
    failed_libraries = {
        trace.library.lower() for trace in trajectories if trace.metrics["task_success"] < 1.0
    }
    components = [
        component
        for library, component in (
            ("numpy", "numpy_strategy"),
            ("pandas", "pandas_strategy"),
        )
        if library in failed_libraries
    ]
    if components:
        return components
    present_libraries = {trace.library.lower() for trace in trajectories}
    return [
        component
        for library, component in (
            ("numpy", "numpy_strategy"),
            ("pandas", "pandas_strategy"),
        )
        if library in present_libraries
    ]
