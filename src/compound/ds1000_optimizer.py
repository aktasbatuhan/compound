from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa import EvaluationBatch

from compound.adapters.ds1000 import build_test_program, load_ds1000_cases, postprocess_completion
from compound.budget import BudgetLedger
from compound.contracts import Partition
from compound.costs import TokenPrices, estimate_cost
from compound.providers import OpenAICompatibleProvider

DEFAULT_SEED_PROMPT = """Solve the data-science coding task in the user message.
Return only the Python completion that belongs inside the requested <code> block: no Markdown,
explanation, example-specific arithmetic, or test code. Derive a general solution that works for
unseen inputs and use the library requested by the task."""


@dataclass(frozen=True, slots=True)
class DS1000Example:
    case_id: str
    prompt: str
    code_context: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DS1000Trace:
    case_id: str
    prompt: str
    completion: str
    score: float
    feedback: str
    model_latency_ms: int = 0
    grader_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    e2e_output_tps: float | None = None
    cost_usd: float = 0.0


def load_optimizer_examples(
    source_path: str | Path,
    manifest_path: str | Path,
    *,
    partition: Partition,
    libraries: set[str] | None = None,
) -> list[DS1000Example]:
    records = load_ds1000_cases(source_path, manifest_path, partitions={partition})
    examples = [
        DS1000Example(
            case_id=record["case_id"],
            prompt=record["prompt"],
            code_context=record["code_context"],
            metadata=record["metadata"],
        )
        for record in records
        if libraries is None or record["metadata"]["library"] in libraries
    ]
    return sorted(examples, key=lambda example: example.case_id)


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return ""


def completion_fingerprint(
    *,
    case_id: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    reasoning_effort: str | None,
    trial_id: int = 0,
) -> str:
    payload = {
        "case_id": case_id,
        "model": model,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    # Preserve the original trial-0 fingerprints so existing paid completions
    # remain reusable after trial-aware evaluation is introduced.
    if trial_id:
        payload["trial_id"] = trial_id
    value = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def grade_fingerprint(*, completion: str, evaluator_image: str) -> str:
    value = json.dumps(
        {"completion": completion, "evaluator_image": evaluator_image}, sort_keys=True
    )
    return hashlib.sha256(value.encode()).hexdigest()


class CachedDS1000Adapter:
    propose_new_texts = None

    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        model: str,
        prices: TokenPrices,
        ledger: BudgetLedger,
        cache_dir: str | Path,
        docker_image: str,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.prices = prices
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        # Two-layer cache: model completions are keyed only on the generation
        # inputs (case, model, prompt, sampling), so re-grading under a corrected
        # evaluator image is a grade-cache miss but a completion-cache hit and
        # never triggers a paid model call.
        self.completion_cache_dir = self.cache_dir / "completions"
        self.grade_cache_dir = self.cache_dir / "grades"
        self.docker_image = docker_image
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def _completion_fingerprint(self, example: DS1000Example, system_prompt: str) -> str:
        return completion_fingerprint(
            case_id=example.case_id,
            model=self.model,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )

    def _grade_fingerprint(self, completion: str) -> str:
        return grade_fingerprint(
            completion=completion,
            evaluator_image=self.docker_image,
        )

    def _grade(self, example: DS1000Example, completion: str) -> tuple[float, str, int]:
        if not completion.strip():
            return 0.0, "Model returned no final completion.", 0
        program = build_test_program(
            {"code_context": example.code_context}, postprocess_completion(completion)
        )
        started = time.perf_counter()
        process = subprocess.run(
            ["docker", "run", "--rm", "-i", self.docker_image],
            input=json.dumps({"program": program}),
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        grader_latency_ms = round((time.perf_counter() - started) * 1000)
        if process.returncode != 0:
            return 0.0, f"Evaluator container failed: {process.stderr.strip()}", grader_latency_ms
        try:
            result = json.loads(process.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return (
                0.0,
                f"Evaluator returned invalid output: {process.stdout[-500:]}",
                grader_latency_ms,
            )
        passed = bool(result.get("passed"))
        feedback = str(result.get("feedback") or "grader did not provide feedback")
        return float(passed), feedback, grader_latency_ms

    def _model_completion(self, example: DS1000Example, system_prompt: str) -> dict[str, Any]:
        fingerprint = self._completion_fingerprint(example, system_prompt)
        cache_path = self.completion_cache_dir / f"{fingerprint}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        self.ledger.require_headroom(0.20)
        extra_body = None
        if self.reasoning_effort:
            extra_body = {"reasoning_effort": self.reasoning_effort}
        response = self.provider.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example.prompt},
            ],
            extra_body=extra_body,
            max_tokens=self.max_tokens,
            telemetry_context={
                "benchmark": "ds1000",
                "case_id": example.case_id,
                "call_type": "candidate",
                "fingerprint": fingerprint,
            },
        )
        completion = response_text(response.output)
        cost = estimate_cost(response.usage, self.prices)
        choices = response.output.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        e2e_output_tps = (
            response.usage.output_tokens / (response.latency_ms / 1000)
            if response.latency_ms > 0
            else None
        )
        record = {
            "completion": completion,
            "model_latency_ms": response.latency_ms,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "reasoning_tokens": response.usage.reasoning_tokens,
            "finish_reason": finish_reason,
            "e2e_output_tps": e2e_output_tps,
            "cost_usd": cost,
        }
        self.ledger.record(cost, label=f"ds1000-opt:{fingerprint}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record

    def _graded(self, example: DS1000Example, completion: str) -> tuple[float, str, int]:
        fingerprint = self._grade_fingerprint(completion)
        cache_path = self.grade_cache_dir / f"{fingerprint}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return cached["score"], cached["feedback"], cached["grader_latency_ms"]
        score, grader_feedback, grader_latency_ms = self._grade(example, completion)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "score": score,
                    "feedback": grader_feedback,
                    "grader_latency_ms": grader_latency_ms,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return score, grader_feedback, grader_latency_ms

    def _run_one(self, example: DS1000Example, system_prompt: str) -> DS1000Trace:
        completion_record = self._model_completion(example, system_prompt)
        completion = completion_record["completion"]
        score, grader_feedback, grader_latency_ms = self._graded(example, completion)
        feedback = (
            "PASS: the completion generalized to all hidden executable assertions."
            if score == 1.0
            else (
                f"FAIL: {grader_feedback.rstrip('.')}. "
                "Revise the instruction to prevent this failure mode."
            )
        )
        return DS1000Trace(
            case_id=example.case_id,
            prompt=example.prompt,
            completion=completion,
            score=score,
            feedback=feedback,
            model_latency_ms=completion_record["model_latency_ms"],
            grader_latency_ms=grader_latency_ms,
            input_tokens=completion_record["input_tokens"],
            output_tokens=completion_record["output_tokens"],
            reasoning_tokens=completion_record["reasoning_tokens"],
            finish_reason=completion_record["finish_reason"],
            e2e_output_tps=completion_record["e2e_output_tps"],
            cost_usd=completion_record["cost_usd"],
        )

    def evaluate(
        self,
        batch: list[DS1000Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DS1000Trace, dict[str, Any]]:
        system_prompt = candidate["system_prompt"]
        traces = [self._run_one(example, system_prompt) for example in batch]
        return EvaluationBatch(
            outputs=[
                {"case_id": trace.case_id, "completion": trace.completion}
                for trace in traces
            ],
            scores=[trace.score for trace in traces],
            trajectories=traces if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DS1000Trace, dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        del candidate
        if eval_batch.trajectories is None:
            raise ValueError("captured trajectories are required for reflection")
        records = [
            {
                "case_id": trace.case_id,
                "Inputs": {"task": trace.prompt},
                "Generated Outputs": trace.completion,
                "Feedback": trace.feedback,
                "score": trace.score,
            }
            for trace in eval_batch.trajectories
        ]
        return {component: records for component in components_to_update}


class MeteredReflectionLM:
    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        model: str,
        prices: TokenPrices,
        ledger: BudgetLedger,
        cache_dir: str | Path,
        max_tokens: int = 1400,
    ) -> None:
        self.provider = provider
        self.model = model
        self.prices = prices
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.max_tokens = max_tokens

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {"model": self.model, "messages": messages, "max_tokens": self.max_tokens},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"{fingerprint}.json"
        if cache_path.exists():
            return str(json.loads(cache_path.read_text())["text"])
        self.ledger.require_headroom(0.50)
        response = self.provider.complete(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            telemetry_context={
                "benchmark": "ds1000",
                "call_type": "reflection",
                "fingerprint": fingerprint,
            },
        )
        text = response_text(response.output)
        cost = estimate_cost(response.usage, self.prices)
        self.ledger.record(cost, label=f"reflection:{fingerprint}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"text": text}, indent=2) + "\n")
        return text
