"""Generic product-task prompt optimization with the real GEPA library.

This is the Python half of Step 7 (docs/optimization-v1.md). It is invoked as a
subprocess by the TS `compound optimize` command with a job JSON, drives
`gepa.optimize` over a task's optimizer_train / optimizer_validation cases with a
single mutable `system_prompt` component, and returns the optimized prompt with
before/after validation scores.

Grading is NOT reimplemented here: the adapter shells back to the one TS grader
(`compound grade-batch`), so `@compound/assertions` stays the single source of
truth. GEPA proposes prompts; TS scores them.

Job JSON (written by the TS orchestrator):
    { task_key, candidate_model, candidate:{base_url,api_key_env},
      reflection_model, reflection:{base_url,api_key_env},
      seed_prompt, trainset:[{case_id, messages, tools}], valset:[...],
      max_metric_calls, reflection_minibatch_size, grade_cmd:[...], run_dir, output }
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gepa
from gepa import EvaluationBatch

from compound.providers import OpenAICompatibleProvider


def _assistant_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract an assistant message in the CONTRACT shape the TS grader reads."""
    choices = raw.get("choices") or []
    msg = (choices[0].get("message") if choices else {}) or {}
    tool_calls = []
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if args_raw else {}
            if not isinstance(args, dict):
                args = {"_raw": args_raw}
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": args_raw or ""}
        tool_calls.append({"id": call.get("id", ""), "name": fn.get("name", ""), "arguments": args})
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _compose(system_prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the system message with the candidate prompt; keep the rest."""
    rest = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}, *rest]


class CompoundAdapter:
    """A generic GEPA adapter over product cases, grading via the TS bridge."""

    # None → GEPA uses its default reflection-LM proposal (make_reflective_dataset
    # + reflection_prompt_template), rather than an adapter-supplied proposer.
    propose_new_texts = None

    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        model: str,
        task_key: str,
        grade_cmd: list[str],
        user_field: str = "request",
    ) -> None:
        self.provider = provider
        self.model = model
        self.task_key = task_key
        self.grade_cmd = grade_cmd
        self.user_field = user_field

    def _grade(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        payload = json.dumps({"task_key": self.task_key, "items": items})
        proc = subprocess.run(
            self.grade_cmd, input=payload, capture_output=True, text=True, check=True
        )
        result = json.loads(proc.stdout)
        return {row["case_id"]: row for row in result["items"]}

    def evaluate(
        self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        system_prompt = candidate["system_prompt"]
        outputs: list[dict[str, Any]] = []
        for case in batch:
            response = self.provider.complete(
                model=self.model,
                messages=_compose(system_prompt, case["messages"]),
                tools=case.get("tools") or None,
                max_tokens=1024,
            )
            message = _assistant_message(response.output)
            outputs.append({"case_id": case["case_id"], "output": message})

        grades = self._grade(outputs)
        scores = [float(grades[o["case_id"]]["score"]) for o in outputs]

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "case_id": batch[i]["case_id"],
                    "request": _user_text(batch[i]),
                    "output": outputs[i]["output"],
                    "feedback": grades[outputs[i]["case_id"]]["feedback"],
                    "score": scores[i],
                }
                for i in range(len(batch))
            ]
        return EvaluationBatch(
            outputs=[o["output"] for o in outputs], scores=scores, trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        del candidate
        traces = eval_batch.trajectories or []
        records = [
            {
                "Inputs": {"request": trace["request"]},
                "Generated Outputs": json.dumps(trace["output"]),
                "Feedback": trace["feedback"] or ("passed" if trace["score"] >= 1 else "failed"),
            }
            for trace in traces
        ]
        return {component: records for component in components_to_update}


def _user_text(case: dict[str, Any]) -> str:
    for message in reversed(case.get("messages", [])):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


class ReflectionLM:
    """Minimal GEPA reflection LM: rewrite a prompt component from the evidence."""

    def __init__(self, *, provider: OpenAICompatibleProvider, model: str, max_calls: int) -> None:
        self.provider = provider
        self.model = model
        self.max_calls = max_calls
        self.calls = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if self.calls >= self.max_calls:
            raise RuntimeError("reflection budget exhausted")
        self.calls += 1
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        response = self.provider.complete(model=self.model, messages=messages, max_tokens=1200)
        return _assistant_message(response.output).get("content") or ""


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run(job: dict[str, Any]) -> dict[str, Any]:
    candidate_provider = OpenAICompatibleProvider(
        name="candidate",
        base_url=job["candidate"]["base_url"],
        api_key_env=job["candidate"]["api_key_env"],
    )
    reflection_provider = OpenAICompatibleProvider(
        name="reflection",
        base_url=job["reflection"]["base_url"],
        api_key_env=job["reflection"]["api_key_env"],
    )
    adapter = CompoundAdapter(
        provider=candidate_provider,
        model=job["candidate_model"],
        task_key=job["task_key"],
        grade_cmd=job["grade_cmd"],
    )
    reflection = ReflectionLM(
        provider=reflection_provider,
        model=job["reflection_model"],
        max_calls=int(job.get("max_metric_calls", 30)),
    )
    seed = {"system_prompt": job["seed_prompt"]}
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    template = (
        "You improve the SYSTEM PROMPT of an assistant for a task. Using the evidence "
        "(inputs, the assistant's outputs, and grader feedback on what failed), rewrite the "
        "prompt so it fixes the failures. Keep only reusable instructions — never hard-code a "
        "specific case's answer. Replace obsolete rules rather than appending. Under 150 words.\n\n"
        "CURRENT SYSTEM PROMPT:\n<curr_param>\n\nEVIDENCE:\n<side_info>\n\n"
        "Return only the complete replacement system prompt."
    )
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=job["trainset"],
        valset=job["valset"],
        adapter=adapter,
        reflection_lm=reflection,
        reflection_minibatch_size=int(job.get("reflection_minibatch_size", 3)),
        reflection_prompt_template={"system_prompt": template},
        max_metric_calls=int(job["max_metric_calls"]),
        run_dir=str(run_dir),
        seed=int(job.get("seed", 0)),
        display_progress_bar=False,
    )

    best = dict(result.best_candidate)
    before = adapter.evaluate(job["valset"], seed, capture_traces=False)
    after = adapter.evaluate(job["valset"], best, capture_traces=False)
    return {
        "task_key": job["task_key"],
        "candidate_model": job["candidate_model"],
        "seed_prompt": job["seed_prompt"],
        "optimized_prompt": best["system_prompt"],
        "before_val_score": _mean(before.scores),
        "after_val_score": _mean(after.scores),
        "val_cases": len(job["valset"]),
        "reflection_calls": reflection.calls,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m compound.optimize_product <job.json>", file=sys.stderr)
        raise SystemExit(2)
    job = json.loads(Path(sys.argv[1]).read_text())
    result = run(job)
    Path(job["output"]).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "optimized_prompt"}))


if __name__ == "__main__":
    main()
