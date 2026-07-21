from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa.strategies.instruction_proposal import InstructionProposalSignature

from compound.adapters.ds1000 import build_test_program, postprocess_completion
from compound.contracts import Partition
from compound.ds1000_optimizer import (
    DS1000Example,
    completion_fingerprint,
    grade_fingerprint,
    load_optimizer_examples,
)


@dataclass(frozen=True, slots=True)
class MigrationResult:
    migrated_completions: int
    migrated_grades: int
    archived_legacy_files: int
    unmatched_legacy_files: tuple[str, ...]


def _legacy_fingerprints(
    *,
    case_id: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    reasoning_effort: str | None,
    evaluator_images: set[str],
) -> dict[str, str]:
    common = {
        "case_id": case_id,
        "model": model,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    variants = {
        "ds1000-official-context-v1": {
            **common,
            "evaluator": "ds1000-official-context-v1",
        }
    }
    for image in evaluator_images:
        variants[image] = {**common, "evaluator_image": image}
    return {
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(): evaluator
        for evaluator, payload in variants.items()
    }


def _candidate_prompts(runs: list[Path], reflection_cache_dir: Path) -> set[str]:
    prompts: set[str] = set()
    for run in runs:
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            prompts.add(json.loads(manifest_path.read_text())["seed_prompt"])
        result_path = run / "result.json"
        if result_path.exists():
            for candidate in json.loads(result_path.read_text()).get("candidates", []):
                prompt = candidate.get("system_prompt")
                if prompt:
                    prompts.add(prompt)
    if reflection_cache_dir.exists():
        for path in reflection_cache_dir.glob("*.json"):
            text = json.loads(path.read_text()).get("text")
            if text:
                prompt = InstructionProposalSignature.output_extractor(text)["new_instruction"]
                if prompt:
                    prompts.add(prompt)
    return prompts


def _raw_grader_feedback(trace: dict[str, Any]) -> str:
    if float(trace.get("score", 0.0)) == 1.0:
        return "all executable assertions passed"
    feedback = str(trace.get("feedback", "grader did not provide feedback"))
    match = re.fullmatch(
        r"FAIL: (.*)\. Revise the instruction to prevent this failure mode\.",
        feedback,
        flags=re.DOTALL,
    )
    return match.group(1) if match else feedback


def migrate_legacy_ds1000_cache(
    *,
    cache_dir: str | Path,
    runs_dir: str | Path,
    reflection_cache_dir: str | Path,
    legacy_evaluator_image: str,
    archive: bool = False,
) -> MigrationResult:
    """Split legacy trace files without making provider or grader calls."""
    cache = Path(cache_dir)
    runs = sorted(path for path in Path(runs_dir).glob("ds1000-*") if path.is_dir())
    prompts = _candidate_prompts(runs, Path(reflection_cache_dir))
    completion_dir = cache / "completions"
    grade_dir = cache / "grades"
    legacy_files = {path.stem: path for path in cache.glob("*.json")}
    matched: dict[str, tuple[dict[str, Any], str, str]] = {}

    for run in runs:
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        model = manifest["candidate_model"]
        max_tokens = int(manifest.get("candidate_max_tokens", 1024))
        reasoning_effort = manifest.get("reasoning_effort")
        evaluator_images = {
            legacy_evaluator_image,
            manifest.get("evaluator_image", legacy_evaluator_image),
        }
        case_ids = set(manifest.get("train_case_ids", []))
        case_ids.update(manifest.get("validation_case_ids", []))
        decision_path = run / "decision-test.json"
        if decision_path.exists():
            case_ids.update(json.loads(decision_path.read_text()).get("case_ids", []))

        for case_id in case_ids:
            for prompt in prompts:
                variants = _legacy_fingerprints(
                    case_id=case_id,
                    model=model,
                    system_prompt=prompt,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    evaluator_images=evaluator_images,
                )
                for legacy_hash, evaluator in variants.items():
                    path = legacy_files.get(legacy_hash)
                    if path is None:
                        continue
                    new_hash = completion_fingerprint(
                        case_id=case_id,
                        model=model,
                        system_prompt=prompt,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                    )
                    matched[legacy_hash] = (json.loads(path.read_text()), new_hash, evaluator)

    completion_dir.mkdir(parents=True, exist_ok=True)
    grade_dir.mkdir(parents=True, exist_ok=True)
    migrated_completions = 0
    migrated_grades = 0
    archived = 0
    archive_dir = cache / "legacy"
    for legacy_hash, (trace, new_hash, evaluator) in matched.items():
        completion_path = completion_dir / f"{new_hash}.json"
        completion_record = {
            key: trace.get(key, default)
            for key, default in {
                "completion": "",
                "model_latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "finish_reason": None,
                "e2e_output_tps": None,
                "cost_usd": 0.0,
            }.items()
        }
        if not completion_path.exists():
            completion_path.write_text(
                json.dumps(completion_record, indent=2, sort_keys=True) + "\n"
            )
            migrated_completions += 1

        image = legacy_evaluator_image if evaluator == "ds1000-official-context-v1" else evaluator
        grade_hash = grade_fingerprint(
            completion=completion_record["completion"], evaluator_image=image
        )
        grade_path = grade_dir / f"{grade_hash}.json"
        if not grade_path.exists():
            grade_path.write_text(
                json.dumps(
                    {
                        "score": float(trace.get("score", 0.0)),
                        "feedback": _raw_grader_feedback(trace),
                        "grader_latency_ms": int(trace.get("grader_latency_ms", 0) or 0),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            migrated_grades += 1

        if archive:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(
                str(legacy_files[legacy_hash]),
                archive_dir / legacy_files[legacy_hash].name,
            )
            archived += 1

    unmatched = tuple(sorted(f"{key}.json" for key in set(legacy_files) - set(matched)))
    return MigrationResult(
        migrated_completions=migrated_completions,
        migrated_grades=migrated_grades,
        archived_legacy_files=archived,
        unmatched_legacy_files=unmatched,
    )


def _execute_grade(
    example: DS1000Example, completion: str, evaluator_image: str
) -> dict[str, Any]:
    if not completion.strip():
        return {
            "score": 0.0,
            "feedback": "Model returned no final completion.",
            "grader_latency_ms": 0,
        }
    program = build_test_program(
        {"code_context": example.code_context}, postprocess_completion(completion)
    )
    started = time.perf_counter()
    process = subprocess.run(
        ["docker", "run", "--rm", "-i", evaluator_image],
        input=json.dumps({"program": program}),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    if process.returncode != 0:
        return {
            "score": 0.0,
            "feedback": f"Evaluator container failed: {process.stderr.strip()}",
            "grader_latency_ms": latency_ms,
        }
    try:
        result = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result = {
            "passed": False,
            "feedback": f"Evaluator returned invalid output: {process.stdout[-500:]}",
        }
    return {
        "score": float(bool(result.get("passed"))),
        "feedback": str(result.get("feedback") or "grader did not provide feedback"),
        "grader_latency_ms": latency_ms,
    }


def regrade_ds1000_run(
    *,
    run_dir: str | Path,
    cache_dir: str | Path,
    source_path: str | Path,
    evaluator_image: str,
    output_path: str | Path | None = None,
    case_ids: set[str] | None = None,
) -> Path:
    """Regrade a frozen run exclusively from cached completions; never calls a model."""
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    benchmark_manifest = manifest.get("benchmark_manifest", "benchmarks/manifests/ds1000.json")
    all_examples: dict[str, DS1000Example] = {}
    for partition in Partition:
        for example in load_optimizer_examples(
            source_path, benchmark_manifest, partition=partition, libraries={"Numpy", "Pandas"}
        ):
            all_examples[example.case_id] = example

    model = manifest["candidate_model"]
    max_tokens = int(manifest.get("candidate_max_tokens", 1024))
    reasoning_effort = manifest.get("reasoning_effort")
    completion_dir = Path(cache_dir) / "completions"
    grade_dir = Path(cache_dir) / "grades"
    grade_dir.mkdir(parents=True, exist_ok=True)
    missing: list[dict[str, Any]] = []

    def grade_case(case_id: str, prompt: str) -> float | None:
        fingerprint = completion_fingerprint(
            case_id=case_id,
            model=model,
            system_prompt=prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        completion_path = completion_dir / f"{fingerprint}.json"
        if not completion_path.exists():
            missing.append({"case_id": case_id, "completion_fingerprint": fingerprint})
            return None
        completion = json.loads(completion_path.read_text())["completion"]
        grade_hash = grade_fingerprint(completion=completion, evaluator_image=evaluator_image)
        grade_path = grade_dir / f"{grade_hash}.json"
        if grade_path.exists():
            grade = json.loads(grade_path.read_text())
        else:
            grade = _execute_grade(all_examples[case_id], completion, evaluator_image)
            grade_path.write_text(json.dumps(grade, indent=2, sort_keys=True) + "\n")
        return float(grade["score"])

    validation = []
    validation_ids = manifest["validation_case_ids"]
    candidates = result["candidates"]
    selected_validation_ids = [
        case_id for case_id in validation_ids if case_ids is None or case_id in case_ids
    ]
    for index, candidate in enumerate(candidates):
        original_scores = result["val_subscores"][index]
        original_passed = int(sum(float(score) for score in original_scores.values()))
        corrections = []
        corrected_passed = original_passed
        for case_id in selected_validation_ids:
            position = validation_ids.index(case_id)
            old_score = float(original_scores[str(position)])
            new_score = grade_case(case_id, candidate["system_prompt"])
            corrections.append(
                {"case_id": case_id, "old_score": old_score, "new_score": new_score}
            )
            if new_score is not None:
                corrected_passed += int(new_score - old_score)
        validation.append(
            {
                "candidate_index": index,
                "original_passed": original_passed,
                "corrected_passed": corrected_passed,
                "total": len(validation_ids),
                "corrections": corrections,
                "complete": all(item["new_score"] is not None for item in corrections),
            }
        )

    decision: dict[str, Any] | None = None
    decision_path = run / "decision-test.json"
    if decision_path.exists():
        decision_data = json.loads(decision_path.read_text())
        decision_ids = decision_data["case_ids"]
        selected_decision_ids = [
            case_id for case_id in decision_ids if case_ids is None or case_id in case_ids
        ]
        best_index = int(result["best_idx"])
        decision = {}
        for label, prompt in {
            "baseline": manifest["seed_prompt"],
            "optimized": candidates[best_index]["system_prompt"],
        }.items():
            original_passed = int(decision_data[label]["passed"])
            original_by_case = {
                trace["case_id"]: float(trace["score"])
                for trace in decision_data[label]["traces"]
            }
            corrected_passed = original_passed
            corrections = []
            for case_id in selected_decision_ids:
                old_score = original_by_case[case_id]
                new_score = grade_case(case_id, prompt)
                corrections.append(
                    {"case_id": case_id, "old_score": old_score, "new_score": new_score}
                )
                if new_score is not None:
                    corrected_passed += int(new_score - old_score)
            decision[label] = {
                "original_passed": original_passed,
                "corrected_passed": corrected_passed,
                "total": len(decision_ids),
                "corrections": corrections,
                "complete": all(item["new_score"] is not None for item in corrections),
            }

    report = {
        "run_dir": run.name,
        "evaluator_image": evaluator_image,
        "target_case_ids": sorted(case_ids) if case_ids is not None else None,
        "model_calls_made": 0,
        "validation": validation,
        "decision": decision,
        "missing_completions": missing,
    }
    destination = Path(output_path or run / "regrade.json")
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination
