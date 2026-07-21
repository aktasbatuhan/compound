from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from compound.ds1000_cache import migrate_legacy_ds1000_cache, regrade_ds1000_run
from compound.ds1000_optimizer import completion_fingerprint, grade_fingerprint


def test_legacy_migration_splits_and_archives_trace(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    runs = tmp_path / "runs"
    run = runs / "ds1000-run"
    run.mkdir(parents=True)
    prompt = "solve"
    model = "candidate"
    manifest = {
        "candidate_model": model,
        "candidate_max_tokens": 1024,
        "seed_prompt": prompt,
        "train_case_ids": ["ds1000_1"],
        "validation_case_ids": [],
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "result.json").write_text(
        json.dumps({"candidates": [{"system_prompt": prompt}]})
    )
    legacy_payload = {
        "case_id": "ds1000_1",
        "model": model,
        "system_prompt": prompt,
        "max_tokens": 1024,
        "reasoning_effort": None,
        "evaluator": "ds1000-official-context-v1",
    }
    legacy_hash = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True).encode()
    ).hexdigest()
    trace = {
        "completion": "result = 1",
        "score": 0.0,
        "feedback": "FAIL: AssertionError. Revise the instruction to prevent this failure mode.",
        "grader_latency_ms": 10,
        "model_latency_ms": 20,
        "output_tokens": 5,
    }
    (cache / f"{legacy_hash}.json").write_text(json.dumps(trace))

    result = migrate_legacy_ds1000_cache(
        cache_dir=cache,
        runs_dir=runs,
        reflection_cache_dir=tmp_path / "reflection",
        legacy_evaluator_image="old-image",
        archive=True,
    )

    new_hash = completion_fingerprint(
        case_id="ds1000_1",
        model=model,
        system_prompt=prompt,
        max_tokens=1024,
        reasoning_effort=None,
    )
    grade_hash = grade_fingerprint(completion="result = 1", evaluator_image="old-image")
    assert (cache / "completions" / f"{new_hash}.json").exists()
    assert (cache / "grades" / f"{grade_hash}.json").exists()
    assert (cache / "legacy" / f"{legacy_hash}.json").exists()
    assert result.archived_legacy_files == 1
    assert result.unmatched_legacy_files == ()


def test_regrade_uses_cached_completions_only(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "ds1000.jsonl.gz"
    records = []
    for problem_id in (1, 2):
        records.append(
            {
                "prompt": f"task {problem_id}",
                "code_context": "def test_execution(code): pass",
                "metadata": {
                    "problem_id": problem_id,
                    "library": "Numpy",
                    "perturbation_type": "Origin",
                },
            }
        )
    with gzip.open(source, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    benchmark_manifest = tmp_path / "manifest.json"
    benchmark_manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "ds1000_1",
                        "partition": "optimizer_validation",
                        "metadata": {"problem_id": 1},
                    },
                    {
                        "case_id": "ds1000_2",
                        "partition": "decision_test",
                        "metadata": {"problem_id": 2},
                    },
                ]
            }
        )
    )
    run = tmp_path / "run"
    run.mkdir()
    prompts = ["seed", "optimized"]
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark_manifest": str(benchmark_manifest),
                "candidate_model": "candidate",
                "candidate_max_tokens": 100,
                "seed_prompt": prompts[0],
                "validation_case_ids": ["ds1000_1"],
            }
        )
    )
    (run / "result.json").write_text(
        json.dumps(
            {
                "candidates": [{"system_prompt": prompt} for prompt in prompts],
                "best_idx": 1,
                "val_subscores": [{"0": 0.0}, {"0": 0.0}],
            }
        )
    )
    (run / "decision-test.json").write_text(
        json.dumps(
            {
                "case_ids": ["ds1000_2"],
                "baseline": {
                    "passed": 0,
                    "traces": [{"case_id": "ds1000_2", "score": 0.0}],
                },
                "optimized": {
                    "passed": 0,
                    "traces": [{"case_id": "ds1000_2", "score": 0.0}],
                },
            }
        )
    )
    cache = tmp_path / "cache"
    completion_dir = cache / "completions"
    completion_dir.mkdir(parents=True)
    for case_id in ("ds1000_1", "ds1000_2"):
        for prompt in prompts:
            fingerprint = completion_fingerprint(
                case_id=case_id,
                model="candidate",
                system_prompt=prompt,
                max_tokens=100,
                reasoning_effort=None,
            )
            (completion_dir / f"{fingerprint}.json").write_text(
                json.dumps({"completion": f"pass {case_id} {prompt}"})
            )

    monkeypatch.setattr(
        "compound.ds1000_cache._execute_grade",
        lambda example, completion, evaluator_image: {
            "score": float("optimized" in completion),
            "feedback": "graded",
            "grader_latency_ms": 1,
        },
    )
    output = regrade_ds1000_run(
        run_dir=run,
        cache_dir=cache,
        source_path=source,
        evaluator_image="new-image",
    )
    report = json.loads(output.read_text())

    assert report["model_calls_made"] == 0
    assert report["validation"][0]["corrected_passed"] == 0
    assert report["validation"][1]["corrected_passed"] == 1
    assert report["decision"]["baseline"]["corrected_passed"] == 0
    assert report["decision"]["optimized"]["corrected_passed"] == 1
    assert report["missing_completions"] == []
