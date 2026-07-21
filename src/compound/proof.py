from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gepa

from compound.budget import BudgetLedger
from compound.config import load_config, require_paid_run_budget
from compound.contracts import Partition
from compound.costs import TokenPrices
from compound.ds1000_optimizer import (
    DEFAULT_SEED_PROMPT,
    CachedDS1000Adapter,
    MeteredReflectionLM,
    load_optimizer_examples,
)
from compound.env import load_env
from compound.providers import OpenAICompatibleProvider


def _prices(config: dict[str, Any], model: str) -> TokenPrices:
    values = config["pricing_usd_per_million_tokens"][model]
    return TokenPrices(input_per_million=values["input"], output_per_million=values["output"])


def _provider(config: dict[str, Any], name: str) -> OpenAICompatibleProvider:
    values = config["providers"][name]
    return OpenAICompatibleProvider(
        name=name,
        base_url=values["base_url"],
        api_key_env=values["api_key_env"],
        telemetry_path=Path(config["artifacts_dir"]) / "telemetry" / "model_calls.jsonl",
    )


def run_ds1000_proof(
    *,
    config_path: str | Path = "compound.yaml",
    max_metric_calls: int = 20,
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    candidate_max_tokens: int = 1024,
    reflection_max_tokens: int = 1400,
) -> Path:
    load_env()
    config = load_config(config_path)
    limit = require_paid_run_budget(config)
    artifacts = Path(config["artifacts_dir"])
    ledger = BudgetLedger.load(artifacts / "budget.json", limit)

    candidate_model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    reflection_model = "openai/gpt-5.6-sol"
    candidate_provider = _provider(config, "doubleword")
    reflection_provider = _provider(config, "openrouter")

    source_path = Path(".compound/sources/ds1000/data/ds1000.jsonl.gz")
    benchmark_manifest = Path(
        manifest_path or Path(config["manifests_dir"]) / "ds1000.json"
    )
    libraries = {"Numpy", "Pandas"}
    trainset = load_optimizer_examples(
        source_path,
        benchmark_manifest,
        partition=Partition.OPTIMIZER_TRAIN,
        libraries=libraries,
    )
    valset = load_optimizer_examples(
        source_path,
        benchmark_manifest,
        partition=Partition.OPTIMIZER_VALIDATION,
        libraries=libraries,
    )
    if not trainset or not valset:
        raise RuntimeError("DS-1000 NumPy/Pandas train and validation examples are required")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_dir or artifacts / "optimization" / f"ds1000-{timestamp}")
    destination.mkdir(parents=True, exist_ok=False)

    adapter = CachedDS1000Adapter(
        provider=candidate_provider,
        model=candidate_model,
        prices=_prices(config, candidate_model),
        ledger=ledger,
        cache_dir=artifacts / "cache" / "ds1000",
        docker_image="compound-ds1000-numpy:20260717-v2",
        max_tokens=candidate_max_tokens,
    )
    reflection_lm = MeteredReflectionLM(
        provider=reflection_provider,
        model=reflection_model,
        prices=_prices(config, reflection_model),
        ledger=ledger,
        cache_dir=artifacts / "cache" / "reflection",
        max_tokens=reflection_max_tokens,
    )

    run_manifest = {
        "benchmark": "ds1000",
        "benchmark_manifest": str(benchmark_manifest),
        "candidate_model": candidate_model,
        "reflection_model": reflection_model,
        "seed_prompt": DEFAULT_SEED_PROMPT,
        "train_case_ids": [example.case_id for example in trainset],
        "validation_case_ids": [example.case_id for example in valset],
        "decision_case_ids_exposed": [],
        "max_metric_calls": max_metric_calls,
        "candidate_max_tokens": candidate_max_tokens,
        "reflection_max_tokens": reflection_max_tokens,
        "evaluator_image": "compound-ds1000-numpy:20260717-v2",
        "budget_before_usd": ledger.spent_usd,
    }
    (destination / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": DEFAULT_SEED_PROMPT},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=2,
        max_metric_calls=max_metric_calls,
        run_dir=str(destination / "gepa"),
        cache_evaluation=True,
        seed=int(config["seed"]),
        display_progress_bar=True,
    )
    summary = {
        "best_candidate": result.best_candidate,
        "best_candidate_index": result.best_idx,
        "baseline_validation_score": result.val_aggregate_scores[0],
        "best_validation_score": result.val_aggregate_scores[result.best_idx],
        "candidate_count": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "validation_scores": result.val_aggregate_scores,
        "budget_after_usd": ledger.spent_usd,
        "incremental_cost_usd": ledger.spent_usd - run_manifest["budget_before_usd"],
        "remaining_budget_usd": ledger.remaining_usd,
    }
    (destination / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return destination


def evaluate_ds1000_decision_test(
    optimization_dir: str | Path,
    *,
    config_path: str | Path = "compound.yaml",
) -> Path:
    """Run the final comparison after optimization; this is the only decision-set entrypoint."""
    load_env()
    config = load_config(config_path)
    limit = require_paid_run_budget(config)
    artifacts = Path(config["artifacts_dir"])
    ledger = BudgetLedger.load(artifacts / "budget.json", limit)
    destination = Path(optimization_dir)
    summary = json.loads((destination / "summary.json").read_text())
    run_manifest = json.loads((destination / "manifest.json").read_text())

    candidate_model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    adapter = CachedDS1000Adapter(
        provider=_provider(config, "doubleword"),
        model=candidate_model,
        prices=_prices(config, candidate_model),
        ledger=ledger,
        cache_dir=artifacts / "cache" / "ds1000",
        docker_image=run_manifest.get(
            "evaluator_image", "compound-ds1000-numpy:20260717-v2"
        ),
        max_tokens=int(run_manifest.get("candidate_max_tokens", 1024)),
    )
    decision_examples = load_optimizer_examples(
        ".compound/sources/ds1000/data/ds1000.jsonl.gz",
        run_manifest.get(
            "benchmark_manifest", str(Path(config["manifests_dir"]) / "ds1000.json")
        ),
        partition=Partition.DECISION_TEST,
        libraries={"Numpy", "Pandas"},
    )
    before = ledger.spent_usd
    comparisons: dict[str, Any] = {}
    prompts = {
        "baseline": DEFAULT_SEED_PROMPT,
        "optimized": summary["best_candidate"]["system_prompt"],
    }
    for name, prompt in prompts.items():
        batch = adapter.evaluate(
            decision_examples, {"system_prompt": prompt}, capture_traces=True
        )
        comparisons[name] = {
            "score": sum(batch.scores) / len(batch.scores),
            "passed": int(sum(batch.scores)),
            "total": len(batch.scores),
            "traces": [asdict(trace) for trace in batch.trajectories or []],
        }
    result = {
        "case_ids": [example.case_id for example in decision_examples],
        "optimizer_accessed_decision_cases": False,
        "baseline": comparisons["baseline"],
        "optimized": comparisons["optimized"],
        "incremental_cost_usd": ledger.spent_usd - before,
        "total_spent_usd": ledger.spent_usd,
        "remaining_budget_usd": ledger.remaining_usd,
    }
    output_path = destination / "decision-test.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output_path
