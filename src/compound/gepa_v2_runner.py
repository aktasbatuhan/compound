from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gepa

from compound.budget import BudgetLedger
from compound.config import load_config, require_paid_run_budget
from compound.contracts import Partition
from compound.costs import TokenPrices
from compound.ds1000_optimizer import load_optimizer_examples
from compound.env import load_env
from compound.flex import CachedFlexRunner
from compound.gepa_v2 import (
    MAX_CANDIDATE_WORDS,
    PERFECT_COMPOSITE_SCORE,
    SEED_COMPONENTS,
    DS1000GEPAAdapterV2,
    ExperimentCap,
    MeteredFailureCritic,
    MeteredV2ReflectionLM,
    NamespacedTrialLoader,
    TrialExample,
    candidate_word_count,
    expand_trials,
    select_library_components,
    select_relevant_components,
)
from compound.providers import OpenAICompatibleProvider

EVALUATOR_IMAGE = "compound-ds1000-numpy:20260720-v3"
CANDIDATE_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
REFLECTION_MODEL = "openai/gpt-5.6-sol"


def _prices(config: dict[str, Any], model: str) -> TokenPrices:
    values = config["pricing_usd_per_million_tokens"][model]
    return TokenPrices(values["input"], values["output"])


def _flex_prices(config: dict[str, Any], model: str) -> TokenPrices:
    values = config["flex_pricing_usd_per_million_tokens"][model]
    return TokenPrices(values["input"], values["output"])


def _provider(config: dict[str, Any], name: str) -> OpenAICompatibleProvider:
    values = config["providers"][name]
    return OpenAICompatibleProvider(
        name=name,
        base_url=values["base_url"],
        api_key_env=values["api_key_env"],
        telemetry_path=Path(config["artifacts_dir"]) / "telemetry" / "model_calls.jsonl",
    )


def _summary(batch: Any) -> dict[str, Any]:
    traces = batch.trajectories or []
    task_scores = [trace.metrics["task_success"] for trace in traces]
    by_case: dict[str, list[float]] = {}
    for trace in traces:
        by_case.setdefault(trace.case_id, []).append(trace.metrics["task_success"])
    case_rates = {case_id: statistics.fmean(scores) for case_id, scores in by_case.items()}
    latencies = [trace.model_latency_ms for trace in traces if trace.model_latency_ms > 0]
    timed_output_tokens = sum(trace.output_tokens for trace in traces if trace.model_latency_ms > 0)
    latency_ms_total = sum(latencies)
    ordered_latencies = sorted(latencies)

    def percentile(quantile: float) -> float | None:
        if not ordered_latencies:
            return None
        index = round((len(ordered_latencies) - 1) * quantile)
        return ordered_latencies[index]

    return {
        "trials": len(traces),
        "cases": len(by_case),
        "passed_trials": int(sum(task_scores)),
        "task_success_rate": statistics.fmean(task_scores) if task_scores else 0.0,
        "case_success_rates": case_rates,
        "composite_score_mean": (
            statistics.fmean(trace.composite_score for trace in traces) if traces else 0.0
        ),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_p50": percentile(0.50),
        "latency_ms_p95": percentile(0.95) if len(latencies) >= 20 else None,
        "timed_trials": len(latencies),
        "timed_output_tokens": timed_output_tokens,
        "e2e_output_tps_aggregate": (
            timed_output_tokens / (latency_ms_total / 1000) if latency_ms_total > 0 else None
        ),
        "output_tokens": sum(trace.output_tokens for trace in traces),
        "reasoning_tokens": sum(trace.reasoning_tokens for trace in traces),
        "cost_usd": sum(trace.cost_usd for trace in traces),
    }


def _trial_records(batch: Any) -> list[dict[str, Any]]:
    return [
        {
            "case_id": trace.case_id,
            "trial_id": trace.trial_id,
            "library": trace.library,
            "passed": trace.metrics["task_success"] == 1.0,
            "metrics": trace.metrics,
            "feedback": trace.feedback,
            "finish_reason": trace.finish_reason,
            "model_latency_ms": trace.model_latency_ms,
            "grader_latency_ms": trace.grader_latency_ms,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "reasoning_tokens": trace.reasoning_tokens,
            "e2e_output_tps": trace.e2e_output_tps,
            "cost_usd": trace.cost_usd,
        }
        for trace in (batch.trajectories or [])
    ]


def _adapter(
    *,
    config: dict[str, Any],
    ledger: BudgetLedger,
    cap: ExperimentCap,
    max_tokens: int,
    reasoning_effort: str | None,
    with_critic: bool,
    candidate_model: str = CANDIDATE_MODEL,
    candidate_backend: str = "chat_completions",
    cache_only: bool = False,
    teacher_traces: dict[str, dict[str, str]] | None = None,
) -> DS1000GEPAAdapterV2:
    if candidate_backend not in {"chat_completions", "flex"}:
        raise ValueError(f"unsupported candidate backend: {candidate_backend}")
    candidate_provider = _provider(config, "doubleword")
    reflection_provider = _provider(config, "openrouter")
    critic = None
    if with_critic:
        critic = MeteredFailureCritic(
            provider=reflection_provider,
            model=REFLECTION_MODEL,
            prices=_prices(config, REFLECTION_MODEL),
            ledger=ledger,
            experiment_cap=cap,
            cache_dir=Path(config["artifacts_dir"]) / "cache" / "critic-v2",
            teacher_traces=teacher_traces,
        )
    flex_runner = None
    candidate_prices = (
        _flex_prices(config, candidate_model)
        if candidate_backend == "flex"
        else _prices(config, candidate_model)
    )
    if candidate_backend == "flex":
        flex_runner = CachedFlexRunner(
            provider=candidate_provider,
            prices_by_model={candidate_model: candidate_prices},
            ledger=ledger,
            cache_dir=Path(config["artifacts_dir"]) / "cache" / "flex-responses",
            poll_interval_seconds=2,
            timeout_seconds=3600,
            reserve_per_new_request_usd=0.03,
            headroom_checker=cap.require_headroom,
        )
    return DS1000GEPAAdapterV2(
        provider=candidate_provider,
        model=candidate_model,
        prices=candidate_prices,
        ledger=ledger,
        experiment_cap=cap,
        cache_dir=Path(config["artifacts_dir"]) / "cache" / "ds1000",
        docker_image=EVALUATOR_IMAGE,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        critic=critic,
        cache_only=cache_only,
        telemetry_call_type="candidate_gepa_v2",
        flex_runner=flex_runner,
    )


def _load_teacher_traces(
    artifact_path: str | Path,
    *,
    preferred_models: list[str],
    allowed_case_ids: set[str],
) -> dict[str, dict[str, str]]:
    payload = json.loads(Path(artifact_path).read_text())
    models = {str(record["model"]): record for record in payload.get("models") or []}
    unknown = sorted(set(preferred_models) - set(models))
    if unknown:
        raise ValueError(f"teacher models are absent from artifact: {unknown}")
    teachers: dict[str, dict[str, str]] = {}
    for model in preferred_models:
        cases = sorted(
            models[model].get("cases") or [],
            key=lambda record: (str(record["case_id"]), int(record.get("trial_id", 0))),
        )
        for record in cases:
            case_id = str(record["case_id"])
            if (
                case_id in allowed_case_ids
                and case_id not in teachers
                and bool(record.get("passed"))
                and str(record.get("completion") or "").strip()
            ):
                teachers[case_id] = {
                    "model": model,
                    "completion": str(record["completion"]),
                }
    return teachers


def _paired_comparison(baseline: Any, optimized: Any) -> dict[str, Any]:
    baseline_traces = baseline.trajectories or []
    optimized_traces = optimized.trajectories or []
    baseline_ids = [(trace.case_id, trace.trial_id) for trace in baseline_traces]
    optimized_ids = [(trace.case_id, trace.trial_id) for trace in optimized_traces]
    if baseline_ids != optimized_ids:
        raise ValueError("baseline and optimized decision traces are not aligned")

    transitions = {
        "both_fail": 0,
        "baseline_only_pass": 0,
        "optimized_only_pass": 0,
        "both_pass": 0,
    }
    baseline_by_case: dict[str, list[float]] = {}
    optimized_by_case: dict[str, list[float]] = {}
    for baseline_trace, optimized_trace in zip(baseline_traces, optimized_traces, strict=True):
        baseline_pass = baseline_trace.metrics["task_success"] == 1.0
        optimized_pass = optimized_trace.metrics["task_success"] == 1.0
        if baseline_pass and optimized_pass:
            transitions["both_pass"] += 1
        elif baseline_pass:
            transitions["baseline_only_pass"] += 1
        elif optimized_pass:
            transitions["optimized_only_pass"] += 1
        else:
            transitions["both_fail"] += 1
        baseline_by_case.setdefault(baseline_trace.case_id, []).append(float(baseline_pass))
        optimized_by_case.setdefault(optimized_trace.case_id, []).append(float(optimized_pass))

    baseline_rates = {
        case_id: statistics.fmean(scores) for case_id, scores in baseline_by_case.items()
    }
    optimized_rates = {
        case_id: statistics.fmean(scores) for case_id, scores in optimized_by_case.items()
    }
    gains = [
        case_id for case_id, score in optimized_rates.items() if score > baseline_rates[case_id]
    ]
    losses = [
        case_id for case_id, score in optimized_rates.items() if score < baseline_rates[case_id]
    ]
    return {
        "trial_transitions": transitions,
        "case_rate_gains": gains,
        "case_rate_losses": losses,
        "case_rate_unchanged": len(optimized_rates) - len(gains) - len(losses),
        "fully_reliable_cases": {
            "baseline": sum(score == 1.0 for score in baseline_rates.values()),
            "optimized": sum(score == 1.0 for score in optimized_rates.values()),
        },
        "cases_with_any_pass": {
            "baseline": sum(score > 0.0 for score in baseline_rates.values()),
            "optimized": sum(score > 0.0 for score in optimized_rates.values()),
        },
    }


def _decision_manifest_cases(
    *,
    run_manifest: dict[str, Any],
    decision_manifest_path: str | Path,
) -> list[dict[str, Any]]:
    decision_path = Path(decision_manifest_path)
    payload = json.loads(decision_path.read_text())
    cases = list(payload.get("cases") or [])
    if not cases:
        raise ValueError("decision manifest has no cases")
    if any(case.get("partition") != Partition.DECISION_TEST.value for case in cases):
        raise ValueError("decision manifest must contain only decision_test cases")

    optimization_path = Path(str(run_manifest["benchmark_manifest"]))
    external_manifest = decision_path.resolve() != optimization_path.resolve()
    if external_manifest and payload.get("cohort_role") != "sealed_decision":
        raise ValueError("external decision manifest must have cohort_role=sealed_decision")

    optimization_payload = json.loads(optimization_path.read_text())
    optimization_cases = list(optimization_payload.get("cases") or [])
    optimization_ids = {
        str(case["case_id"])
        for case in optimization_cases
        if case.get("partition")
        in {Partition.OPTIMIZER_TRAIN.value, Partition.OPTIMIZER_VALIDATION.value}
    }
    decision_ids = {str(case["case_id"]) for case in cases}
    case_overlap = sorted(optimization_ids & decision_ids)
    if case_overlap:
        raise ValueError(f"decision cases overlap optimizer cases: {case_overlap}")

    def origin_group(case: dict[str, Any]) -> str:
        metadata = case.get("metadata") or {}
        origin = metadata.get("perturbation_origin_id")
        stratum = case.get("stratum")
        if stratum is None or origin is None:
            raise ValueError("decision firewall requires stratum and perturbation_origin_id")
        return f"{stratum}:{origin}"

    optimization_groups = {
        origin_group(case)
        for case in optimization_cases
        if case.get("partition")
        in {Partition.OPTIMIZER_TRAIN.value, Partition.OPTIMIZER_VALIDATION.value}
    }
    decision_groups = {origin_group(case) for case in cases}
    group_overlap = sorted(optimization_groups & decision_groups)
    if group_overlap:
        raise ValueError(f"decision origin groups overlap optimizer origin groups: {group_overlap}")
    if len(decision_groups) != len(cases):
        raise ValueError("decision manifest contains repeated perturbation-origin groups")
    return cases


def run_gepa_v2(
    *,
    manifest_path: str | Path,
    config_path: str | Path = "compound.yaml",
    max_metric_calls: int = 120,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    validation_trials: int = 2,
    decision_trials: int = 2,
    experiment_cap_usd: float = 4.0,
    output_dir: str | Path | None = None,
    candidate_model: str = CANDIDATE_MODEL,
    candidate_backend: str = "chat_completions",
    component_policy: str = "adaptive",
    teacher_artifact_path: str | Path | None = None,
    teacher_models: list[str] | None = None,
) -> Path:
    load_env()
    config = load_config(config_path)
    hard_limit = require_paid_run_budget(config)
    artifacts = Path(config["artifacts_dir"])
    ledger = BudgetLedger.load(artifacts / "budget.json", hard_limit)
    source = ".compound/sources/ds1000/data/ds1000.jsonl.gz"
    train_base = load_optimizer_examples(source, manifest_path, partition=Partition.OPTIMIZER_TRAIN)
    validation_base = load_optimizer_examples(
        source, manifest_path, partition=Partition.OPTIMIZER_VALIDATION
    )
    trainset = expand_trials(train_base, 1)
    valset = expand_trials(validation_base, validation_trials)
    reflection_minibatch_size = 3
    worst_case_iteration_calls = len(valset) + 2 * reflection_minibatch_size
    engine_metric_call_limit = max(len(valset), max_metric_calls - worst_case_iteration_calls + 1)
    component_selectors = {
        "adaptive": select_relevant_components,
        "library_only": select_library_components,
    }
    if component_policy not in component_selectors:
        raise ValueError(f"unsupported component policy: {component_policy}")
    selected_teacher_models = teacher_models or []
    if bool(teacher_artifact_path) != bool(selected_teacher_models):
        raise ValueError("teacher artifact and teacher models must be provided together")
    teacher_traces = (
        _load_teacher_traces(
            teacher_artifact_path,
            preferred_models=selected_teacher_models,
            allowed_case_ids={example.case_id for example in train_base},
        )
        if teacher_artifact_path is not None
        else {}
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_dir or artifacts / "optimization" / f"ds1000-v2-{timestamp}")
    expected_manifest = {
        "algorithm": "gepa-v2",
        "benchmark_manifest": str(manifest_path),
        "candidate_model": candidate_model,
        "candidate_backend": candidate_backend,
        "component_policy": component_policy,
        "teacher_artifact": str(teacher_artifact_path) if teacher_artifact_path else None,
        "teacher_models": selected_teacher_models,
        "teacher_case_ids": sorted(teacher_traces),
        "reflection_model": REFLECTION_MODEL,
        "evaluator_image": EVALUATOR_IMAGE,
        "candidate_max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "validation_trials": validation_trials,
        "decision_trials": decision_trials,
        "max_metric_calls": max_metric_calls,
        "engine_metric_call_limit": engine_metric_call_limit,
        "gepa_data_id_policy": "disjoint train:* and validation:* ids",
        "experiment_cap_usd": experiment_cap_usd,
        "budget_before_usd": ledger.spent_usd,
        "train_case_ids": [example.case_id for example in train_base],
        "validation_case_ids": [example.case_id for example in validation_base],
        "decision_case_ids_exposed": [],
        "seed_candidate": SEED_COMPONENTS,
        "max_candidate_words": MAX_CANDIDATE_WORDS,
    }
    manifest_path_in_run = destination / "manifest.json"
    if destination.exists():
        if output_dir is None or not manifest_path_in_run.exists():
            raise FileExistsError(f"run directory already exists: {destination}")
        if (destination / "summary.json").exists():
            raise ValueError(f"run is already complete: {destination}")
        manifest = json.loads(manifest_path_in_run.read_text())
        for key, expected in expected_manifest.items():
            if key == "budget_before_usd":
                continue
            if manifest.get(key) != expected:
                raise ValueError(
                    f"resume configuration mismatch for {key}: "
                    f"{manifest.get(key)!r} != {expected!r}"
                )
    else:
        destination.mkdir(parents=True, exist_ok=False)
        manifest = expected_manifest
        manifest_path_in_run.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cap = ExperimentCap(
        ledger,
        float(manifest["budget_before_usd"]),
        float(manifest["experiment_cap_usd"]),
    )

    adapter = _adapter(
        config=config,
        ledger=ledger,
        cap=cap,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        with_critic=True,
        candidate_model=candidate_model,
        candidate_backend=candidate_backend,
        teacher_traces=teacher_traces,
    )
    reflection = MeteredV2ReflectionLM(
        provider=_provider(config, "openrouter"),
        model=REFLECTION_MODEL,
        prices=_prices(config, REFLECTION_MODEL),
        ledger=ledger,
        experiment_cap=cap,
        cache_dir=artifacts / "cache" / "reflection-v2",
    )
    template = (
        "Rewrite this prompt component using the evidence. Keep only reusable principles; never "
        "include a full case solution, displayed values, or case-specific variable names. Replace "
        "obsolete rules instead of appending. The replacement must be under 200 words.\n\n"
        "CURRENT COMPONENT:\n<curr_param>\n\nEVIDENCE:\n<side_info>\n\n"
        "Return only the complete replacement component in a fenced text block."
    )
    result = gepa.optimize(
        seed_candidate=dict(SEED_COMPONENTS),
        trainset=NamespacedTrialLoader(trainset, "train"),
        valset=NamespacedTrialLoader(valset, "validation"),
        adapter=adapter,
        reflection_lm=reflection,
        candidate_selection_strategy="pareto",
        frontier_type="hybrid",
        reflection_minibatch_size=reflection_minibatch_size,
        reflection_prompt_template={key: template for key in SEED_COMPONENTS},
        module_selector=component_selectors[component_policy],
        perfect_score=PERFECT_COMPOSITE_SCORE,
        max_metric_calls=engine_metric_call_limit,
        run_dir=str(destination / "gepa"),
        cache_evaluation=True,
        seed=int(config["seed"]) + 2,
        display_progress_bar=True,
    )

    baseline_batch = adapter.evaluate(valset, dict(SEED_COMPONENTS), capture_traces=True)
    best_candidate = dict(result.best_candidate)
    best_batch = adapter.evaluate(valset, best_candidate, capture_traces=True)
    baseline_summary = _summary(baseline_batch)
    best_summary = _summary(best_batch)
    gepa_objectives = result.val_aggregate_subscores
    if gepa_objectives is None:
        raise RuntimeError("GEPA did not retain validation objective scores")
    consistency = {
        "baseline_task_success": {
            "gepa": gepa_objectives[0]["task_success"],
            "direct_cached_recheck": baseline_summary["task_success_rate"],
        },
        "best_task_success": {
            "gepa": gepa_objectives[result.best_idx]["task_success"],
            "direct_cached_recheck": best_summary["task_success_rate"],
        },
    }
    if any(
        abs(values["gepa"] - values["direct_cached_recheck"]) > 1e-12
        for values in consistency.values()
    ):
        raise RuntimeError(f"GEPA validation cache consistency failure: {consistency}")
    summary = {
        "best_candidate_index": result.best_idx,
        "candidate_count": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "best_candidate": best_candidate,
        "best_candidate_words": candidate_word_count(best_candidate),
        "baseline_validation": baseline_summary,
        "best_validation": best_summary,
        "validation_consistency": consistency,
        "validation_consistency_verified": True,
        "gepa_validation_scores": result.val_aggregate_scores,
        "incremental_cost_usd": cap.spent_usd,
        "total_spent_usd": ledger.spent_usd,
        "remaining_budget_usd": ledger.remaining_usd,
    }
    (destination / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return destination


def probe_gepa_v2(
    *,
    manifest_path: str | Path,
    config_path: str | Path = "compound.yaml",
    max_tokens: int = 4096,
    reasoning_effort: str | None = "low",
    experiment_cap_usd: float = 0.50,
    candidate_model: str = CANDIDATE_MODEL,
    candidate_backend: str = "chat_completions",
) -> Path:
    """Exercise the exact candidate route on one optimizer-training case."""
    load_env()
    config = load_config(config_path)
    hard_limit = require_paid_run_budget(config)
    artifacts = Path(config["artifacts_dir"])
    ledger = BudgetLedger.load(artifacts / "budget.json", hard_limit)
    cap = ExperimentCap(ledger, ledger.spent_usd, experiment_cap_usd)
    examples = load_optimizer_examples(
        ".compound/sources/ds1000/data/ds1000.jsonl.gz",
        manifest_path,
        partition=Partition.OPTIMIZER_TRAIN,
    )
    if not examples:
        raise ValueError("manifest has no optimizer-training cases")
    adapter = _adapter(
        config=config,
        ledger=ledger,
        cap=cap,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        with_critic=False,
        candidate_model=candidate_model,
        candidate_backend=candidate_backend,
    )
    batch = adapter.evaluate(
        [TrialExample(examples[0], 0)], dict(SEED_COMPONENTS), capture_traces=True
    )
    report = {
        "candidate_model": candidate_model,
        "candidate_backend": candidate_backend,
        "evaluator_image": EVALUATOR_IMAGE,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "incremental_cost_usd": cap.spent_usd,
        "result": _summary(batch),
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = artifacts / "optimization" / f"ds1000-v2-probe-{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output


def evaluate_gepa_v2_decision(
    run_dir: str | Path,
    *,
    config_path: str | Path = "compound.yaml",
    decision_manifest_path: str | Path | None = None,
    reference_models: list[str] | None = None,
    reference_trials: int | None = None,
    experiment_cap_usd: float = 4.0,
    overwrite_cached: bool = False,
) -> Path:
    load_env()
    config = load_config(config_path)
    hard_limit = require_paid_run_budget(config)
    artifacts = Path(config["artifacts_dir"])
    ledger = BudgetLedger.load(artifacts / "budget.json", hard_limit)
    run = Path(run_dir)
    output = run / "decision-v2.json"
    prior_report = json.loads(output.read_text()) if output.exists() else None
    if output.exists() and not overwrite_cached:
        raise FileExistsError(f"decision report already exists: {output}")
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    decision_budget_before = ledger.spent_usd
    cap = ExperimentCap(ledger, decision_budget_before, experiment_cap_usd)
    selected_decision_manifest = decision_manifest_path or manifest["benchmark_manifest"]
    decision_records = _decision_manifest_cases(
        run_manifest=manifest,
        decision_manifest_path=selected_decision_manifest,
    )
    decision_base = load_optimizer_examples(
        ".compound/sources/ds1000/data/ds1000.jsonl.gz",
        selected_decision_manifest,
        partition=Partition.DECISION_TEST,
    )
    expected_ids = [str(case["case_id"]) for case in decision_records]
    loaded_ids = [example.case_id for example in decision_base]
    if len(loaded_ids) != len(expected_ids) or set(loaded_ids) != set(expected_ids):
        raise ValueError("decision loader membership does not match the frozen manifest")
    decision: list[TrialExample] = expand_trials(decision_base, int(manifest["decision_trials"]))
    adapter = _adapter(
        config=config,
        ledger=ledger,
        cap=cap,
        max_tokens=int(manifest["candidate_max_tokens"]),
        reasoning_effort=manifest.get("reasoning_effort"),
        with_critic=False,
        candidate_model=manifest.get("candidate_model", CANDIDATE_MODEL),
        candidate_backend=manifest.get("candidate_backend", "chat_completions"),
        cache_only=overwrite_cached,
    )
    baseline = adapter.evaluate(decision, dict(manifest["seed_candidate"]), capture_traces=True)
    optimized = adapter.evaluate(decision, dict(summary["best_candidate"]), capture_traces=True)
    selected_reference_models = reference_models or []
    selected_reference_trials = (
        int(manifest["decision_trials"]) if reference_trials is None else reference_trials
    )
    if selected_reference_trials < 1:
        raise ValueError("reference_trials must be positive")
    reference_batches: dict[str, Any] = {}
    for model in selected_reference_models:
        reference_adapter = DS1000GEPAAdapterV2(
            provider=_provider(config, "openrouter"),
            model=model,
            prices=_prices(config, model),
            ledger=ledger,
            experiment_cap=cap,
            cache_dir=artifacts / "cache" / "ds1000",
            docker_image=EVALUATOR_IMAGE,
            max_tokens=int(manifest["candidate_max_tokens"]),
            reasoning_effort=None,
            critic=None,
            cache_only=overwrite_cached,
            telemetry_call_type="reference_gepa_v2_decision",
        )
        reference_batches[model] = reference_adapter.evaluate(
            expand_trials(decision_base, selected_reference_trials),
            dict(manifest["seed_candidate"]),
            capture_traces=True,
        )
    optimized_reference_comparisons = {}
    if selected_reference_trials == int(manifest["decision_trials"]):
        optimized_reference_comparisons = {
            model: _paired_comparison(batch, optimized)
            for model, batch in reference_batches.items()
        }
    case_count = len(decision_base)
    trial_count = int(manifest["decision_trials"])
    report = {
        "optimizer_accessed_decision_cases": False,
        "decision_manifest": str(selected_decision_manifest),
        "origin_group_overlap_with_optimizer": [],
        "case_ids": [example.case_id for example in decision_base],
        "trials_per_case": trial_count,
        "baseline": _summary(baseline),
        "optimized": _summary(optimized),
        "paired_comparison": _paired_comparison(baseline, optimized),
        "reference_trials_per_case": selected_reference_trials,
        "references": {
            model: _summary(batch) for model, batch in reference_batches.items()
        },
        "trial_records": {
            "baseline": _trial_records(baseline),
            "optimized": _trial_records(optimized),
            "references": {
                model: _trial_records(batch) for model, batch in reference_batches.items()
            },
        },
        "optimized_vs_references": optimized_reference_comparisons,
        "confidence_note": (
            f"This result uses {case_count} origin-group-isolated cases with {trial_count} "
            "candidate trials per case. Trials within a case are not independent benchmark "
            "cases, so trial-rate deltas should not be treated as precise population estimates."
        ),
        "decision_budget_before_usd": (
            prior_report["decision_budget_before_usd"]
            if prior_report is not None
            else decision_budget_before
        ),
        "decision_cap_usd": (
            prior_report["decision_cap_usd"]
            if prior_report is not None
            else experiment_cap_usd
        ),
        "decision_spend_usd": (
            prior_report["decision_spend_usd"] if prior_report is not None else cap.spent_usd
        ),
        "cache_replay_spend_usd": cap.spent_usd if prior_report is not None else 0.0,
        "total_spent_usd": ledger.spent_usd,
        "remaining_budget_usd": ledger.remaining_usd,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output
