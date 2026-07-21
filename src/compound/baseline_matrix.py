from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compound.budget import BudgetLedger
from compound.config import load_config, require_paid_run_budget
from compound.contracts import Partition
from compound.costs import TokenPrices
from compound.ds1000_optimizer import completion_fingerprint, load_optimizer_examples
from compound.env import load_env
from compound.flex import CachedFlexRunner, FlexRequest
from compound.flex_smoke import (
    DEFAULT_EVALUATOR_IMAGE,
    flex_prices,
    grade_ds1000_completion,
)
from compound.gepa_v2 import (
    SEED_COMPONENTS,
    DS1000GEPAAdapterV2,
    ExperimentCap,
    compose_system_prompt,
    expand_trials,
)
from compound.providers import OpenAICompatibleProvider


def _token_prices(config: dict[str, Any], model: str) -> TokenPrices:
    values = config["pricing_usd_per_million_tokens"][model]
    return TokenPrices(
        input_per_million=float(values["input"]),
        output_per_million=float(values["output"]),
    )


def _provider(
    config: dict[str, Any], name: str, *, timeout_seconds: float = 180.0
) -> OpenAICompatibleProvider:
    values = config["providers"][name]
    return OpenAICompatibleProvider(
        name=name,
        base_url=values["base_url"],
        api_key_env=values["api_key_env"],
        telemetry_path=Path(config["artifacts_dir"]) / "telemetry" / "model_calls.jsonl",
        timeout_seconds=timeout_seconds,
        max_retries=2,
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _model_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_ms"]) for record in records if record["latency_ms"] > 0]
    timed_output_tokens = sum(
        int(record["output_tokens"]) for record in records if record["latency_ms"] > 0
    )
    latency_total = sum(latencies)
    submitted_at = [
        datetime.fromisoformat(record["submitted_at"])
        for record in records
        if record.get("submitted_at")
    ]
    completed_at = [
        datetime.fromisoformat(record["completed_at"])
        for record in records
        if record.get("completed_at")
    ]
    batch_makespan_ms = (
        max(0, round((max(completed_at) - min(submitted_at)).total_seconds() * 1000))
        if len(submitted_at) == len(records) and len(completed_at) == len(records) and records
        else None
    )
    libraries = sorted({str(record["library"]) for record in records})
    case_ids = {str(record["case_id"]) for record in records}
    case_success_rates = {
        case_id: statistics.fmean(
            float(record["passed"]) for record in records if record["case_id"] == case_id
        )
        for case_id in sorted(case_ids)
    }
    return {
        "cases": len(case_ids),
        "trials": len(records),
        "passed": sum(bool(record["passed"]) for record in records),
        "passed_trials": sum(bool(record["passed"]) for record in records),
        "pass_rate": (
            statistics.fmean(float(record["passed"]) for record in records) if records else 0.0
        ),
        "case_success_rates": case_success_rates,
        "by_library": {
            library: {
                "cases": len(
                    {
                        str(record["case_id"])
                        for record in records
                        if record["library"] == library
                    }
                ),
                "trials": sum(record["library"] == library for record in records),
                "passed": sum(
                    bool(record["passed"]) for record in records if record["library"] == library
                ),
            }
            for library in libraries
        },
        "timed_calls": len(latencies),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95) if len(latencies) >= 20 else None,
        "e2e_output_tps_aggregate": (
            timed_output_tokens / (latency_total / 1000) if latency_total > 0 else None
        ),
        "batch_makespan_ms": batch_makespan_ms,
        "batch_output_tps": (
            sum(int(record["output_tokens"]) for record in records)
            / (batch_makespan_ms / 1000)
            if batch_makespan_ms and batch_makespan_ms > 0
            else None
        ),
        "input_tokens": sum(int(record["input_tokens"]) for record in records),
        "output_tokens": sum(int(record["output_tokens"]) for record in records),
        "reasoning_tokens": sum(int(record["reasoning_tokens"]) for record in records),
        "estimated_cost_usd": sum(float(record["estimated_cost_usd"]) for record in records),
    }


def _reference_records(
    *,
    config: dict[str, Any],
    model: str,
    examples: list[Any],
    ledger: BudgetLedger,
    cap: ExperimentCap,
    evaluator_image: str,
    max_output_tokens: int,
    trials_per_case: int,
) -> list[dict[str, Any]]:
    adapter = DS1000GEPAAdapterV2(
        provider=_provider(config, "openrouter"),
        model=model,
        prices=_token_prices(config, model),
        ledger=ledger,
        experiment_cap=cap,
        cache_dir=Path(config["artifacts_dir"]) / "cache" / "ds1000",
        docker_image=evaluator_image,
        max_tokens=max_output_tokens,
        reasoning_effort=None,
        critic=None,
        telemetry_call_type="reference_baseline_matrix",
    )
    batch = adapter.evaluate(
        expand_trials(examples, trials_per_case), SEED_COMPONENTS, capture_traces=True
    )
    traces = batch.trajectories or []
    return [
        {
            "case_id": trace.case_id,
            "trial_id": trace.trial_id,
            "library": trace.library,
            "passed": trace.metrics["task_success"] == 1.0,
            "score": trace.metrics["task_success"],
            "feedback": trace.feedback,
            "completion": trace.completion,
            "finish_reason": trace.finish_reason,
            "latency_ms": trace.model_latency_ms,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "reasoning_tokens": trace.reasoning_tokens,
            "e2e_output_tps": trace.e2e_output_tps,
            "estimated_cost_usd": trace.cost_usd,
        }
        for trace in traces
    ]


def _candidate_records(
    *,
    config: dict[str, Any],
    models: list[str],
    examples: list[Any],
    ledger: BudgetLedger,
    cap: ExperimentCap,
    evaluator_image: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    trials_per_case: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, list[dict[str, Any]]]:
    if not models:
        return {}
    artifacts = Path(config["artifacts_dir"])
    runner = CachedFlexRunner(
        provider=_provider(config, "doubleword", timeout_seconds=60),
        prices_by_model={model: flex_prices(config, model) for model in models},
        ledger=ledger,
        cache_dir=artifacts / "cache" / "flex-responses",
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        reserve_per_new_request_usd=0.02,
        headroom_checker=cap.require_headroom,
    )
    request_keys: list[tuple[str, Any, int]] = []
    requests = []
    for model in models:
        for example in examples:
            system_prompt = compose_system_prompt(
                SEED_COMPONENTS, str(example.metadata["library"])
            )
            for trial_id in range(trials_per_case):
                fingerprint = completion_fingerprint(
                    case_id=example.case_id,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                    trial_id=trial_id,
                )
                request_keys.append((model, example, trial_id))
                requests.append(
                    FlexRequest(
                        model=model,
                        input=example.prompt,
                        instructions=system_prompt,
                        max_output_tokens=max_output_tokens,
                        reasoning_effort=reasoning_effort,
                        cache_discriminator=fingerprint if trial_id else None,
                        telemetry_context={
                            "benchmark": "ds1000",
                            "case_id": example.case_id,
                            "trial_id": trial_id,
                            "call_type": "candidate_baseline_matrix",
                            "completion_fingerprint": fingerprint,
                        },
                    )
                )
    responses = runner.run_many(requests)
    records: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    grade_cache = artifacts / "cache" / "ds1000" / "grades"
    for (model, example, trial_id), response in zip(request_keys, responses, strict=True):
        grade = grade_ds1000_completion(
            case={"code_context": example.code_context},
            completion=response["output_text"],
            evaluator_image=evaluator_image,
            cache_dir=grade_cache,
        )
        records[model].append(
            {
                "case_id": example.case_id,
                "trial_id": trial_id,
                "library": str(example.metadata["library"]),
                "passed": grade["score"] == 1.0,
                "score": grade["score"],
                "feedback": grade["feedback"],
                "completion": response["output_text"],
                "finish_reason": response["finish_reason"],
                "status": response["status"],
                "response_id": response["response_id"],
                "submitted_at": response["submitted_at"],
                "completed_at": response["completed_at"],
                "latency_ms": response["e2e_latency_ms"],
                "submit_latency_ms": response["submit_latency_ms"],
                "poll_count": response["poll_count"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "reasoning_tokens": response["reasoning_tokens"],
                "e2e_output_tps": response["e2e_output_tps"],
                "estimated_cost_usd": response["estimated_cost_usd"],
                "billing_tier_confirmed_from_response": response[
                    "billing_tier_confirmed_from_response"
                ],
            }
        )
    return records


def run_ds1000_baseline_matrix(
    *,
    manifest_path: str | Path,
    config_path: str | Path = "compound.yaml",
    models: list[str] | None = None,
    case_ids: list[str] | None = None,
    max_output_tokens: int = 4096,
    reasoning_effort: str | None = None,
    trials_per_case: int = 1,
    experiment_cap_usd: float = 8.0,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 3600.0,
    evaluator_image: str = DEFAULT_EVALUATOR_IMAGE,
    output_path: str | Path | None = None,
) -> Path:
    load_env()
    if trials_per_case < 1:
        raise ValueError("trials_per_case must be positive")
    config = load_config(config_path)
    hard_limit = require_paid_run_budget(config)
    artifacts = Path(config["artifacts_dir"])
    ledger = BudgetLedger.load(artifacts / "budget.json", hard_limit)
    budget_before = ledger.spent_usd
    cap = ExperimentCap(ledger, budget_before, experiment_cap_usd)
    source = ".compound/sources/ds1000/data/ds1000.jsonl.gz"
    examples = load_optimizer_examples(
        source,
        manifest_path,
        partition=Partition.OPTIMIZER_TRAIN,
        libraries={"Numpy", "Pandas"},
    )
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("cohort_role") != "opened_baseline":
        raise ValueError("baseline matrix requires an opened_baseline manifest")
    if len(examples) != len(manifest["cases"]):
        raise ValueError("baseline manifest must contain only NumPy/Pandas optimizer_train cases")
    if case_ids:
        available_case_ids = {example.case_id for example in examples}
        unknown_case_ids = sorted(set(case_ids) - available_case_ids)
        if unknown_case_ids:
            raise ValueError(f"cases are not in the baseline manifest: {unknown_case_ids}")
        selected_case_ids = set(case_ids)
        examples = [example for example in examples if example.case_id in selected_case_ids]

    configured = config["models"]["frontier"] + config["models"]["candidates"]
    configured_by_id = {item["id"]: item for item in configured}
    selected_models = models or [item["id"] for item in configured]
    unknown = sorted(set(selected_models) - set(configured_by_id))
    if unknown:
        raise ValueError(f"models are not configured: {unknown}")
    reference_models = [
        model for model in selected_models if configured_by_id[model]["role"] == "reference"
    ]
    candidate_models = [
        model for model in selected_models if configured_by_id[model]["role"] == "candidate"
    ]

    candidate_records = _candidate_records(
        config=config,
        models=candidate_models,
        examples=examples,
        ledger=ledger,
        cap=cap,
        evaluator_image=evaluator_image,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        trials_per_case=trials_per_case,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    all_records: dict[str, list[dict[str, Any]]] = dict(candidate_records)
    for model in reference_models:
        all_records[model] = _reference_records(
            config=config,
            model=model,
            examples=examples,
            ledger=ledger,
            cap=cap,
            evaluator_image=evaluator_image,
            max_output_tokens=max_output_tokens,
            trials_per_case=trials_per_case,
        )

    model_results = []
    for model in selected_models:
        route = configured_by_id[model]["provider"]
        records = all_records[model]
        model_results.append(
            {
                "model": model,
                "role": configured_by_id[model]["role"],
                "provider": route,
                "api_surface": "responses" if route == "doubleword" else "chat.completions",
                "service_tier": "flex" if route == "doubleword" else "default",
                "summary": _model_summary(records),
                "cases": records,
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "ds1000",
        "cohort_role": "opened_baseline",
        "benchmark_manifest": str(manifest_path),
        "case_ids": [example.case_id for example in examples],
        "trials_per_case": trials_per_case,
        "seed_candidate": SEED_COMPONENTS,
        "max_output_tokens": max_output_tokens,
        "candidate_reasoning_effort": reasoning_effort,
        "evaluator_image": evaluator_image,
        "experiment_cap_usd": experiment_cap_usd,
        "budget_before_usd": budget_before,
        "budget_after_usd": ledger.spent_usd,
        "incremental_cost_usd": ledger.spent_usd - budget_before,
        "remaining_budget_usd": ledger.remaining_usd,
        "route_note": (
            "Reference models use OpenRouter's default Chat Completions route. Candidate models "
            "use Doubleword background Responses with requested service_tier=flex. Candidate "
            f"reasoning effort is {reasoning_effort or 'provider default'}. Candidate latency "
            "and TPS include async queueing."
        ),
        "models": model_results,
    }
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = artifacts / "baselines" / f"ds1000-matrix-{timestamp}.json"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output
