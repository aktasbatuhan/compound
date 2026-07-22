from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compound.adapters.bfcl import (
    MULTI_TURN_STRATUM,
    SINGLE_TURN_STRATUM,
    BFCLCase,
    grade_bfcl_single_turn,
    load_bfcl_cases,
    render_bfcl_single_turn_prompt,
)
from compound.budget import BudgetLedger
from compound.config import load_config, require_paid_run_budget
from compound.costs import TokenPrices, estimate_cost
from compound.ds1000_optimizer import completion_fingerprint, response_text
from compound.env import load_env
from compound.flex import CachedFlexRunner, FlexRequest
from compound.flex_smoke import flex_prices
from compound.gepa_v2 import ExperimentCap
from compound.providers import OpenAICompatibleProvider

DEFAULT_BFCL_SOURCE_DIR = ".compound/sources/gorilla"
MULTI_TURN_LIMITATION = (
    "multi_turn cases are frozen in the manifest but not graded: BFCL multi-turn "
    "scoring requires the official live execution harness (stateful API backends "
    "driven turn by turn), which this baseline runner does not implement. No "
    "grades are fabricated for them."
)
_RESERVE_PER_REFERENCE_CALL_USD = 0.30


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


def select_single_turn_cases(
    cases: list[BFCLCase], case_ids: list[str] | None
) -> tuple[list[BFCLCase], list[BFCLCase]]:
    """Split manifest cases into gradable single-turn cases and skipped rest."""
    single_turn = [case for case in cases if case.stratum == SINGLE_TURN_STRATUM]
    skipped = [case for case in cases if case.stratum != SINGLE_TURN_STRATUM]
    if not case_ids:
        return single_turn, skipped
    requested = set(case_ids)
    multi_turn_requested = sorted(
        requested & {case.case_id for case in skipped if case.stratum == MULTI_TURN_STRATUM}
    )
    if multi_turn_requested:
        raise ValueError(
            f"multi_turn cases cannot be graded by this runner: {multi_turn_requested}"
        )
    unknown = sorted(requested - {case.case_id for case in single_turn})
    if unknown:
        raise ValueError(f"cases are not in the BFCL manifest: {unknown}")
    return [case for case in single_turn if case.case_id in requested], skipped


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
    categories = sorted({str(record["category"]) for record in records})
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
        "by_category": {
            category: {
                "cases": len(
                    {
                        str(record["case_id"])
                        for record in records
                        if record["category"] == category
                    }
                ),
                "trials": sum(record["category"] == category for record in records),
                "passed": sum(
                    bool(record["passed"]) for record in records if record["category"] == category
                ),
            }
            for category in categories
        },
        "error_types": {
            error_type: sum(record.get("error_type") == error_type for record in records)
            for error_type in sorted(
                {
                    str(record["error_type"])
                    for record in records
                    if record.get("error_type")
                }
            )
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
            sum(int(record["output_tokens"]) for record in records) / (batch_makespan_ms / 1000)
            if batch_makespan_ms and batch_makespan_ms > 0
            else None
        ),
        "input_tokens": sum(int(record["input_tokens"]) for record in records),
        "output_tokens": sum(int(record["output_tokens"]) for record in records),
        "reasoning_tokens": sum(int(record["reasoning_tokens"]) for record in records),
        "estimated_cost_usd": sum(float(record["estimated_cost_usd"]) for record in records),
    }


def _base_record(case: BFCLCase, trial_id: int, grade: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "trial_id": trial_id,
        "category": case.category,
        "partition": case.partition.value,
        "passed": grade["score"] == 1.0,
        "score": grade["score"],
        "feedback": grade["feedback"],
        "error_type": grade["error_type"],
    }


def _reference_records(
    *,
    config: dict[str, Any],
    model: str,
    cases: list[BFCLCase],
    ledger: BudgetLedger,
    cap: ExperimentCap,
    max_output_tokens: int,
    trials_per_case: int,
) -> list[dict[str, Any]]:
    provider = _provider(config, "openrouter")
    prices = _token_prices(config, model)
    cache_dir = Path(config["artifacts_dir"]) / "cache" / "bfcl" / "completions"
    records: list[dict[str, Any]] = []
    for case in cases:
        system_prompt, user_prompt = render_bfcl_single_turn_prompt(case)
        for trial_id in range(trials_per_case):
            fingerprint = completion_fingerprint(
                case_id=case.case_id,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_output_tokens,
                reasoning_effort=None,
                trial_id=trial_id,
            )
            path = cache_dir / f"{fingerprint}.json"
            if path.exists():
                completion_record = json.loads(path.read_text())
            else:
                cap.require_headroom(_RESERVE_PER_REFERENCE_CALL_USD)
                response = provider.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_output_tokens,
                    telemetry_context={
                        "benchmark": "bfcl",
                        "case_id": case.case_id,
                        "trial_id": trial_id,
                        "call_type": "reference_baseline_matrix",
                        "completion_fingerprint": fingerprint,
                    },
                )
                choices = response.output.get("choices") or []
                completion_record = {
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
                    "cost_usd": estimate_cost(response.usage, prices),
                }
                ledger.record(completion_record["cost_usd"], label=f"bfcl-baseline:{fingerprint}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(completion_record, indent=2, sort_keys=True) + "\n")
            grade = grade_bfcl_single_turn(case, completion_record["completion"])
            records.append(
                {
                    **_base_record(case, trial_id, grade),
                    "completion": completion_record["completion"],
                    "finish_reason": completion_record["finish_reason"],
                    "latency_ms": completion_record["model_latency_ms"],
                    "input_tokens": completion_record["input_tokens"],
                    "output_tokens": completion_record["output_tokens"],
                    "reasoning_tokens": completion_record["reasoning_tokens"],
                    "e2e_output_tps": completion_record["e2e_output_tps"],
                    "estimated_cost_usd": completion_record["cost_usd"],
                }
            )
    return records


def _candidate_records(
    *,
    config: dict[str, Any],
    models: list[str],
    cases: list[BFCLCase],
    ledger: BudgetLedger,
    cap: ExperimentCap,
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
    request_keys: list[tuple[str, BFCLCase, int]] = []
    requests: list[FlexRequest] = []
    for model in models:
        for case in cases:
            system_prompt, user_prompt = render_bfcl_single_turn_prompt(case)
            for trial_id in range(trials_per_case):
                fingerprint = completion_fingerprint(
                    case_id=case.case_id,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                    trial_id=trial_id,
                )
                request_keys.append((model, case, trial_id))
                requests.append(
                    FlexRequest(
                        model=model,
                        input=user_prompt,
                        instructions=system_prompt,
                        max_output_tokens=max_output_tokens,
                        reasoning_effort=reasoning_effort,
                        cache_discriminator=fingerprint if trial_id else None,
                        telemetry_context={
                            "benchmark": "bfcl",
                            "case_id": case.case_id,
                            "trial_id": trial_id,
                            "call_type": "candidate_baseline_matrix",
                            "completion_fingerprint": fingerprint,
                        },
                    )
                )
    responses = runner.run_many(requests)
    records: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    for (model, case, trial_id), response in zip(request_keys, responses, strict=True):
        grade = grade_bfcl_single_turn(case, response["output_text"])
        records[model].append(
            {
                **_base_record(case, trial_id, grade),
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


def run_bfcl_baseline_matrix(
    *,
    manifest_path: str | Path = "benchmarks/manifests/bfcl.json",
    config_path: str | Path = "compound.yaml",
    source_dir: str | Path = DEFAULT_BFCL_SOURCE_DIR,
    models: list[str] | None = None,
    case_ids: list[str] | None = None,
    max_output_tokens: int = 4096,
    reasoning_effort: str | None = None,
    trials_per_case: int = 1,
    experiment_cap_usd: float = 4.0,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 3600.0,
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

    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("benchmark") != "bfcl":
        raise ValueError("BFCL baseline matrix requires a bfcl manifest")
    cases = load_bfcl_cases(source_dir, manifest_path)
    graded_cases, skipped_cases = select_single_turn_cases(cases, case_ids)
    if not graded_cases:
        raise ValueError("no single_turn cases selected")

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
        cases=graded_cases,
        ledger=ledger,
        cap=cap,
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
            cases=graded_cases,
            ledger=ledger,
            cap=cap,
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
        "benchmark": "bfcl",
        "benchmark_manifest": str(manifest_path),
        "bfcl_eval_package": config["benchmarks"]["bfcl"].get("package"),
        "source_revision": manifest.get("revision"),
        "case_ids": [case.case_id for case in graded_cases],
        "graded_strata": [SINGLE_TURN_STRATUM],
        "skipped_cases": {
            "case_ids": [case.case_id for case in skipped_cases],
            "stratum": MULTI_TURN_STRATUM,
            "reason": MULTI_TURN_LIMITATION,
        },
        "trials_per_case": trials_per_case,
        "max_output_tokens": max_output_tokens,
        "candidate_reasoning_effort": reasoning_effort,
        "experiment_cap_usd": experiment_cap_usd,
        "budget_before_usd": budget_before,
        "budget_after_usd": ledger.spent_usd,
        "incremental_cost_usd": ledger.spent_usd - budget_before,
        "remaining_budget_usd": ledger.remaining_usd,
        "grading_note": (
            "Completions are rendered with BFCL's default prompting-mode system "
            "prompt and graded with the official bfcl-eval AST checker "
            "(default_decode_ast_prompting + ast_checker) against the pinned "
            "possible-answer files."
        ),
        "route_note": (
            "Reference models use OpenRouter's default Chat Completions route. Candidate "
            "models use Doubleword background Responses with requested service_tier=flex. "
            f"Candidate reasoning effort is {reasoning_effort or 'provider default'}. "
            "Candidate latency and TPS include async queueing."
        ),
        "models": model_results,
    }
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = artifacts / "baselines" / f"bfcl-matrix-{timestamp}.json"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output
