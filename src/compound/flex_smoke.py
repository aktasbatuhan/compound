from __future__ import annotations

import gzip
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compound.adapters.ds1000 import build_test_program, postprocess_completion
from compound.budget import BudgetLedger
from compound.config import load_config, require_paid_run_budget
from compound.costs import TokenPrices
from compound.ds1000_optimizer import DEFAULT_SEED_PROMPT, grade_fingerprint
from compound.env import load_env
from compound.flex import CachedFlexRunner, FlexRequest
from compound.providers import OpenAICompatibleProvider

DEFAULT_EVALUATOR_IMAGE = "compound-ds1000-numpy:20260720-v3"
DEFAULT_CASE_ID = "ds1000_428"


def load_ds1000_case(source_path: str | Path, case_id: str) -> dict[str, Any]:
    try:
        problem_id = int(case_id.removeprefix("ds1000_"))
    except ValueError as error:
        raise ValueError(f"invalid DS-1000 case id: {case_id}") from error
    with gzip.open(source_path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if int(record["metadata"]["problem_id"]) == problem_id:
                return {
                    "case_id": case_id,
                    "prompt": record["prompt"],
                    "code_context": record["code_context"],
                    "metadata": record["metadata"],
                }
    raise KeyError(f"DS-1000 case not found: {case_id}")


def grade_ds1000_completion(
    *,
    case: dict[str, Any],
    completion: str,
    evaluator_image: str,
    cache_dir: str | Path,
) -> dict[str, Any]:
    fingerprint = grade_fingerprint(
        completion=completion,
        evaluator_image=evaluator_image,
    )
    cache_path = Path(cache_dir) / f"{fingerprint}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if not completion.strip():
        result = {
            "score": 0.0,
            "feedback": "Model returned no final completion.",
            "grader_latency_ms": 0,
        }
    else:
        program = build_test_program(
            {"code_context": case["code_context"]},
            postprocess_completion(completion),
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
        grader_latency_ms = round((time.perf_counter() - started) * 1000)
        if process.returncode != 0:
            result = {
                "score": 0.0,
                "feedback": f"Evaluator container failed: {process.stderr.strip()}",
                "grader_latency_ms": grader_latency_ms,
            }
        else:
            try:
                evaluated = json.loads(process.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                evaluated = {
                    "passed": False,
                    "feedback": f"Evaluator returned invalid output: {process.stdout[-500:]}",
                }
            result = {
                "score": float(bool(evaluated.get("passed"))),
                "feedback": str(evaluated.get("feedback") or "grader did not provide feedback"),
                "grader_latency_ms": grader_latency_ms,
            }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def flex_prices(config: dict[str, Any], model: str) -> TokenPrices:
    values = config["flex_pricing_usd_per_million_tokens"][model]
    return TokenPrices(
        input_per_million=float(values["input"]),
        output_per_million=float(values["output"]),
    )


def run_doubleword_flex_smoke(
    *,
    config_path: str | Path = "compound.yaml",
    source_path: str | Path = ".compound/sources/ds1000/data/ds1000.jsonl.gz",
    case_id: str = DEFAULT_CASE_ID,
    models: list[str] | None = None,
    max_output_tokens: int = 4096,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 3600.0,
    evaluator_image: str = DEFAULT_EVALUATOR_IMAGE,
    output_path: str | Path | None = None,
) -> Path:
    load_env()
    config = load_config(config_path)
    hard_limit = require_paid_run_budget(config)
    case = load_ds1000_case(source_path, case_id)
    configured_models = [
        item["id"] for item in config["models"]["candidates"] if item["provider"] == "doubleword"
    ]
    selected_models = models or configured_models
    unknown = sorted(set(selected_models) - set(configured_models))
    if unknown:
        raise ValueError(f"models are not configured as Doubleword candidates: {unknown}")

    artifacts = Path(config["artifacts_dir"])
    provider_values = config["providers"]["doubleword"]
    provider = OpenAICompatibleProvider(
        name="doubleword",
        base_url=provider_values["base_url"],
        api_key_env=provider_values["api_key_env"],
        telemetry_path=artifacts / "telemetry" / "model_calls.jsonl",
        timeout_seconds=60,
        max_retries=2,
    )
    ledger = BudgetLedger.load(artifacts / "budget.json", hard_limit)
    prices_by_model = {model: flex_prices(config, model) for model in selected_models}
    runner = CachedFlexRunner(
        provider=provider,
        prices_by_model=prices_by_model,
        ledger=ledger,
        cache_dir=artifacts / "cache" / "flex-responses",
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    requests = [
        FlexRequest(
            model=model,
            input=case["prompt"],
            instructions=DEFAULT_SEED_PROMPT,
            max_output_tokens=max_output_tokens,
            telemetry_context={
                "benchmark": "ds1000",
                "case_id": case_id,
                "call_type": "candidate_flex_smoke",
            },
        )
        for model in selected_models
    ]
    responses = runner.run_many(requests)
    model_results = []
    for response in responses:
        evaluation = grade_ds1000_completion(
            case=case,
            completion=response["output_text"],
            evaluator_image=evaluator_image,
            cache_dir=artifacts / "cache" / "ds1000" / "grades",
        )
        model_results.append(
            {
                **response,
                "evaluation": {
                    "passed": evaluation["score"] == 1.0,
                    **evaluation,
                },
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "ds1000",
        "case_id": case_id,
        "library": case["metadata"].get("library"),
        "requested_service_tier": "flex",
        "background": True,
        "billing_note": (
            "Costs are estimated from the configured published Doubleword async rates. "
            "The response-level tier confirmation field records whether the API echoed flex."
        ),
        "latency_note": (
            "e2e_latency_ms and e2e_output_tps include async queueing; submit_latency_ms "
            "measures only the background-job acknowledgement."
        ),
        "evaluator_image": evaluator_image,
        "system_prompt": DEFAULT_SEED_PROMPT,
        "max_output_tokens": max_output_tokens,
        "models": model_results,
        "summary": {
            "models": len(model_results),
            "passed": sum(result["evaluation"]["passed"] for result in model_results),
            "estimated_cost_usd": sum(result["estimated_cost_usd"] for result in model_results),
        },
    }
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = artifacts / "smoke" / f"ds1000-flex-{case_id}-{timestamp}.json"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output
