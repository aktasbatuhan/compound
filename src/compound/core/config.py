from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "compound.yaml") -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("compound.yaml must be a version 1 mapping")

    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("at least one benchmark must be configured")
    for name, benchmark in benchmarks.items():
        partitions = benchmark.get("partitions", {})
        total = sum(partitions.values())
        if total != benchmark.get("sample_count"):
            raise ValueError(f"{name}: partition sizes do not match sample_count")
    budget = data.get("budget", {})
    if budget.get("paid_runs_enabled") and not isinstance(
        budget.get("hard_limit_usd"), (int, float)
    ):
        raise ValueError("paid runs require a numeric budget.hard_limit_usd")
    flex_prices = data.get("flex_pricing_usd_per_million_tokens", {})
    for model in data.get("models", {}).get("candidates", []):
        if model.get("provider") != "doubleword":
            continue
        model_id = model.get("id")
        prices = flex_prices.get(model_id)
        if not isinstance(prices, dict) or not all(
            isinstance(prices.get(key), (int, float)) and prices[key] >= 0
            for key in ("input", "output")
        ):
            raise ValueError(f"{model_id}: numeric Doubleword Flex pricing is required")
    return data


def require_paid_run_budget(config: dict[str, Any]) -> float:
    budget = config.get("budget", {})
    if not budget.get("paid_runs_enabled"):
        raise RuntimeError("paid model calls are disabled in compound.yaml")
    limit = budget.get("hard_limit_usd")
    if not isinstance(limit, (int, float)) or limit <= 0:
        raise RuntimeError("a positive budget.hard_limit_usd is required")
    return float(limit)
