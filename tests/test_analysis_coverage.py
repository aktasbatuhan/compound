"""Reporting consumers must not dilute measured rates with missing evidence."""

import importlib.util
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arm_rates_use_matching_evidence_and_unknown_cost_stays_unknown():
    analyze = script("analyze_arms")
    row = analyze.summarize_arm("a", [
        {"status": 200, "prompt_tokens": 100, "cached_tokens": 80, "cost_usd": 0.01},
        {"status": 200, "prompt_tokens": 900, "cached_tokens": None, "cost_usd": None},
    ])
    assert row["cache_ratio"] == 0.8
    assert row["cost_per_1k_prompt"] == pytest.approx(0.1)
    assert row["priced_calls"] == 1
    unknown = analyze.summarize_arm("a", [{"status": 200, "prompt_tokens": 100}])
    assert unknown["cost_usd"] is None
    assert unknown["cost_per_1k_prompt"] is None
    assert unknown["cache_ratio"] is None


def test_serving_table_discloses_partial_cost_and_uses_reported_cache(capsys):
    analyze = script("analyze_serving")
    analyze.speed_table([
        {"route": "a", "status": 200, "prompt_tokens": 100,
         "cached_tokens": 80, "cost_usd": 0.01},
        {"route": "a", "status": 200, "prompt_tokens": 900},
    ], {})
    output = capsys.readouterr().out
    assert "100.0000*1/2" in output
    row = next(line for line in output.splitlines() if line.startswith("  a "))
    assert row.split()[-2] == "80"
