import pytest

from compound.config import load_config, require_paid_run_budget


def test_default_config_is_valid() -> None:
    config = load_config("compound.yaml")
    assert sum(item["sample_count"] for item in config["benchmarks"].values()) == 80
    doubleword_models = {
        item["id"]
        for item in config["models"]["candidates"]
        if item["provider"] == "doubleword"
    }
    assert doubleword_models == {
        "thinkingmachines/Inkling-NVFP4",
        "zai-org/GLM-5.2-FP8",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "deepseek-ai/DeepSeek-V4-Flash",
    }
    assert doubleword_models == set(config["flex_pricing_usd_per_million_tokens"])


def test_configured_paid_budget_is_available() -> None:
    config = load_config("compound.yaml")
    assert require_paid_run_budget(config) == 25.0


def test_disabled_paid_calls_are_rejected() -> None:
    config = {"budget": {"paid_runs_enabled": False, "hard_limit_usd": 25.0}}
    with pytest.raises(RuntimeError, match="disabled"):
        require_paid_run_budget(config)
