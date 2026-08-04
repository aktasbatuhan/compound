import json

from compound.adapters.tau import TauModel, task_ids_by_domain
from compound.contracts import Partition


def test_tau_model_routes_openrouter_and_doubleword() -> None:
    assert TauModel("openrouter", "openai/gpt-5.6-sol").litellm_name() == (
        "openrouter/openai/gpt-5.6-sol"
    )
    target = TauModel("doubleword", "zai-org/GLM-5.2-FP8", "https://api.doubleword.ai/v1")
    assert target.litellm_name() == "openai/zai-org/GLM-5.2-FP8"
    assert target.llm_args()["api_base"] == "https://api.doubleword.ai/v1"
    assert "temperature" not in target.llm_args()
    low = TauModel("doubleword", "model", reasoning_effort="low")
    assert low.llm_args()["reasoning_effort"] == "low"
    longer = TauModel("doubleword", "model", max_tokens=4096)
    assert longer.llm_args()["max_tokens"] == 4096


def test_tau_model_pins_openrouter_upstream() -> None:
    pinned = TauModel("openrouter", "moonshotai/kimi-k3", openrouter_provider="baseten/fp8")
    assert pinned.llm_args()["extra_body"] == {
        "provider": {"only": ["baseten/fp8"], "allow_fallbacks": False}
    }
    # The pin joins the output identity, so two upstreams never share a file.
    assert pinned.slug() == "moonshotai--kimi-k3--at-baseten-fp8"
    assert TauModel("openrouter", "moonshotai/kimi-k3").slug() == "moonshotai--kimi-k3"
    assert "extra_body" not in TauModel("openrouter", "m").llm_args()

    import pytest

    with pytest.raises(ValueError, match="openrouter"):
        TauModel("doubleword", "m", openrouter_provider="baseten/fp8").llm_args()


def test_tau_model_doubleword_tiers() -> None:
    realtime = TauModel("doubleword", "zai-org/GLM-5.2-FP8", "https://api.doubleword.ai/v1")
    assert realtime.slug() == "zai-org--GLM-5.2-FP8--at-doubleword-realtime"
    assert "extra_body" not in realtime.llm_args()
    # A tier-flagged run rides extra_body; it stays UNVERIFIED until billing says so.
    flagged = TauModel("doubleword", "m", "https://x", service_tier="flex")
    assert flagged.llm_args()["extra_body"] == {"service_tier": "flex"}
    assert flagged.slug() == "m--at-doubleword-flex"


def test_tau_manifest_groups_ids_by_domain(tmp_path) -> None:
    manifest = {
        "cases": [
            {"case_id": "retail:1", "partition": "optimizer_train"},
            {"case_id": "airline:2", "partition": "optimizer_train"},
            {"case_id": "retail:3", "partition": "decision_test"},
        ]
    }
    path = tmp_path / "tau.json"
    path.write_text(json.dumps(manifest))
    assert task_ids_by_domain(path, Partition.OPTIMIZER_TRAIN) == {
        "retail": ["1"],
        "airline": ["2"],
    }
    # partition=None selects the whole manifest — the sweep's full-corpus mode.
    assert task_ids_by_domain(path, None) == {
        "retail": ["1", "3"],
        "airline": ["2"],
    }
