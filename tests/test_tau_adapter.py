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
