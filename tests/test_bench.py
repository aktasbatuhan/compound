import json

import pytest

from compound.bench import BENCHMARKS, select_case_ids
from compound.tau_sweep import SweepConfig


CASES = [
    {"case_id": "retail:10", "partition": "optimizer_validation"},
    {"case_id": "retail:33", "partition": "optimizer_train"},
    {"case_id": "airline:3", "partition": "optimizer_train"},
]


def test_select_case_ids_filters_and_validates() -> None:
    assert select_case_ids(CASES) == ["retail:10", "retail:33", "airline:3"]
    assert select_case_ids(CASES, partition="optimizer_train") == ["retail:33", "airline:3"]
    assert select_case_ids(CASES, contains="RETAIL") == ["retail:10", "retail:33"]
    # Explicit ids bypass filters but must exist in the manifest.
    assert select_case_ids(CASES, explicit=["airline:3"]) == ["airline:3"]
    with pytest.raises(SystemExit):
        select_case_ids(CASES, explicit=["bogus:1"])


def test_registry_manifests_are_partitioned() -> None:
    for bench in BENCHMARKS.values():
        cases = json.loads(bench.manifest.read_text())["cases"]
        assert cases, bench.name
        assert all("case_id" in c and "partition" in c for c in cases), bench.name


def test_sweep_config_custom_provider_key_env() -> None:
    custom = SweepConfig("m", "up", "fp8", 1.0, 2.0, provider="myhost", api_key_env="MY_KEY")
    assert custom.required_key_env() == "MY_KEY"
    assert SweepConfig("m", "up", "q", 1, 2).required_key_env() == "OPENROUTER_API_KEY"
    assert (
        SweepConfig("m", "up", "q", 1, 2, provider="doubleword").required_key_env()
        == "DOUBLEWORD_API_KEY"
    )
    with pytest.raises(ValueError):
        SweepConfig("m", "up", "q", 1, 2, provider="myhost").required_key_env()
