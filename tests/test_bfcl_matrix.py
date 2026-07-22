import pytest

from compound.adapters.bfcl import BFCLCase
from compound.bfcl_matrix import _model_summary, select_single_turn_cases
from compound.contracts import Partition


def _case(case_id: str, stratum: str, category: str) -> BFCLCase:
    return BFCLCase(
        case_id=case_id,
        category=category,
        stratum=stratum,
        partition=Partition.OPTIMIZER_TRAIN,
        question=[[{"role": "user", "content": "hi"}]],
        functions=[],
        ground_truth=None,
    )


def _cases() -> list[BFCLCase]:
    return [
        _case("simple_python_1", "single_turn", "simple_python"),
        _case("multiple_1", "single_turn", "multiple"),
        _case("multi_turn_base_1", "multi_turn", "multi_turn_base"),
    ]


def test_select_single_turn_cases_splits_gradable_from_skipped() -> None:
    graded, skipped = select_single_turn_cases(_cases(), None)

    assert [case.case_id for case in graded] == ["simple_python_1", "multiple_1"]
    assert [case.case_id for case in skipped] == ["multi_turn_base_1"]


def test_select_single_turn_cases_filters_requested_ids() -> None:
    graded, skipped = select_single_turn_cases(_cases(), ["multiple_1"])

    assert [case.case_id for case in graded] == ["multiple_1"]
    assert [case.case_id for case in skipped] == ["multi_turn_base_1"]


def test_select_single_turn_cases_rejects_multi_turn_and_unknown_ids() -> None:
    with pytest.raises(ValueError, match="multi_turn cases cannot be graded"):
        select_single_turn_cases(_cases(), ["multi_turn_base_1"])
    with pytest.raises(ValueError, match="not in the BFCL manifest"):
        select_single_turn_cases(_cases(), ["simple_python_999"])


def test_model_summary_distinguishes_cases_from_repeated_trials() -> None:
    records = [
        {
            "case_id": case_id,
            "category": "simple_python",
            "passed": passed,
            "error_type": None if passed else "value_error:string",
            "latency_ms": 100,
            "output_tokens": 10,
            "input_tokens": 20,
            "reasoning_tokens": 5,
            "estimated_cost_usd": 0.001,
        }
        for case_id, passed in [("a", True), ("a", False), ("b", True), ("b", False)]
    ]

    summary = _model_summary(records)

    assert summary["cases"] == 2
    assert summary["trials"] == 4
    assert summary["passed_trials"] == 2
    assert summary["case_success_rates"] == {"a": 0.5, "b": 0.5}
    assert summary["by_category"]["simple_python"] == {"cases": 2, "trials": 4, "passed": 2}
    assert summary["error_types"] == {"value_error:string": 2}
