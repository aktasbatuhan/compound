from compound.baseline_matrix import _model_summary


def test_model_summary_distinguishes_cases_from_repeated_trials() -> None:
    records = [
        {
            "case_id": case_id,
            "library": "Numpy",
            "passed": passed,
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
    assert summary["by_library"]["Numpy"] == {"cases": 2, "trials": 4, "passed": 2}
