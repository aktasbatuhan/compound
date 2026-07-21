from unittest.mock import patch

import pytest

from compound.contracts import Partition
from compound.manifest import SourceCase, assign_partitions, stratified_sample, write_manifest
from compound.prepare import (
    open_ds1000_decision_manifest_for_diagnostics,
    prepare_ds1000_gepa_decision_manifest,
    prepare_gepa_manifest_from_opened_baseline,
    prepare_gepa_v2_ds1000_manifest,
)


def test_stratified_sample_is_stable_and_balanced() -> None:
    cases = [SourceCase(f"a-{i}", "a") for i in range(10)] + [
        SourceCase(f"b-{i}", "b") for i in range(10)
    ]
    first = stratified_sample(cases, benchmark="demo", seed=7, count=6)
    second = stratified_sample(reversed(cases), benchmark="demo", seed=7, count=6)
    assert first == second
    assert [case.stratum for case in first] == ["a", "b", "a", "b", "a", "b"]


def test_partition_assignment_is_exact() -> None:
    cases = [SourceCase(str(i), "all") for i in range(8)]
    result = assign_partitions(cases, train=4, validation=2, decision=2)
    assert list(result.values()).count(Partition.OPTIMIZER_TRAIN) == 4
    assert list(result.values()).count(Partition.OPTIMIZER_VALIDATION) == 2
    assert list(result.values()).count(Partition.DECISION_TEST) == 2


def test_partition_assignment_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        assign_partitions([SourceCase("1", "all")], train=1, validation=1, decision=0)


def test_manifest_does_not_embed_case_inputs_or_answers(tmp_path) -> None:
    cases = [SourceCase("1", "a", {"category": "simple"})]
    output = tmp_path / "manifest.json"
    write_manifest(
        output,
        benchmark="demo",
        revision="abc123",
        seed=7,
        cases=cases,
        partitions={"1": Partition.DECISION_TEST},
    )
    content = output.read_text()
    assert '"case_id": "1"' in content
    assert "prompt" not in content
    assert "answer" not in content


@patch("compound.prepare.load_ds1000_catalog")
def test_gepa_v2_manifest_keeps_perturbation_groups_isolated(load_catalog, tmp_path) -> None:
    load_catalog.return_value = [
        SourceCase(
            f"{library}-{origin}-{variant}",
            library,
            {
                "problem_id": index,
                "perturbation_origin_id": origin,
                "perturbation_type": variant,
            },
        )
        for index, (library, origin, variant) in enumerate(
            [
                ("numpy", 1, "Origin"),
                ("numpy", 1, "Semantic"),
                ("numpy", 2, "Origin"),
                ("numpy", 3, "Surface"),
                ("pandas", 4, "Origin"),
                ("pandas", 4, "Difficult-Rewrite"),
                ("pandas", 5, "Origin"),
                ("pandas", 6, "Semantic"),
            ]
        )
    ]
    output = tmp_path / "manifest.json"

    prepare_gepa_v2_ds1000_manifest(
        source_dir=tmp_path,
        output_path=output,
        revision="revision",
        seed=7,
        train=2,
        validation=1,
        decision=1,
        excluded_case_ids=set(),
        excluded_origin_groups={"numpy:3", "pandas:6"},
    )

    payload = __import__("json").loads(output.read_text())
    groups = [
        f"{case['stratum']}:{case['metadata']['perturbation_origin_id']}"
        for case in payload["cases"]
    ]
    assert len(groups) == len(set(groups)) == 4
    assert "numpy:3" not in groups
    assert "pandas:6" not in groups


def test_opened_baseline_is_repartitioned_without_creating_a_decision_set(tmp_path) -> None:
    source = tmp_path / "baseline.json"
    source.write_text(
        __import__("json").dumps(
            {
                "benchmark": "ds1000",
                "cohort_role": "opened_baseline",
                "cases": [
                    {"case_id": f"case-{index}", "partition": "optimizer_train"}
                    for index in range(4)
                ],
            }
        )
    )
    output = tmp_path / "gepa.json"

    prepare_gepa_manifest_from_opened_baseline(
        baseline_manifest_path=source,
        output_path=output,
        train=2,
        validation=2,
    )

    payload = __import__("json").loads(output.read_text())
    assert payload["cohort_role"] == "opened_optimization"
    assert [case["partition"] for case in payload["cases"]] == [
        "optimizer_train",
        "optimizer_train",
        "optimizer_validation",
        "optimizer_validation",
    ]
    assert payload["decision_cases"] == 0


@patch("compound.prepare.load_ds1000_catalog")
def test_decision_manifest_contains_only_sealed_decision_cases(load_catalog, tmp_path) -> None:
    load_catalog.return_value = [
        SourceCase(
            f"case-{index}",
            "numpy" if index % 2 == 0 else "pandas",
            {"perturbation_origin_id": index},
        )
        for index in range(4)
    ]
    output = tmp_path / "decision.json"

    prepare_ds1000_gepa_decision_manifest(
        source_dir=tmp_path,
        output_path=output,
        revision="revision",
        seed=7,
        count=4,
        excluded_case_ids=set(),
        excluded_origin_groups=set(),
    )

    payload = __import__("json").loads(output.read_text())
    assert payload["cohort_role"] == "sealed_decision"
    assert {case["partition"] for case in payload["cases"]} == {"decision_test"}


def test_completed_decision_manifest_can_be_opened_for_diagnostics(tmp_path) -> None:
    source = tmp_path / "decision.json"
    source.write_text(
        __import__("json").dumps(
            {
                "cohort_role": "sealed_decision",
                "cases": [{"case_id": "case-1", "partition": "decision_test"}],
            }
        )
    )
    report = tmp_path / "report.json"
    report.write_text(
        __import__("json").dumps(
            {
                "decision_manifest": str(source),
                "case_ids": ["case-1"],
                "optimizer_accessed_decision_cases": False,
            }
        )
    )
    output = tmp_path / "opened.json"

    open_ds1000_decision_manifest_for_diagnostics(
        decision_manifest_path=source,
        decision_report_path=report,
        output_path=output,
    )

    payload = __import__("json").loads(output.read_text())
    assert payload["cohort_role"] == "opened_baseline"
    assert payload["cases"][0]["partition"] == "optimizer_train"
    assert payload["source_decision_report"] == str(report)
