import json

import pytest

from compound.adapters.bfcl import (
    BFCLCase,
    grade_bfcl_single_turn,
    load_bfcl_cases,
    render_bfcl_single_turn_prompt,
    write_bfcl_run_ids,
)
from compound.contracts import Partition

TRIANGLE_FUNCTION = {
    "name": "calculate_triangle_area",
    "description": "Calculate the area of a triangle given its base and height.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer", "description": "The base of the triangle."},
            "height": {"type": "integer", "description": "The height of the triangle."},
            "unit": {"type": "string", "description": "The unit of measure."},
        },
        "required": ["base", "height"],
    },
}
TRIANGLE_GROUND_TRUTH = [
    {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
]


def _triangle_case(**overrides) -> BFCLCase:
    values = {
        "case_id": "simple_python_1",
        "category": "simple_python",
        "stratum": "single_turn",
        "partition": Partition.OPTIMIZER_TRAIN,
        "question": [[{"role": "user", "content": "Area of a triangle with base 10, height 5?"}]],
        "functions": [TRIANGLE_FUNCTION],
        "ground_truth": TRIANGLE_GROUND_TRUTH,
    }
    values.update(overrides)
    return BFCLCase(**values)


def _write_source_tree(tmp_path) -> str:
    data_dir = (
        tmp_path / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    )
    (data_dir / "possible_answer").mkdir(parents=True)
    simple_entry = {
        "id": "simple_python_1",
        "question": [[{"role": "user", "content": "Area of a triangle?"}]],
        "function": [TRIANGLE_FUNCTION],
    }
    (data_dir / "BFCL_v4_simple_python.json").write_text(json.dumps(simple_entry) + "\n")
    (data_dir / "possible_answer" / "BFCL_v4_simple_python.json").write_text(
        json.dumps({"id": "simple_python_1", "ground_truth": TRIANGLE_GROUND_TRUTH}) + "\n"
    )
    multi_entry = {
        "id": "multi_turn_base_1",
        "question": [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "next"}],
        ],
        "initial_config": {},
        "involved_classes": ["GorillaFileSystem"],
    }
    (data_dir / "BFCL_v4_multi_turn_base.json").write_text(json.dumps(multi_entry) + "\n")
    return str(tmp_path / "gorilla")


def _write_manifest(tmp_path, cases: list[dict]) -> str:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"benchmark": "bfcl", "cases": cases}))
    return str(manifest)


def test_bfcl_run_ids_exclude_decision_set(tmp_path) -> None:
    manifest = {
        "cases": [
            {
                "case_id": "simple_python_1",
                "partition": "optimizer_train",
                "metadata": {"category": "simple_python"},
            },
            {
                "case_id": "simple_python_2",
                "partition": "decision_test",
                "metadata": {"category": "simple_python"},
            },
        ]
    }
    source = tmp_path / "manifest.json"
    output = tmp_path / "ids.json"
    source.write_text(json.dumps(manifest))
    payload = write_bfcl_run_ids(
        source,
        output,
        partitions={Partition.OPTIMIZER_TRAIN},
    )
    assert payload == {"simple_python": ["simple_python_1"]}
    assert "simple_python_2" not in output.read_text()


def test_load_bfcl_cases_joins_prompts_and_ground_truth(tmp_path) -> None:
    source_dir = _write_source_tree(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "case_id": "multi_turn_base_1",
                "partition": "optimizer_train",
                "stratum": "multi_turn",
                "metadata": {"category": "multi_turn_base"},
            },
            {
                "case_id": "simple_python_1",
                "partition": "decision_test",
                "stratum": "single_turn",
                "metadata": {"category": "simple_python"},
            },
        ],
    )

    cases = load_bfcl_cases(source_dir, manifest_path)

    assert [case.case_id for case in cases] == ["multi_turn_base_1", "simple_python_1"]
    simple = cases[1]
    assert simple.partition is Partition.DECISION_TEST
    assert simple.functions == [TRIANGLE_FUNCTION]
    assert simple.ground_truth == TRIANGLE_GROUND_TRUTH
    multi = cases[0]
    assert multi.stratum == "multi_turn"
    assert multi.functions == []
    assert multi.ground_truth is None


def test_load_bfcl_cases_filters_strata_and_rejects_unknown_ids(tmp_path) -> None:
    source_dir = _write_source_tree(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "case_id": "simple_python_1",
                "partition": "optimizer_train",
                "stratum": "single_turn",
                "metadata": {"category": "simple_python"},
            },
            {
                "case_id": "multi_turn_base_1",
                "partition": "optimizer_train",
                "stratum": "multi_turn",
                "metadata": {"category": "multi_turn_base"},
            },
        ],
    )

    single_only = load_bfcl_cases(source_dir, manifest_path, strata={"single_turn"})
    assert [case.case_id for case in single_only] == ["simple_python_1"]

    missing_manifest = _write_manifest(
        tmp_path,
        [
            {
                "case_id": "simple_python_999",
                "partition": "optimizer_train",
                "stratum": "single_turn",
                "metadata": {"category": "simple_python"},
            }
        ],
    )
    with pytest.raises(KeyError, match="simple_python_999"):
        load_bfcl_cases(source_dir, missing_manifest)


def test_render_bfcl_single_turn_prompt_uses_official_template() -> None:
    pytest.importorskip("bfcl_eval")
    case = _triangle_case()

    system_prompt, user_prompt = render_bfcl_single_turn_prompt(case)

    assert "You are an expert in composing functions." in system_prompt
    assert "calculate_triangle_area" in system_prompt
    assert user_prompt == "Area of a triangle with base 10, height 5?"

    multi_turn = _triangle_case(stratum="multi_turn")
    with pytest.raises(ValueError, match="not single-turn"):
        render_bfcl_single_turn_prompt(multi_turn)


def test_grade_bfcl_single_turn_uses_official_checker() -> None:
    pytest.importorskip("bfcl_eval")
    case = _triangle_case()

    passed = grade_bfcl_single_turn(case, "[calculate_triangle_area(base=10, height=5)]")
    assert passed["score"] == 1.0
    assert passed["error_type"] is None

    wrong_value = grade_bfcl_single_turn(case, "[calculate_triangle_area(base=12, height=5)]")
    assert wrong_value["score"] == 0.0
    assert wrong_value["error_type"].startswith("value_error")

    prose = grade_bfcl_single_turn(case, "I cannot call any function here.")
    assert prose["score"] == 0.0
    assert prose["error_type"] == "ast_decoder:decoder_failed"

    empty = grade_bfcl_single_turn(case, "   ")
    assert empty["score"] == 0.0
    assert empty["error_type"] == "empty_completion"


def test_grade_bfcl_single_turn_requires_ground_truth() -> None:
    pytest.importorskip("bfcl_eval")
    case = _triangle_case(ground_truth=None)
    with pytest.raises(ValueError, match="ground truth"):
        grade_bfcl_single_turn(case, "[calculate_triangle_area(base=10, height=5)]")
