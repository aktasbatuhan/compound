from compound.adapters.ds1000 import build_test_program, postprocess_completion


def test_ds1000_postprocessing_matches_official_rules() -> None:
    assert postprocess_completion("```python\nanswer = 1\n```\nextra") == "answer = 1"


def test_ds1000_program_uses_repr_not_raw_interpolation() -> None:
    case = {"code_context": "def test_execution(code):\n    assert code", "case_id": "1"}
    program = build_test_program(case, "x = 'quoted'")
    assert "code = \"x = 'quoted'\"" in program
    assert program.endswith("test_execution(code)\n")
