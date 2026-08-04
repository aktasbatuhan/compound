import json

from compound.adapters.mmlu import (
    PARTITION_WEIGHTS,
    build_manifest,
    grade,
    parse_letter,
    render_prompt,
    stable_partition,
)


def _case(answer=0):
    return {
        "case_id": "mmlu:anatomy:0",
        "metadata": {
            "question": "Q?",
            "choices": ["w", "x", "y", "z"],
            "answer": answer,
        },
    }


def test_parse_letter_takes_the_final_verdict() -> None:
    # Reasoning traces mention wrong options first; the conclusion wins.
    assert parse_letter("A is tempting but wrong. The answer is C.") == "C"
    assert parse_letter("B") == "B"
    assert parse_letter("no letter here") is None


def test_grade_and_prompt() -> None:
    assert grade(_case(answer=2), "... therefore C") is True
    assert grade(_case(answer=2), "A") is False
    prompt = render_prompt(_case())
    assert "A. w" in prompt and "D. z" in prompt


def test_partition_split_is_stable_and_proportioned() -> None:
    ids = [f"mmlu:sub:{i}" for i in range(2000)]
    assignments = [stable_partition(i) for i in ids]
    # Re-running yields identical assignments (hash-based, order-free).
    assert assignments == [stable_partition(i) for i in ids]
    from collections import Counter

    counts = Counter(assignments)
    assert set(counts) == {name for name, _ in PARTITION_WEIGHTS}
    # decision_test is ~30% and never empty.
    assert 0.25 < counts["decision_test"] / len(ids) < 0.35


def test_build_manifest_offline(tmp_path) -> None:
    def fake_fetch(url: str) -> dict:
        if "splits" in url:
            return {"splits": [
                {"config": "anatomy"}, {"config": "all"}, {"config": "auxiliary_train"},
            ]}
        return {"rows": [
            {"row_idx": 0, "row": {"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 1}},
        ]}

    out = build_manifest(tmp_path / "mmlu.json", per_subject=1, fetch=fake_fetch)
    manifest = json.loads(out.read_text())
    # "all" and "auxiliary_train" are dropped; only real subjects survive.
    assert [c["metadata"]["subject"] for c in manifest["cases"]] == ["anatomy"]
    assert manifest["cases"][0]["case_id"] == "mmlu:anatomy:0"
    assert manifest["cases"][0]["partition"] in {n for n, _ in PARTITION_WEIGHTS}
