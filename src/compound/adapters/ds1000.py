from __future__ import annotations

import gzip
import json
from pathlib import Path

from compound.contracts import Partition


def postprocess_completion(code: str | list[str]) -> str:
    """Apply the official DS-1000 completion cleanup rules."""
    if isinstance(code, list):
        code = code[0]
    code = code.split("</code>", 1)[0]
    code = code.replace("```python", "")
    code = code.split("```", 1)[0]
    code = code.split("\nEND SOLUTION", 1)[0]
    return code.replace("<code>", "").strip()


def load_ds1000_cases(
    source_path: str | Path,
    manifest_path: str | Path,
    *,
    partitions: set[Partition],
) -> list[dict]:
    """Load only allowed cases; decision-test content is never loaded during optimization."""
    manifest = json.loads(Path(manifest_path).read_text())
    allowed_partitions = {partition.value for partition in partitions}
    selected = {
        int(case["metadata"]["problem_id"]): case
        for case in manifest["cases"]
        if case["partition"] in allowed_partitions
    }
    cases: list[dict] = []
    with gzip.open(source_path, "rt") as stream:
        for line in stream:
            source = json.loads(line)
            problem_id = int(source["metadata"]["problem_id"])
            if problem_id not in selected:
                continue
            cases.append(
                {
                    "case_id": f"ds1000_{problem_id}",
                    "partition": selected[problem_id]["partition"],
                    "prompt": source["prompt"],
                    "code_context": source["code_context"],
                    "metadata": source["metadata"],
                }
            )
    return cases


def build_test_program(case: dict, completion: str) -> str:
    """Build the official evaluator program; callers must execute it in a real sandbox."""
    cleaned = postprocess_completion(completion)
    suffix = "test_execution(code)\n"
    if "test_string(" in case["code_context"]:
        suffix += "test_string(code)\n"
    return f"{case['code_context']}\ncode = {cleaned!r}\n{suffix}"
