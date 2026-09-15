"""Fetch pinned official SWE grading metadata without changing frozen tasks."""

import json
from pathlib import Path

from datasets import load_dataset

DATASET = "SWE-bench/SWE-bench_Verified"
REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"


def main():
    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    frozen = json.loads(Path(spec["sources"]["coding"]["data_path"]).read_text())
    upstream = {r["instance_id"]: r for r in load_dataset(DATASET, split="test", revision=REVISION)}
    selected = []
    for task in frozen:
        row = upstream[task["instance_id"]]
        for key in ("problem_statement", "patch", "test_patch", "base_commit"):
            if row[key] != task[key]:
                raise ValueError(f"Frozen task changed: {task['instance_id']} {key}")
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            a, b = row[key], task[key]
            a = json.loads(a) if isinstance(a, str) else a
            b = json.loads(b) if isinstance(b, str) else b
            if set(a) != set(b):
                raise ValueError(f"Frozen tests changed: {task['instance_id']} {key}")
        selected.append(row)
    target = Path(".compound/sources/swe-verified-grader.json")
    target.write_text(json.dumps(selected))
    print(f"Verified {len(selected)} tasks; wrote {target}; source {DATASET}@{REVISION}")


if __name__ == "__main__":
    main()
