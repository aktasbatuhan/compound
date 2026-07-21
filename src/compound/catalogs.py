from __future__ import annotations

import gzip
import json
from pathlib import Path

from compound.manifest import SourceCase


def load_ds1000_catalog(source_dir: str | Path) -> list[SourceCase]:
    data_path = Path(source_dir) / "data" / "ds1000.jsonl.gz"
    cases: list[SourceCase] = []
    with gzip.open(data_path, "rt") as stream:
        for line in stream:
            item = json.loads(line)
            metadata = item["metadata"]
            problem_id = str(metadata["problem_id"])
            cases.append(
                SourceCase(
                    case_id=f"ds1000_{problem_id}",
                    stratum=str(metadata["library"]).lower(),
                    metadata={
                        "problem_id": metadata["problem_id"],
                        "perturbation_origin_id": metadata["perturbation_origin_id"],
                        "perturbation_type": metadata["perturbation_type"],
                    },
                )
            )
    return cases


def _read_json_lines(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_bfcl_catalog(source_dir: str | Path) -> list[SourceCase]:
    data_dir = Path(source_dir) / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    files = {
        "single_turn": (
            "BFCL_v4_simple_python.json",
            "BFCL_v4_parallel.json",
            "BFCL_v4_multiple.json",
        ),
        "multi_turn": ("BFCL_v4_multi_turn_base.json",),
    }
    cases: list[SourceCase] = []
    for stratum, names in files.items():
        for name in names:
            category = name.removeprefix("BFCL_v4_").removesuffix(".json")
            for item in _read_json_lines(data_dir / name):
                cases.append(
                    SourceCase(
                        case_id=item["id"],
                        stratum=stratum,
                        metadata={"category": category},
                    )
                )
    return cases


def load_tau_catalog(source_dir: str | Path) -> list[SourceCase]:
    domains_dir = Path(source_dir) / "data" / "tau2" / "domains"
    cases: list[SourceCase] = []
    for domain in ("airline", "retail", "telecom"):
        items = json.loads((domains_dir / domain / "tasks.json").read_text())
        for item in items:
            cases.append(
                SourceCase(
                    case_id=f"{domain}:{item['id']}",
                    stratum=domain,
                    metadata={"domain": domain, "source_task_id": item["id"]},
                )
            )
    return cases
