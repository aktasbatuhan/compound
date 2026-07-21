from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compound.contracts import Partition


@dataclass(frozen=True, slots=True)
class SourceCase:
    case_id: str
    stratum: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _rank(seed: int, benchmark: str, case_id: str) -> str:
    value = f"{seed}:{benchmark}:{case_id}".encode()
    return hashlib.sha256(value).hexdigest()


def stratified_sample(
    cases: Iterable[SourceCase], *, benchmark: str, seed: int, count: int
) -> list[SourceCase]:
    """Select a stable, balanced sample without depending on source ordering."""
    groups: dict[str, list[SourceCase]] = defaultdict(list)
    for case in cases:
        groups[case.stratum].append(case)
    if not groups:
        raise ValueError("cannot sample an empty case collection")
    if count > sum(len(group) for group in groups.values()):
        raise ValueError("sample count exceeds available cases")

    for group in groups.values():
        group.sort(key=lambda case: _rank(seed, benchmark, case.case_id))

    selected: list[SourceCase] = []
    strata = sorted(groups)
    cursor = 0
    while len(selected) < count:
        stratum = strata[cursor % len(strata)]
        if groups[stratum]:
            selected.append(groups[stratum].pop(0))
        elif not any(groups.values()):
            break
        cursor += 1
    return selected


def assign_partitions(
    cases: Iterable[SourceCase],
    *,
    train: int,
    validation: int,
    decision: int,
) -> dict[str, Partition]:
    ordered = list(cases)
    if len(ordered) != train + validation + decision:
        raise ValueError("partition sizes must exactly match the sample size")
    mapping: dict[str, Partition] = {}
    boundaries = (train, train + validation)
    for index, case in enumerate(ordered):
        if index < boundaries[0]:
            partition = Partition.OPTIMIZER_TRAIN
        elif index < boundaries[1]:
            partition = Partition.OPTIMIZER_VALIDATION
        else:
            partition = Partition.DECISION_TEST
        mapping[case.case_id] = partition
    return mapping


def write_manifest(
    path: str | Path,
    *,
    benchmark: str,
    revision: str,
    seed: int,
    cases: Iterable[SourceCase],
    partitions: dict[str, Partition],
) -> None:
    """Write a deterministic manifest without embedding benchmark answers."""
    records = []
    for case in cases:
        records.append(
            {
                "case_id": case.case_id,
                "stratum": case.stratum,
                "partition": partitions[case.case_id].value,
                "metadata": case.metadata,
            }
        )
    payload = {
        "benchmark": benchmark,
        "revision": revision,
        "seed": seed,
        "cases": records,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
