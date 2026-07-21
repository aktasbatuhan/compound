from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from compound.contracts import Partition


def write_bfcl_run_ids(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    partitions: set[Partition],
) -> dict[str, list[str]]:
    """Translate Compound partitions into BFCL's native selective-run file."""
    manifest = json.loads(Path(manifest_path).read_text())
    allowed = {partition.value for partition in partitions}
    categories: dict[str, list[str]] = defaultdict(list)
    for case in manifest["cases"]:
        if case["partition"] not in allowed:
            continue
        category = case["metadata"]["category"]
        categories[category].append(case["case_id"])
    payload = {category: ids for category, ids in sorted(categories.items())}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload

