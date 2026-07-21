import json

from compound.adapters.bfcl import write_bfcl_run_ids
from compound.contracts import Partition


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

