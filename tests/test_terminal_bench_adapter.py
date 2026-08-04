import json

import pytest

from compound.adapters.terminal_bench import build_manifest


def _write_task(root, name, **fields):
    task_dir = root / name
    task_dir.mkdir()
    body = {"instruction": "do a thing", **fields}
    (task_dir / "task.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in body.items())
    )


def test_build_manifest_reads_task_yaml(tmp_path) -> None:
    _write_task(tmp_path, "chess-best-move", difficulty="medium", category="games")
    _write_task(tmp_path, "build-tcc-qemu", difficulty="hard", category="build")
    out = build_manifest(tmp_path / "tb.json", dataset_dir=tmp_path)
    manifest = json.loads(out.read_text())
    assert manifest["benchmark"] == "terminal_bench"
    by_id = {c["case_id"]: c for c in manifest["cases"]}
    assert by_id["chess-best-move"]["metadata"]["difficulty"] == "medium"
    assert by_id["build-tcc-qemu"]["metadata"]["category"] == "build"
    assert all(c["partition"] for c in manifest["cases"])


def test_build_manifest_errors_on_empty_dataset(tmp_path) -> None:
    with pytest.raises(SystemExit):
        build_manifest(tmp_path / "tb.json", dataset_dir=tmp_path)
