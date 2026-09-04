import json

import pytest
import yaml

from compound.adapters.terminal_bench import (
    TIMEOUT_BASE_FILE,
    apply_timeout_mult,
    build_manifest,
    timeout_mult_from_env,
    write_run_metadata,
)


def _write_task(root, name, **fields):
    task_dir = root / name
    task_dir.mkdir()
    body = {"instruction": "do a thing", **fields}
    (task_dir / "task.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in body.items())
    )


def _limit(root, name):
    return yaml.safe_load((root / name / "task.yaml").read_text())["max_agent_timeout_sec"]


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


def test_apply_timeout_mult_scales_and_records_base(tmp_path) -> None:
    _write_task(tmp_path, "alpha", max_agent_timeout_sec=100)
    _write_task(tmp_path, "beta", max_agent_timeout_sec=60)
    _write_task(tmp_path, "no-limit")  # tasks without a limit are left untouched
    patched = apply_timeout_mult(tmp_path, 3)
    assert patched == 2
    assert _limit(tmp_path, "alpha") == 300
    assert _limit(tmp_path, "beta") == 180
    base = json.loads((tmp_path / TIMEOUT_BASE_FILE).read_text())
    assert base == {"alpha": 100, "beta": 60}


def test_apply_timeout_mult_is_idempotent_and_recomputes_from_base(tmp_path) -> None:
    _write_task(tmp_path, "alpha", max_agent_timeout_sec=100)
    apply_timeout_mult(tmp_path, 3)
    # Re-applying a different multiplier recomputes from the recorded base, so it
    # never compounds; 1.0 restores the shipped limit.
    apply_timeout_mult(tmp_path, 2)
    assert _limit(tmp_path, "alpha") == 200
    apply_timeout_mult(tmp_path, 1)
    assert _limit(tmp_path, "alpha") == 100


def test_timeout_mult_from_env(monkeypatch) -> None:
    monkeypatch.delenv("COMPOUND_TB_TIMEOUT_MULT", raising=False)
    assert timeout_mult_from_env() is None
    monkeypatch.setenv("COMPOUND_TB_TIMEOUT_MULT", "3")
    assert timeout_mult_from_env() == 3.0
    monkeypatch.setenv("COMPOUND_TB_TIMEOUT_MULT", "0")  # nonsense -> unset, never shrinks
    assert timeout_mult_from_env() is None
    monkeypatch.setenv("COMPOUND_TB_TIMEOUT_MULT", "nope")
    assert timeout_mult_from_env() is None


def test_write_run_metadata_labels_the_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COMPOUND_REASONING", "off")
    monkeypatch.setenv("COMPOUND_DW_CACHE", "1")
    path = write_run_metadata(tmp_path, model="m", agent="terminus", timeout_mult=3.0)
    meta = json.loads(path.read_text())
    assert meta["reasoning_mode"] == "off"
    assert meta["cache_optin"] is True
    assert meta["tb_timeout_mult"] == 3.0
    assert meta["extended_limits"] is True and meta["official_limits"] is False


def test_write_run_metadata_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("COMPOUND_REASONING", raising=False)
    monkeypatch.delenv("COMPOUND_DW_CACHE", raising=False)
    meta = json.loads(
        write_run_metadata(tmp_path, model="m", agent="terminus", timeout_mult=None).read_text()
    )
    assert meta["reasoning_mode"] == "default"
    # Cache markers default ON, so an unset env records True.
    assert meta["cache_optin"] is True
    assert meta["tb_timeout_mult"] == 1.0
    assert meta["extended_limits"] is False and meta["official_limits"] is True
