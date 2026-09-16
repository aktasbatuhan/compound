"""Offline tests for the private checkpoint and exact cloud cleanup evidence."""

from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import sqlite3
import stat
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/migration_cloud.py"
SPEC = importlib.util.spec_from_file_location("migration_cloud", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)


@pytest.mark.parametrize("reason", ["accepted_proposal", "heldout_started"])
def test_reflection_amendment_refuses_past_development_boundary(tmp_path, monkeypatch, reason):
    run = tmp_path / "study"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({"version": 10}))
    (run / "amendment-v10.json").write_text(json.dumps({"kind": "pretest_reflection_conformance"}))
    (run / "baseline.json").write_text(json.dumps({"status": "completed"}))
    frozen = {"methodology": "", "validation_result": {
        "valid_proposals": 1 if reason == "accepted_proposal" else 0, "reflection_calls": 4}}
    (run / "frozen-selection.json").write_text(json.dumps(frozen))
    with sqlite3.connect(run / "ledger.sqlite") as db:
        db.execute("create table calls(stage text)")
        if reason == "heldout_started":
            db.execute("insert into calls values ('final')")

    class EmptyArchive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def add(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(cloud, "RUN", run)
    monkeypatch.setattr(cloud, "verify", lambda: _instance())
    monkeypatch.setattr(cloud.tarfile, "open", lambda *args, **kwargs: EmptyArchive())
    monkeypatch.setattr(cloud, "command", lambda *args, **kwargs: "")

    def fake_ssh(command):
        if command.startswith("sudo python3 -c "):
            code = shlex.split(command)[-1].replace("/opt/compound/results", str(run))
            exec(compile(code, "remote-amendment", "exec"), {})
        return ""

    monkeypatch.setattr(cloud, "ssh", fake_ssh)
    with pytest.raises(AssertionError):
        cloud.upload(False, amendment=True)
    assert json.loads((run / "frozen-selection.json").read_text()) == frozen
    assert not (run / "amendments").exists()


def _results_archive(path: Path) -> None:
    source = path.parent / "archive-source"
    source.mkdir()
    (source / "private.json").write_text('{"private": true}\n')
    with tarfile.open(path, "w:gz") as archive:
        archive.add(source, arcname="results")


def _instance() -> dict:
    return {
        "name": cloud.VM,
        "id": "instance-123",
        "selfLink": f"https://compute.googleapis.com/compute/v1/projects/{cloud.PROJECT}/zones/{cloud.ZONE}/instances/{cloud.VM}",
        "zone": f"https://compute.googleapis.com/compute/v1/projects/{cloud.PROJECT}/zones/{cloud.ZONE}",
        "creationTimestamp": "2026-09-13T16:51:54.558-07:00",
        "scheduling": {"instanceTerminationAction": "DELETE"},
        "disks": [
            {
                "source": f"https://compute.googleapis.com/compute/v1/projects/{cloud.PROJECT}/zones/{cloud.ZONE}/disks/private-disk-42",
                "autoDelete": True,
            }
        ],
    }


def test_checkpoint_precreates_private_archive_and_removes_transients(tmp_path, monkeypatch):
    run = tmp_path / "study"
    remote = tmp_path / "remote-results.tgz"
    _results_archive(remote)
    remote_path = "/tmp/migration-results-safe_123.tgz"
    ssh_calls: list[str] = []

    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(cloud, "RUN", run)
    monkeypatch.setattr(cloud, "verify", lambda: _instance())

    def fake_ssh(code):
        ssh_calls.append(code)
        if code.startswith("sudo python3 -c "):
            assert "SUDO_UID" in code
            assert "os.chmod(archive,0o600)" in code
            return remote_path + "\n"
        assert code == "rm -f -- " + remote_path
        return ""

    def fake_command(args, **_kwargs):
        assert args[:3] == ["gcloud", "compute", "scp"]
        destination = Path(args[4])
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
        assert not destination.is_symlink()
        shutil.copyfile(remote, destination)
        return ""

    monkeypatch.setattr(cloud, "ssh", fake_ssh)
    monkeypatch.setattr(cloud, "command", fake_command)

    cloud.checkpoint()

    assert json.loads((run / "private.json").read_text()) == {"private": True}
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
    assert stat.S_IMODE((run / "private.json").stat().st_mode) == 0o600
    assert not (tmp_path / ".compound/migration-results.tgz").exists()
    assert ssh_calls[-1] == "rm -f -- " + remote_path


def test_private_archive_preparation_refuses_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("do not truncate")
    link = tmp_path / "archive.tgz"
    link.symlink_to(target)

    with pytest.raises(OSError):
        cloud._prepare_private_file(link)

    assert target.read_text() == "do not truncate"


def test_delete_records_exact_instance_and_attached_disk_identity(tmp_path, monkeypatch):
    run = tmp_path / "study"
    run.mkdir()
    instance = _instance()
    commands: list[list[str]] = []

    monkeypatch.setattr(cloud, "RUN", run)
    monkeypatch.setattr(cloud, "checkpoint", lambda: None)
    monkeypatch.setattr(cloud, "verify", lambda: instance)

    def fake_command(args, **_kwargs):
        commands.append(args)
        if "describe" in args and "disks" in args:
            return json.dumps(
                {
                    "name": "private-disk-42",
                    "id": "disk-987",
                    "selfLink": instance["disks"][0]["source"],
                    "zone": instance["zone"],
                    "creationTimestamp": "2026-09-13T16:51:55.000-07:00",
                }
            )
        if "instances" in args and "delete" in args:
            return ""
        if "list" in args:
            return "[]"
        raise AssertionError(args)

    monkeypatch.setattr(cloud, "command", fake_command)

    cloud.delete()

    cleanup = json.loads((run / "cleanup.json").read_text())
    assert cleanup["vm_deleted"] is True
    assert cleanup["disk_deleted"] is True
    assert cleanup["instance"]["id"] == "instance-123"
    assert cleanup["instance"]["creationTimestamp"] == instance["creationTimestamp"]
    assert cleanup["instance"]["confirmed_absent_at"] == cleanup["at"]
    assert cleanup["disks"][0]["name"] == "private-disk-42"
    assert cleanup["disks"][0]["id"] == "disk-987"
    assert cleanup["disks"][0]["confirmed_absent_at"] == cleanup["at"]
    assert stat.S_IMODE((run / "cleanup.json").stat().st_mode) == 0o600
    delete = next(args for args in commands if "instances" in args and "delete" in args)
    assert "--delete-disks=all" in delete


def test_delete_rejects_surviving_attached_disk_with_non_vm_name(tmp_path, monkeypatch):
    run = tmp_path / "study"
    run.mkdir()
    instance = _instance()

    monkeypatch.setattr(cloud, "RUN", run)
    monkeypatch.setattr(cloud, "checkpoint", lambda: None)
    monkeypatch.setattr(cloud, "verify", lambda: instance)

    def fake_command(args, **_kwargs):
        if "describe" in args and "disks" in args:
            return json.dumps(
                {
                    "name": "private-disk-42",
                    "id": "disk-987",
                    "selfLink": instance["disks"][0]["source"],
                    "zone": instance["zone"],
                }
            )
        if "instances" in args and "delete" in args:
            return ""
        if "instances" in args and "list" in args:
            return "[]"
        if "disks" in args and "list" in args:
            return json.dumps(
                [
                    {
                        "name": "private-disk-42",
                        "id": "disk-987",
                        "selfLink": instance["disks"][0]["source"],
                        "zone": instance["zone"],
                    }
                ]
            )
        raise AssertionError(args)

    monkeypatch.setattr(cloud, "command", fake_command)

    with pytest.raises(RuntimeError, match="cleanup not verified"):
        cloud.delete()

    assert not (run / "cleanup.json").exists()
