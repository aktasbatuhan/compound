"""Checkpoint and clean up the explicitly authorized continuation VM."""

import argparse
import fcntl
import json
import shlex
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import supervise_flex_gcp as pilot

VM = "compound-flex-finish-20260909"
CREATED = datetime.fromisoformat("2026-09-09T07:53:30.462-07:00")
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/flex-agentic-finish"


def checkpoint():
    script = pilot.SNAPSHOT_SCRIPT.replace("flex-agentic-gcp", "flex-agentic-finish")
    # Baseline outcomes are carried forward, but their files remain in the original checkpoint.
    script = script.replace(
        "for eid in ids:\n", "for eid in ids:\n  if not (source/eid).exists(): continue\n"
    )
    pilot.ssh("sudo python3 -c " + shlex.quote(script))
    archive = ROOT / ".compound/flex-finish-results.tgz"
    pilot.run(
        pilot.GCLOUD + ["scp", VM + ":flex-results.tgz", str(archive), "--zone=" + pilot.ZONE]
    )
    with tarfile.open(archive) as tar:
        members = [
            m for m in tar.getmembers() if m.name.startswith("artifacts/flex-agentic-finish/")
        ]
        tar.extractall(ROOT, members=members, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args()
    if not args.go:
        print("Dry run; --go checkpoints and deletes the fixed continuation VM after execution.")
        return
    lock = (ROOT / ".compound/flex-finish-supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    DIRECTORY.mkdir(exist_ok=True)
    status = DIRECTORY / "supervisor-status.json"
    if status.exists() and json.loads(status.read_text()).get("vm_deleted"):
        return
    pilot.VM = VM
    actual = pilot.instance()
    if datetime.fromisoformat(actual["creationTimestamp"]) != CREATED:
        raise RuntimeError("Unexpected VM identity")
    pilot.STATE_SCRIPT = pilot.STATE_SCRIPT.replace("flex-agentic-gcp", "flex-agentic-finish")
    pilot.STATE_SCRIPT = pilot.STATE_SCRIPT.replace(
        " raw=json.loads(proof.read_text()) if proof.exists() else {}",
        " if not proof.exists(): proof=Path('/opt/compound/artifacts/flex-agentic-gcp')"
        "/row['episode_id']/'official.json'\n"
        " raw=json.loads(proof.read_text()) if proof.exists() else {}",
    )
    last_checkpoint = 0
    while True:
        try:
            current = pilot.state()
            report = {**current, "updated_at": datetime.now(UTC).isoformat(), "vm": VM}
            status.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            if not current["active"] or time.monotonic() - last_checkpoint > 300:
                checkpoint()
                last_checkpoint = time.monotonic()
            if not current["active"]:
                pilot.run(
                    pilot.GCLOUD + ["instances", "delete", VM, "--zone=" + pilot.ZONE], timeout=240
                )
                for resource in ("instances", "disks"):
                    remains = json.loads(
                        pilot.run(
                            pilot.GCLOUD
                            + [resource, "list", "--filter=name=" + VM, "--format=json(name)"]
                        )
                    )
                    if remains:
                        raise RuntimeError(resource + " remains after deletion")
                report.update(
                    vm_deleted=True,
                    boot_disk_deleted=True,
                    finished_at=datetime.now(UTC).isoformat(),
                )
                status.write_text(json.dumps(report, indent=2) + "\n")
                print("FINAL", json.dumps(report), flush=True)
                return
        except Exception as exc:
            print("SUPERVISOR_ERROR", str(exc), flush=True)
        time.sleep(45)


if __name__ == "__main__":
    main()
