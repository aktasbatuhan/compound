"""Checkpoint and delete the replication VM for the harder-task trial."""

# ruff: noqa: E501 -- Fixed remote commands and the embedded state query.

import fcntl
import json
import shlex
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import supervise_flex_gcp as cloud

VM = "compound-flex-hard-rep2"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/flex-hard-rep2"
STATE = """
import json
from pathlib import Path
p=Path('/opt/compound/artifacts/flex-hard-rep2')
rows=p/'outcomes.jsonl'
f=p/'spend.json'
spend=sum(x['charged_or_reserved_usd'] for x in json.loads(f.read_text())) if f.exists() else 0
end=Path('/opt/compound/hard2.final.exit')
controller=Path('/opt/compound/hard2.exit')
print(json.dumps({'recorded':len(rows.read_text().splitlines()) if rows.exists() else 0,'spend_guard_usd':{'run':spend},'exit_code':int(end.read_text()) if end.exists() else None,'controller_exit':int(controller.read_text()) if controller.exists() else None}))
"""


def checkpoint():
    cloud.ssh(
        'sudo tar -czf /home/batuhanaktas/flex-rep2-results.tgz -C /opt/compound artifacts/flex-hard-rep2; archive_status=$?; if [ "$archive_status" -gt 1 ]; then exit "$archive_status"; fi; sudo chmod 644 /home/batuhanaktas/flex-rep2-results.tgz'
    )
    archive = ROOT / ".compound/flex-rep2-results.tgz"
    cloud.run(
        cloud.GCLOUD + ["scp", VM + ":flex-rep2-results.tgz", str(archive), "--zone=" + cloud.ZONE],
        timeout=600,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(
            ROOT,
            members=[m for m in tar if m.name.startswith("artifacts/flex-hard-rep2/")],
            filter="data",
        )


def main():
    lock = (OUT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cloud.VM = VM
    expected = json.loads((OUT / "vm.json").read_text())
    assert cloud.instance()["creationTimestamp"] == expected["creationTimestamp"]
    last = 0
    while True:
        state = json.loads(cloud.ssh("sudo python3 -c " + shlex.quote(STATE)))
        print(json.dumps(state), flush=True)
        if time.monotonic() - last > 180 or state["exit_code"] is not None:
            checkpoint()
            last = time.monotonic()
        state["checked_at"] = datetime.now(UTC).isoformat()
        (OUT / "supervisor-status.json").write_text(json.dumps(state, indent=2) + "\n")
        if state["exit_code"] is not None:
            cloud.run(
                cloud.GCLOUD + ["instances", "delete", VM, "--zone=" + cloud.ZONE], timeout=600
            )
            for kind in ("instances", "disks"):
                assert not json.loads(
                    cloud.run(
                        cloud.GCLOUD + [kind, "list", "--filter=name=" + VM, "--format=json(name)"]
                    )
                )
            state.update(
                vm_deleted=True, boot_disk_deleted=True, finished_at=datetime.now(UTC).isoformat()
            )
            (OUT / "supervisor-status.json").write_text(json.dumps(state, indent=2) + "\n")
            print("FINAL", json.dumps(state), flush=True)
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
