"""Checkpoint and delete the fixed, authorized $10 experiment VM."""

# ruff: noqa: E501 -- Fixed remote commands and the embedded state query.

import fcntl
import json
import shlex
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import supervise_flex_gcp as cloud

VM = "compound-flex-budget-20260910"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/flex-budget-20260910"
STATE = """
import json
from pathlib import Path
p=Path('/opt/compound/artifacts/flex-budget-20260910')
rows=p/'token-informed/outcomes.jsonl'
if not (p/'token-informed/study-spec.json').exists(): rows=p/'main/outcomes.jsonl'
spend={}
for stage in ('smoke','main','token-informed'):
 f=p/stage/'spend.json'
 spend[stage]=sum(x['charged_or_reserved_usd'] for x in json.loads(f.read_text())) if f.exists() else 0
end=Path('/opt/final.exit')
print(json.dumps({'recorded':len(rows.read_text().splitlines()) if rows.exists() else 0,'spend_guard_usd':spend,'exit_code':int(end.read_text()) if end.exists() else None}))
"""


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
            cloud.ssh(
                "sudo tar -czf /home/batuhanaktas/flex-budget-results.tgz -C /opt/compound artifacts/flex-budget-20260910; archive_status=$?; if [ \"$archive_status\" -gt 1 ]; then exit \"$archive_status\"; fi; sudo chmod 644 /home/batuhanaktas/flex-budget-results.tgz"
            )
            archive = ROOT / ".compound/flex-budget-results.tgz"
            cloud.run(
                cloud.GCLOUD
                + ["scp", VM + ":flex-budget-results.tgz", str(archive), "--zone=" + cloud.ZONE]
            )
            with tarfile.open(archive) as tar:
                tar.extractall(
                    ROOT,
                    members=[
                        m for m in tar if m.name.startswith("artifacts/flex-budget-20260910/")
                    ],
                    filter="data",
                )
            last = time.monotonic()
        state["checked_at"] = datetime.now(UTC).isoformat()
        (OUT / "supervisor-status.json").write_text(json.dumps(state, indent=2) + "\n")
        if state["exit_code"] is not None:
            cloud.run(
                cloud.GCLOUD + ["instances", "delete", VM, "--zone=" + cloud.ZONE], timeout=240
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
