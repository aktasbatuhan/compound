"""Supervise this authorized $15 GCP pilot; never create or target another VM."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tarfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "core-planet-439017-m2"
ZONE = "us-central1-a"
VM = "compound-flex-20260909"
RESULTS = ROOT / "artifacts/flex-agentic-gcp"
GCLOUD = ["gcloud", "--quiet", "--project=" + PROJECT, "compute"]
EXPECTED_CREATION = datetime.fromisoformat("2026-09-09T03:03:48.951-07:00")
STATE_SCRIPT = """
import json, os
from pathlib import Path
p=Path('/opt/compound/artifacts/flex-agentic-gcp')
rows=[json.loads(x) for x in (p/'outcomes.jsonl').read_text().splitlines()]
spend=json.loads((p/'spend.json').read_text())
plan=json.loads((p/'plan.json').read_text())
episodes={e['episode_id']:e for e in plan['episodes']}
infra=[]
for row in rows:
 if row['status']!='infrastructure_error': continue
 proof=p/row['episode_id']/'official.json'
 raw=json.loads(proof.read_text()) if proof.exists() else {}
 if episodes[row['episode_id']]['task_id'] not in raw.get('empty_patch_ids',[]):
  infra.append(row['episode_id'])
active=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
 try: args=f.read_bytes().decode(errors='replace').split('\\0')
 except OSError: continue
 if 'compound.agentic_run' in args: active.append(int(f.parent.name))
print(json.dumps({'active':active,'recorded':len(rows),'planned':plan['episode_count'],
 'guard_usd':sum(x['charged_or_reserved_usd'] for x in spend),'infra':infra}))
"""
SNAPSHOT_SCRIPT = """
import json, shutil, tarfile, tempfile, os
from pathlib import Path
root=Path('/opt/compound')
relative=Path('artifacts/flex-agentic-gcp')
source=root/relative
rows=[json.loads(x) for x in (source/'outcomes.jsonl').read_text().splitlines()]
ids=[r['episode_id'] for r in rows]
ids += [p.name for p in source.glob('oracle-*') if (p/'outcome.json').exists()]
with tempfile.TemporaryDirectory(prefix='compound-snapshot-') as tmp:
 staging=Path(tmp); target=staging/relative; target.mkdir(parents=True)
 for p in source.iterdir():
  if p.is_file() and p.suffix in ('.json','.jsonl'):
   text=p.read_text()
   if p.suffix=='.json': json.loads(text)
   else: [json.loads(line) for line in text.splitlines()]
   (target/p.name).write_text(text)
 # Use exactly the outcome snapshot whose completed directories we copy.
 (target/'outcomes.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
 for eid in ids:
  shutil.copytree(source/eid,target/eid)
  log=root/'logs/evaluation'/eid
  if log.exists(): shutil.copytree(log,staging/'logs/evaluation'/eid)
 data=Path('.compound/sources/swe-verified-grader.json')
 (staging/data).parent.mkdir(parents=True)
 shutil.copy2(root/data,staging/data)
 with tarfile.open('/home/batuhanaktas/flex-results.tgz','w:gz') as tar:
  for name in ('artifacts','.compound','logs'):
   if (staging/name).exists(): tar.add(staging/name,arcname=name)
os.chmod('/home/batuhanaktas/flex-results.tgz',0o644)
"""
RECONCILE_SCRIPT = """
import fcntl, json
from pathlib import Path
from compound.agentic_gateway import is_context_rejection, VALIDATION_WAIVER_SOURCE
p=Path('/opt/compound/artifacts/flex-agentic-gcp')
with (p/'runner.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 spend=json.loads((p/'spend.json').read_text())
 calls=[json.loads(x) for x in (p/'calls.jsonl').read_text().splitlines()]
 models={m['id']:m['provider'] for m in json.loads((p/'study-spec.json').read_text())['models']}
 events=[]
 for call in calls:
  if models.get(call['route'])!='openrouter': continue
  if not is_context_rejection(call['status'],call.get('error_body','')): continue
  matches=[x for x in spend if not x['settled'] and x['episode_id']==call['episode_id']
   and x['role']==call['role'] and x['reservation_usd']==call['reservation_usd']]
  if len(matches)!=1: continue
  entry=matches[0]
  events.append({'episode_id':call['episode_id'],'request_sha256':call['request_sha256'],
   'released_usd':entry['charged_or_reserved_usd'],'source':VALIDATION_WAIVER_SOURCE})
  entry.update(settled=True,charged_or_reserved_usd=0,settlement_source=VALIDATION_WAIVER_SOURCE)
  call.update(cost_usd=0,cost_kind='documented_validation_waiver',
   cost_source=VALIDATION_WAIVER_SOURCE,failure_reason='context_limit')
 if events:
  with (p/'reconciliations.jsonl').open('a') as f:
   for event in events: f.write(json.dumps(event)+'\\n')
  for name,body in [('spend.json',json.dumps(spend,indent=2)),
                    ('calls.jsonl',''.join(json.dumps(c)+'\\n' for c in calls))]:
   temp=p/(name+'.reconciled.tmp'); temp.write_text(body); temp.replace(p/name)
 print(json.dumps({'released_usd':sum(e['released_usd'] for e in events)}))
"""


def run(args, *, timeout=180):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args[:3]}: {result.stderr[-1500:]}")
    return result.stdout.strip()


def ssh(command):
    return run(GCLOUD + ["ssh", VM, "--zone=" + ZONE, "--command=" + command])


def state():
    return json.loads(ssh("sudo python3 -c " + shlex.quote(STATE_SCRIPT)).splitlines()[-1])


def instance():
    return json.loads(
        run(
            GCLOUD
            + [
                "instances",
                "describe",
                VM,
                "--zone=" + ZONE,
                "--format=json(creationTimestamp,lastStartTimestamp,scheduling)",
            ]
        )
    )


def checkpoint():
    ssh("sudo python3 -c " + shlex.quote(SNAPSHOT_SCRIPT))
    archive = ROOT / ".compound/flex-results.tgz"
    run(GCLOUD + ["scp", VM + ":flex-results.tgz", str(archive), "--zone=" + ZONE])
    with tarfile.open(archive) as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.name.startswith("artifacts/flex-agentic-gcp/")
            or m.name == ".compound/sources/swe-verified-grader.json"
        ]
        tar.extractall(ROOT, members=members, filter="data")
    run([str(ROOT / ".venv/bin/python"), "scripts/summarize_flex_pilot.py"])


def choose_action(current, *, checkpoint_due, boots, released_usd=0):
    if current["active"]:
        return "wait"
    if current["recorded"] == current["planned"]:
        return "complete" if not current["infra"] else "infrastructure_stop"
    if current["guard_usd"] >= 11.5:
        return "budget_stop"
    if current["infra"]:
        return "infrastructure_stop"
    if checkpoint_due:
        return "restart" if boots < 2 else "runtime_stop"
    if released_usd > 0:
        return "continue"
    return "unexpected_stop"


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args()
    if not args.go:
        print(
            "Dry run. --go enables checkpointing, one resume, and deletion of the fixed pilot VM."
        )
        return
    lock = (ROOT / ".compound/flex-supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = RESULTS / "supervisor-status.json"
    saved = json.loads(status_path.read_text()) if status_path.exists() else {}
    if saved.get("vm_deleted"):
        print("This pilot VM was already deleted.")
        return
    started = instance()
    if datetime.fromisoformat(started["creationTimestamp"]) != EXPECTED_CREATION:
        raise RuntimeError("The VM is not the originally authorized instance")
    if int(started["scheduling"]["maxRunDuration"]["seconds"]) > 14400:
        raise RuntimeError("Unexpected VM runtime allowance")
    first_start = datetime.fromisoformat(saved.get("first_started_at", "2026-09-09T10:03:56.965Z"))
    boots, last_checkpoint = saved.get("boots", 1), 0.0
    stop_at = datetime.fromisoformat(saved.get("stop_at", "2026-09-09T13:20:00Z"))
    history = saved.get("history", [])
    actual_start = datetime.fromisoformat(started["lastStartTimestamp"])
    if actual_start > first_start + timedelta(minutes=1) and boots < 2:
        boots = 2
        stop_at = min(
            actual_start + timedelta(hours=3, minutes=15),
            first_start + timedelta(hours=7, minutes=15),
        )
    consecutive_errors = 0
    while True:
        try:
            current = state()
            released_usd = 0
            if not current["active"]:
                reconciliation = ssh(
                    "sudo /opt/compound/.venv/bin/python -c " + shlex.quote(RECONCILE_SCRIPT)
                )
                released_usd = json.loads(reconciliation.splitlines()[-1])["released_usd"]
                current = state()
            action = choose_action(
                current,
                checkpoint_due=datetime.now(UTC) >= stop_at,
                boots=boots,
                released_usd=released_usd,
            )
            if datetime.now(UTC) >= first_start + timedelta(hours=8):
                action = "runtime_stop"
            report = {
                **current,
                "action": action,
                "boots": boots,
                "updated_at": datetime.now(UTC).isoformat(),
                "history": history,
                "stop_at": stop_at.isoformat(),
                "first_started_at": first_start.isoformat(),
            }
            status_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            if action != "wait" or time.monotonic() - last_checkpoint > 600:
                checkpoint()
                last_checkpoint = time.monotonic()
            consecutive_errors = 0
            if action == "wait":
                time.sleep(45)
                continue
            if action == "restart":
                # Two runs of at most four hours: compute + disk/network remain
                # inside the separate $3 infrastructure reserve for this VM.
                run(GCLOUD + ["instances", "stop", VM, "--zone=" + ZONE], timeout=240)
                run(GCLOUD + ["instances", "start", VM, "--zone=" + ZONE], timeout=240)
                boots += 1
                new_start = datetime.fromisoformat(instance()["lastStartTimestamp"])
                stop_at = min(
                    new_start + timedelta(hours=3, minutes=15),
                    first_start + timedelta(hours=7, minutes=15),
                )
                report.update(boots=boots, stop_at=stop_at.isoformat())
                status_path.write_text(json.dumps(report, indent=2) + "\n")
                for _ in range(12):
                    try:
                        ssh("sudo systemctl is-active docker")
                        break
                    except Exception:
                        time.sleep(10)
                else:
                    raise RuntimeError("VM did not become ready after restart")
            if action in {"restart", "continue"}:
                command = (
                    "cd /opt/compound && nohup .venv/bin/python -m compound.agentic_run "
                    "run --count 150 --parallel-routes --stop-at "
                    + shlex.quote(stop_at.isoformat())
                    + " --go >/opt/full-resumed.log 2>&1 </dev/null &"
                )
                ssh("sudo bash -c " + shlex.quote(command))
                history.append(
                    {
                        "action": action,
                        "started_at": datetime.now(UTC).isoformat(),
                        "stop_at": stop_at.isoformat(),
                    }
                )
                time.sleep(10)
                continue
            # Preserve artifacts before removing the dedicated VM and its boot disk.
            run(GCLOUD + ["instances", "delete", VM, "--zone=" + ZONE], timeout=240)
            remaining = json.loads(
                run(GCLOUD + ["instances", "list", "--filter=name=" + VM, "--format=json(name)"])
            )
            disks = json.loads(
                run(GCLOUD + ["disks", "list", "--filter=name=" + VM, "--format=json(name)"])
            )
            if remaining or disks:
                raise RuntimeError("VM or boot disk remains after deletion")
            report.update(
                vm_deleted=True,
                boot_disk_deleted=True,
                finished_at=datetime.now(UTC).isoformat(),
                first_started_at=first_start.isoformat(),
            )
            status_path.write_text(json.dumps(report, indent=2) + "\n")
            (ROOT / ".compound/flex-gcp-run.json").write_text(json.dumps(report, indent=2) + "\n")
            print("FINAL", json.dumps(report), flush=True)
            return
        except Exception as exc:
            consecutive_errors += 1
            print("SUPERVISOR_ERROR", str(exc), flush=True)
            if consecutive_errors >= 5:
                # No new paid work is launched on supervisor errors. The existing
                # GCP deletion deadline remains the last-resort compute stop.
                status_path.write_text(
                    json.dumps(
                        {
                            "action": "supervisor_error",
                            "error": str(exc),
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    + "\n"
                )
                raise
            time.sleep(20)


if __name__ == "__main__":
    main()
