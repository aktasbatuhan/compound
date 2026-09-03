#!/usr/bin/env bash
# Does a FrontierSWE v2 task actually grade? Run its own reference solution
# through the verifier on one throwaway VM and read the reward back.
#
# This is the check to run BEFORE a provider grid, not after. The first grid
# produced 2,163 traced calls and zero quality signal, because every task failed
# at scoring with "Task environment directory <task>/tests has no environment
# definition" while the agent phase looked perfectly healthy. An oracle pass
# costs one VM and no model tokens, and it fails loudly for exactly that reason.
#
# Each task ships solution/solve.sh and declares
# [metadata] oracle_reward_threshold. Harbor's built-in `oracle` agent runs that
# script instead of a model, so a correct pipeline scores at or above the
# threshold. A task that scores 0 with valid=1 is a broken solution; valid=0 is
# an infrastructure error; no reward file at all means grading never ran.
#
#   GCP_PROJECT=... bash scripts/cloud/gcp-fswe-oracle.sh
#   GCP_PROJECT=... FSWE_TASKS=verilog-simulator-in-swift bash scripts/cloud/gcp-fswe-oracle.sh
#
# Verify nothing is left afterwards:  gcloud compute instances list
set -uo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
ZONE="${GCP_ZONE:-us-central1-a}"
# 8 vCPU / 32GB covers every CPU-only task's declared verifier ceiling; the
# cheap tier asks for at most 4 cpu / 8GB, and the verifier gets its own
# container alongside the agent's.
MACHINE="${GCP_MACHINE:-e2-standard-8}"
DISK="${GCP_DISK:-200GB}"
DISK_TYPE="${GCP_DISK_TYPE:-pd-standard}"
MAX_DURATION="${GCP_MAX_DURATION:-21600s}"

REPO_URL="${FSWE_REPO:-https://github.com/Proximal-Labs/frontier-swe-v2.git}"
REPO_COMMIT="${FSWE_COMMIT:-}"
TASKS="${FSWE_TASKS:-verilog-simulator-in-swift,crash-proof-flash-filesystem,libexpat-optimization,spice-circuit-simulator-in-rust}"
STAMP="$(date +%s)"
OUT_ROOT="${FSWE_OUT:-artifacts/fswe-oracle-$STAMP}"
VM="fswe-oracle-$STAMP"

mkdir -p "$OUT_ROOT"
echo "== oracle check on $(echo "$TASKS" | tr ',' ' ' | wc -w | tr -d ' ') tasks"
echo "== vm:   $VM ($MACHINE, self-deletes after $MAX_DURATION)"
echo "== out:  $OUT_ROOT"

cleanup() {
  echo "== deleting $VM =="
  gcloud compute instances delete "$VM" --zone="$ZONE" --project="$PROJECT" --quiet >/dev/null 2>&1
}
trap cleanup EXIT

gssh() {
  gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --command="$1" \
    -- -n -o StrictHostKeyChecking=no -o ConnectTimeout=15
}

STARTUP='#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
mkdir -p /opt/compound && chmod 777 /opt/compound
apt-get update -y
apt-get install -y git curl ca-certificates tar rsync python3
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sh /tmp/get-docker.sh
systemctl enable --now docker
for i in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 3; done
chmod 666 /var/run/docker.sock || true
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh || true
docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && touch /tmp/setup-done'

STARTUP_FILE="$(mktemp)"; printf '%s\n' "$STARTUP" > "$STARTUP_FILE"

echo "== creating $VM =="
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size="$DISK" --boot-disk-type="$DISK_TYPE" \
  --provisioning-model=STANDARD \
  --max-run-duration="$MAX_DURATION" --instance-termination-action=DELETE \
  --metadata-from-file=startup-script="$STARTUP_FILE" >/dev/null 2>&1
rm -f "$STARTUP_FILE"

echo "== waiting for docker =="
for i in $(seq 1 60); do
  gssh "test -f /tmp/setup-done" >/dev/null 2>&1 && break
  sleep 15
done
gssh "test -f /tmp/setup-done" >/dev/null 2>&1 && echo "  ready" || echo "  NOT READY (trying anyway)"

SRC_TGZ="/tmp/compound-oracle-$STAMP.tgz"
tar czf "$SRC_TGZ" scripts 2>/dev/null
gcloud compute scp "$SRC_TGZ" "$VM":/opt/compound/src.tgz --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
rm -f "$SRC_TGZ"

RUNNER="$(mktemp)"
cat > "$RUNNER" <<EOF
#!/bin/bash
cd /opt/compound
rm -f RUN_DONE
{
  set -x
  tar xzf src.tgz
  [ -d repo ] || git clone --depth 1 "$REPO_URL" repo
  if [ -n "$REPO_COMMIT" ]; then (cd repo && git fetch --depth 1 origin "$REPO_COMMIT" && git checkout "$REPO_COMMIT"); fi
  (cd repo && git rev-parse HEAD > /opt/compound/REPO_COMMIT.txt)

  # The fix under test: give Harbor the separate-verifier build context that
  # FrontierSWE ships no equivalent of. Without this every task fails at scoring.
  for TASK in \$(echo "$TASKS" | tr ',' ' '); do
    python3 scripts/fswe_prepare.py repo --task "\$TASK"
  done

  for TASK in \$(echo "$TASKS" | tr ',' ' '); do
    echo "===== ORACLE \$TASK \$(date -u +%H:%M:%S) ====="
    timeout 14400 uvx --from harbor harbor run \
      --path "repo/tasks/\$TASK" \
      --agent oracle \
      --n-attempts 1 --n-concurrent 1 \
      --jobs-dir "out/\$TASK"
    echo "===== exit \$? \$(date -u +%H:%M:%S) ====="
  done
} > /opt/compound/oracle.log 2>&1
# reward.json is bind-mounted back from the verifier container into the trial dir.
find out -name 'reward.json' -o -name 'reward.txt' -o -name 'results.json' \
  | tar czf /opt/compound/rewards.tgz -T - 2>/dev/null
tar czf /opt/compound/results.tgz out REPO_COMMIT.txt oracle.log 2>/dev/null
touch /opt/compound/RUN_DONE
EOF
gcloud compute scp "$RUNNER" "$VM":/opt/compound/oracle.sh --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
rm -f "$RUNNER"

echo "== launching oracle run =="
gssh "chmod +x /opt/compound/oracle.sh && sudo systemd-run --unit=fswe-oracle --collect /opt/compound/oracle.sh" >/dev/null 2>&1

echo "== waiting (poll every 60s) =="
for i in $(seq 1 240); do
  if gssh "test -f /opt/compound/RUN_DONE" >/dev/null 2>&1; then
    echo "  done after ~$((i)) minutes"
    break
  fi
  sleep 60
done

echo "== collecting =="
gcloud compute scp "$VM":/opt/compound/results.tgz "$OUT_ROOT/results.tgz" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
gcloud compute scp "$VM":/opt/compound/oracle.log "$OUT_ROOT/oracle.log" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
tar xzf "$OUT_ROOT/results.tgz" -C "$OUT_ROOT" 2>/dev/null

echo
echo "== rewards =="
find "$OUT_ROOT" -name 'reward.json' -print -exec cat {} \; 2>/dev/null
echo
echo "== full log: $OUT_ROOT/oracle.log"
