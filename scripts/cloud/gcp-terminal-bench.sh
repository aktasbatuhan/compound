#!/usr/bin/env bash
# One-shot: provision a Docker VM on GCP, run the terminal-bench provider sweep
# there, copy results back, and delete the VM. Cost is bounded three ways:
#   - Standard (non-preemptible) e2-standard-4, ~$0.13/hr.
#   - GCP server-side --max-run-duration=4h --instance-termination-action=DELETE,
#     so the VM deletes itself even if this script dies.
#   - A cleanup trap that deletes the VM as soon as results land (usually <2h).
# Absolute ceiling ~$0.50; expected ~$0.25.
#
# Run it from the repo root with your keys in the environment (or in .env):
#   OPENROUTER_API_KEY=... DOUBLEWORD_API_KEY=... bash scripts/cloud/gcp-terminal-bench.sh
#
# Verify nothing is left afterwards:  gcloud compute instances list
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT to your GCP project id}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM="${GCP_VM:-tb-dsflash}"
MACHINE="${GCP_MACHINE:-e2-standard-4}"
# Server-side hard kill-switch; raise for many hosts/tasks (8 hosts x ~14 tasks
# needs > the 4h default). Still deletes itself even if this script dies.
MAX_DURATION="${GCP_MAX_DURATION:-14400s}"
MODEL="${TB_MODEL:-deepseek/deepseek-v4-flash-0731}"
DW_MODEL="${TB_DW_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
OR_PROVIDERS="${TB_OR_PROVIDERS:-openrouter/deepinfra/fp4,openrouter/ionstream/fp4,openrouter/deepseek/fp8,openrouter/digitalocean}"
# No colon: an explicit empty TB_DW_PROVIDERS skips the DW half; unset uses the
# default. (With :- an empty value would fall through to the default.)
DW_PROVIDERS="${TB_DW_PROVIDERS-doubleword/realtime,doubleword/flex}"
TASKS="${TB_TASKS:-count-dataset-tokens,create-bucket,csv-to-parquet,extract-safely,fix-permissions,chess-best-move,conda-env-conflict-resolution,crack-7z-hash,crack-7z-hash.easy,crack-7z-hash.hard,cron-broken-network,decommissioning-service-with-sensitive-data,configure-git-webserver,git-multibranch}"
LOCAL_OUT="${TB_LOCAL_OUT:-artifacts/tb-dsflash-cloud}"

# Pull keys from .env if not already in the environment.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${DOUBLEWORD_API_KEY:?set DOUBLEWORD_API_KEY}"

gssh() { gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --command="$1" -- -o StrictHostKeyChecking=no -o ConnectTimeout=15; }

cleanup() {
  # Best-effort salvage BEFORE deleting: tar whatever exists on the VM and pull
  # the single file (a recursive scp of many small files dies on flaky networks
  # and leaves silent partial data). The happy path already pulled and verified
  # the tarball; this is only for early/failed exits.
  echo "== copy-back on exit (best effort) =="
  if [ ! -f /tmp/tb-results.tgz ]; then
    gssh "cd /opt/compound && tar czf tb-results.tgz '$LOCAL_OUT' 2>/dev/null" 2>/dev/null || true
    gcloud compute scp "$VM":/opt/compound/tb-results.tgz /tmp/tb-results.tgz \
      --zone="$ZONE" --project="$PROJECT" 2>/dev/null \
      && tar xzf /tmp/tb-results.tgz -C . 2>/dev/null || true
  fi
  # If the detached sweep is still mid-run, this exit is the CONTROLLER dying,
  # not the run finishing. Leave the VM alone: it holds the only copy of the
  # data and max-run-duration will reap it. Deleting here loses the run.
  if gssh "test -f /opt/compound/runall.sh && ! test -f /opt/compound/RUN_DONE" 2>/dev/null; then
    echo "== sweep still in progress; VM left alive (reaper bounds cost) =="
    return 0
  fi
  echo "== deleting VM $VM (cleanup) =="
  gcloud compute instances delete "$VM" --zone="$ZONE" --project="$PROJECT" -q 2>/dev/null || true
  echo "== remaining instances (should be empty) =="
  gcloud compute instances list --project="$PROJECT" 2>/dev/null || true
}
trap cleanup EXIT

STARTUP='#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io git curl ca-certificates tar rsync
systemctl enable --now docker
# Wait for the daemon to actually respond before signalling ready, then open the
# socket so the SSH login user (not in the docker group on a fresh image) can
# reach it. Gating the marker on docker avoids the "daemon not responding" race.
for i in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 3; done
chmod 666 /var/run/docker.sock || true
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh || true
mkdir -p /opt/compound && chmod 777 /opt/compound
docker info >/dev/null 2>&1 && touch /tmp/setup-done'

echo "== creating $VM ($MACHINE, 100GB, auto-delete after $MAX_DURATION) =="
# Pass the startup script via a file, not inline --metadata: gcloud splits
# --metadata values on commas, so any comma in the script (even in a comment)
# corrupts the metadata. --metadata-from-file has no such parsing.
STARTUP_FILE="$(mktemp)"; printf '%s\n' "$STARTUP" > "$STARTUP_FILE"
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --provisioning-model=STANDARD \
  --max-run-duration="$MAX_DURATION" --instance-termination-action=DELETE \
  --metadata-from-file=startup-script="$STARTUP_FILE"
rm -f "$STARTUP_FILE"

echo "== waiting for setup (docker + uv) =="
for i in $(seq 1 40); do
  if gssh "test -f /tmp/setup-done" 2>/dev/null; then echo "setup ready"; break; fi
  sleep 15
done

echo "== shipping working tree (no push needed) =="
tar czf /tmp/compound-src.tgz --exclude .git --exclude artifacts --exclude .venv \
  --exclude .compound --exclude '__pycache__' --exclude node_modules \
  src scripts benchmarks tests pyproject.toml uv.lock compound.yaml README.md
gcloud compute scp /tmp/compound-src.tgz "$VM":/opt/compound/ --zone="$ZONE" --project="$PROJECT"
gssh "cd /opt/compound && tar xzf compound-src.tgz && uv sync --extra dev >/tmp/uvsync.log 2>&1 || true"

TB_CONCURRENT="${TB_CONCURRENT:-2}"
TB_TRIALS="${TB_TRIALS:-1}"

# Run the sweep DETACHED on the VM and poll for a completion marker. A
# synchronous `gssh "...run-terminal-bench.sh..."` dies with the laptop's SSH
# connection (network change, sleep), taking the whole run down via set -e +
# the cleanup trap. Detached, the VM works autonomously; each poll is a fresh
# short SSH, so any number of connection drops cost nothing.
RUNALL="$(mktemp)"
{
  echo '#!/bin/bash'
  echo 'cd /opt/compound'
  echo 'rm -f RUN_DONE RUN_FAIL'
  echo '{'
  echo "  OPENROUTER_API_KEY='$OPENROUTER_API_KEY' TB_CONCURRENT='$TB_CONCURRENT' TB_TRIALS='$TB_TRIALS' \\"
  echo "    bash scripts/cloud/run-terminal-bench.sh '$MODEL' '$OR_PROVIDERS' '$TASKS' '$LOCAL_OUT' || touch RUN_FAIL"
  if [ -n "$DW_PROVIDERS" ]; then
    echo "  DOUBLEWORD_API_KEY='$DOUBLEWORD_API_KEY' TB_CONCURRENT='$TB_CONCURRENT' TB_TRIALS='$TB_TRIALS' \\"
    echo "    bash scripts/cloud/run-terminal-bench.sh '$DW_MODEL' '$DW_PROVIDERS' '$TASKS' '$LOCAL_OUT' || touch RUN_FAIL"
  fi
  echo '} > /opt/compound/run.log 2>&1'
  echo "tar czf /opt/compound/tb-results.tgz -C /opt/compound '$LOCAL_OUT' >> /opt/compound/run.log 2>&1"
  echo 'touch RUN_DONE'
} > "$RUNALL"
gcloud compute scp "$RUNALL" "$VM":/opt/compound/runall.sh --zone="$ZONE" --project="$PROJECT"
rm -f "$RUNALL"
echo "== launching detached sweep on the VM (survives SSH drops) =="
gssh "chmod +x /opt/compound/runall.sh && setsid nohup /opt/compound/runall.sh >/dev/null 2>&1 </dev/null & echo detached"

# Poll until the marker appears, a half fails, or we near the VM's own
# max-run-duration. SSH failures here are expected and harmless.
POLL_DEADLINE=$(( $(date +%s) + ${MAX_DURATION%s} - 600 ))
while [ "$(date +%s)" -lt "$POLL_DEADLINE" ]; do
  state="$(gssh 'test -f /opt/compound/RUN_DONE && echo DONE; test -f /opt/compound/RUN_FAIL && echo FAIL; tail -1 /opt/compound/run.log 2>/dev/null' 2>/dev/null || echo SSH-MISS)"
  echo "[poll $(date -u +%H:%M)] $state"
  case "$state" in *DONE*) break ;; esac
  sleep 180
done
case "${state-}" in
  *FAIL*) echo "== a sweep half FAILED on the VM; copying back what exists ==" ;;
  *DONE*) echo "== sweep complete ==" ;;
  *) echo "== poll deadline reached without completion; copying back partials ==" ;;
esac

# Pull ONE tarball (robust against flaky networks; a recursive scp of thousands
# of small files dies mid-transfer and leaves silent partial data), verify its
# contents, and only then let the VM be deleted. Retries survive connection
# resets; the VM (and the data) stays alive until verification passes.
echo "== pulling results tarball back =="
mkdir -p "$(dirname "$LOCAL_OUT")"
PULLED=""
for i in $(seq 1 10); do
  if gcloud compute scp "$VM":/opt/compound/tb-results.tgz /tmp/tb-results.tgz \
      --zone="$ZONE" --project="$PROJECT"; then PULLED=1; break; fi
  echo "scp attempt $i/10 failed; retrying in 30s"; sleep 30
done
if [ -z "$PULLED" ]; then
  echo "FATAL: could not pull results after 10 attempts; VM left for the"
  echo "max-run-duration reaper -- pull /opt/compound/tb-results.tgz manually."
  trap - EXIT   # do NOT delete the VM: it still holds the only copy
  exit 1
fi
GOT=$(tar tzf /tmp/tb-results.tgz | grep -c 'results\.json$' || true)
echo "== tarball verified: $GOT results.json inside =="
if [ "$GOT" -lt 1 ]; then
  echo "FATAL: tarball has no results; VM left alive for manual inspection."
  trap - EXIT
  exit 1
fi
tar xzf /tmp/tb-results.tgz -C .   # tarball paths already start with $LOCAL_OUT

echo "== done; cleanup trap will delete the VM =="
echo "then report: uv run python -m compound.tb_report $LOCAL_OUT --prices deepinfra-fp4=0.14,0.28"
