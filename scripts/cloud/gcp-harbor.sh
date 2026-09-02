#!/usr/bin/env bash
# One-shot: provision a Docker VM on GCP, run the Harbor (Terminal-Bench 4.0)
# provider sweep there, copy results back, and delete the VM.
#
# Why a VM rather than a laptop: TB4 gives each agent an 8-hour budget and the
# sweep runs several tasks concurrently, so a full grid means many long-lived
# containers and tens of GB of images. That is a poor fit for a workstation and
# a good fit for a machine that deletes itself afterwards.
#
# Cost is bounded three ways, as in gcp-terminal-bench.sh:
#   - Standard (non-preemptible) e2-standard-8, ~$0.27/hr.
#   - GCP server-side --max-run-duration with --instance-termination-action=DELETE,
#     so the VM reaps itself even if this script dies.
#   - A cleanup trap that deletes the VM as soon as results land.
# At the 12h default ceiling that is ~$3.20 of compute; expected far less.
#
# Run from the repo root with your key in the environment (or in .env):
#   GCP_PROJECT=... bash scripts/cloud/gcp-harbor.sh
#
# Verify nothing is left afterwards:  gcloud compute instances list
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT to your GCP project id}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM="${GCP_VM:-harbor-tb4}"
MACHINE="${GCP_MACHINE:-e2-standard-8}"
# TB4 task images are large and several run at once, so this is deliberately
# bigger than the TB1 runner's 100GB.
DISK="${GCP_DISK:-200GB}"
MAX_DURATION="${GCP_MAX_DURATION:-43200s}"

MODEL="${HB_MODEL:-z-ai/glm-5.3-flash}"
PROVIDERS="${HB_PROVIDERS:-openrouter/auto,openrouter/z-ai/fp8,openrouter/deepinfra/fp8,openrouter/novita/fp8,openrouter/parasail/fp8}"
TASKS="${HB_TASKS:-html-js-filter,photonic-waveguide-routing,music-harmony,bun-sourcemap-leak,foodstuff-beta-activity}"
DATASET="${HB_DATASET:-terminal-bench/terminal-bench@4.0.0}"
AGENT="${HB_AGENT:-terminus-2}"
# Equal turns rather than an equal clock: a wall-clock cap hands the faster host
# more turns and records a slower host's truncation as a failure it never
# earned. Duration is measured instead of being the thing that stops the run.
MAX_TURNS="${HB_MAX_TURNS:-100}"
# Per-host model ids for hosts that name the weights differently, as
# --host-model KEY=MODEL pairs separated by spaces, e.g.
#   HB_HOST_MODELS="doubleword=zai-org/GLM-5.3-Flash"
HOST_MODELS="${HB_HOST_MODELS:-}"
# Pin the reasoning mode on every arm (on|off); empty leaves each host's default.
REASONING="${HB_REASONING:-}"
ATTEMPTS="${HB_ATTEMPTS:-1}"
CONCURRENT="${HB_CONCURRENT:-5}"
LOCAL_OUT="${HB_LOCAL_OUT:-artifacts/tb4-cloud}"

SRC_TGZ="/tmp/compound-src-$VM.tgz"
RESULTS_TGZ="/tmp/harbor-results-$VM.tgz"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

# -n keeps ssh from holding stdin open, which is one way a remote background
# job can wedge the local client.
gssh() { gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --command="$1" -- -n -o StrictHostKeyChecking=no -o ConnectTimeout=15; }

cleanup() {
  # Salvage before deleting: tar on the VM and pull one file, because a
  # recursive scp of many small trial files dies on a flaky network and leaves
  # silent partial data.
  echo "== copy-back on exit (best effort) =="
  if [ ! -f "$RESULTS_TGZ" ]; then
    gssh "cd /opt/compound && tar czf harbor-results.tgz '$LOCAL_OUT' 2>/dev/null" 2>/dev/null || true
    gcloud compute scp "$VM":/opt/compound/harbor-results.tgz "$RESULTS_TGZ" \
      --zone="$ZONE" --project="$PROJECT" 2>/dev/null \
      && tar xzf "$RESULTS_TGZ" -C . 2>/dev/null || true
  fi
  # A still-running sweep means the CONTROLLER died, not the run. The VM holds
  # the only copy of the data, so leave it: max-run-duration bounds the cost.
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
# Created first, before the slow Docker install: the controller scps into this
# directory, and a half-provisioned VM should not fail that step.
mkdir -p /opt/compound && chmod 777 /opt/compound
apt-get update -y
apt-get install -y git curl ca-certificates tar rsync
# Docker from the official repo, NOT the distro docker.io package: TB4 task
# environments are compose-based and docker.io ships no Compose v2 plugin, so
# every trial dies with "Docker compose command failed". get.docker.com pulls
# docker-ce plus the buildx and compose plugins.
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sh /tmp/get-docker.sh
systemctl enable --now docker
docker compose version >/tmp/compose-version.txt 2>&1 || echo "COMPOSE MISSING" >/tmp/compose-version.txt
for i in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 3; done
chmod 666 /var/run/docker.sock || true
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh || true
# Gate the ready marker on compose too, so a missing plugin fails here rather
# than one trial at a time deep inside the sweep.
docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && touch /tmp/setup-done'

echo "== creating $VM ($MACHINE, $DISK, auto-delete after $MAX_DURATION) =="
# Pass the startup script via a file: gcloud splits --metadata values on commas,
# so any comma in the script corrupts it. --metadata-from-file does not parse.
STARTUP_FILE="$(mktemp)"; printf '%s\n' "$STARTUP" > "$STARTUP_FILE"
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size="$DISK" --boot-disk-type=pd-balanced \
  --provisioning-model=STANDARD \
  --max-run-duration="$MAX_DURATION" --instance-termination-action=DELETE \
  --metadata-from-file=startup-script="$STARTUP_FILE"
rm -f "$STARTUP_FILE"

echo "== waiting for setup (docker + uv) =="
# Docker from get.docker.com is slower to install than the distro package, so
# this waits longer than the TB1 runner. Falling through silently is worse than
# failing: the next scp would fail on a half-built VM and the real cause (still
# provisioning, or a broken install) would never be printed.
SETUP_OK=""
for i in $(seq 1 60); do
  if gssh "test -f /tmp/setup-done" 2>/dev/null; then SETUP_OK=1; echo "setup ready"; break; fi
  sleep 15
done
if [ -z "$SETUP_OK" ]; then
  echo "FATAL: setup never completed. VM startup log follows:"
  gssh "sudo journalctl -u google-startup-scripts --no-pager 2>/dev/null | tail -40; echo '--- compose ---'; cat /tmp/compose-version.txt 2>/dev/null" || true
  exit 1
fi

echo "== shipping working tree (no push needed) =="
tar czf "$SRC_TGZ" --exclude .git --exclude artifacts --exclude .venv \
  --exclude .compound --exclude '__pycache__' --exclude node_modules \
  src scripts benchmarks tests pyproject.toml uv.lock compound.yaml README.md
gcloud compute scp "$SRC_TGZ" "$VM":/opt/compound/compound-src.tgz --zone="$ZONE" --project="$PROJECT"
gssh "cd /opt/compound && tar xzf compound-src.tgz && uv sync --extra dev >/tmp/uvsync.log 2>&1 || true"

# Detached, so a dropped SSH connection cannot take the run down. Each poll is
# a fresh short SSH, so any number of drops cost nothing.
RUNALL="$(mktemp)"
{
  echo '#!/bin/bash'
  echo 'cd /opt/compound'
  echo 'rm -f RUN_DONE RUN_FAIL'
  echo '{'
  echo "  OPENROUTER_API_KEY='$OPENROUTER_API_KEY' DOUBLEWORD_API_KEY='${DOUBLEWORD_API_KEY:-}' \\"
  echo "    uv run python -m compound.bench harbor \\"
  echo "      --providers '$PROVIDERS' --model '$MODEL' \\"
  for pair in $HOST_MODELS; do echo "      --host-model '$pair' \\"; done
  [ -n "$REASONING" ] && echo "      --reasoning '$REASONING' \\"
  echo "      --dataset '$DATASET' --agent '$AGENT' \\"
  echo "      --tasks '$TASKS' --attempts '$ATTEMPTS' \\"
  echo "      --n-concurrent '$CONCURRENT' --ak max_turns='$MAX_TURNS' \\"
  echo "      --jobs-dir '$LOCAL_OUT' --ledger-dir '$LOCAL_OUT/ledger' \\"
  echo "      --go || touch RUN_FAIL"
  echo '} > /opt/compound/run.log 2>&1'
  echo "tar czf /opt/compound/harbor-results.tgz -C /opt/compound '$LOCAL_OUT' >> /opt/compound/run.log 2>&1"
  echo 'touch RUN_DONE'
} > "$RUNALL"
gcloud compute scp "$RUNALL" "$VM":/opt/compound/runall.sh --zone="$ZONE" --project="$PROJECT"
rm -f "$RUNALL"
# systemd-run, not `nohup ... &`: a backgrounded remote job keeps the SSH
# channel open, so the client hangs forever even after the job is launched
# (observed: an 18-minute hang while the sweep itself ran normally). A transient
# unit detaches properly and the ssh call returns at once.
gssh "chmod +x /opt/compound/runall.sh && sudo systemd-run --unit=harbor-sweep --collect \
  --property=WorkingDirectory=/opt/compound /opt/compound/runall.sh && echo started"

echo "== polling until the sweep finishes =="
while true; do
  if gssh "test -f /opt/compound/RUN_DONE" 2>/dev/null; then echo "sweep done"; break; fi
  CALLS=$(gssh "cat /opt/compound/$LOCAL_OUT/ledger/*.jsonl 2>/dev/null | wc -l" 2>/dev/null || echo "?")
  echo "  $(date -u +%H:%M:%S) still running; ledger calls so far: $CALLS"
  sleep 120
done

echo "== pulling results tarball back =="
mkdir -p "$(dirname "$LOCAL_OUT")"
PULLED=""
for i in $(seq 1 10); do
  if gcloud compute scp "$VM":/opt/compound/harbor-results.tgz "$RESULTS_TGZ" \
      --zone="$ZONE" --project="$PROJECT"; then PULLED=1; break; fi
  echo "scp attempt $i/10 failed; retrying in 30s"; sleep 30
done
if [ -z "$PULLED" ]; then
  echo "FATAL: could not pull results; VM left alive for manual retrieval."
  trap - EXIT
  exit 1
fi
GOT=$(tar tzf "$RESULTS_TGZ" | grep -c 'result\.json$' || true)
echo "== tarball verified: $GOT result.json inside =="
tar xzf "$RESULTS_TGZ" -C .

echo "== done; cleanup trap will delete the VM =="
echo "then report: uv run python -m compound.bench ledger $LOCAL_OUT/ledger/<host>.jsonl --hosts"
