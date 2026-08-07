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
MODEL="${TB_MODEL:-deepseek/deepseek-v4-flash-0731}"
DW_MODEL="${TB_DW_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
OR_PROVIDERS="${TB_OR_PROVIDERS:-openrouter/deepinfra/fp4,openrouter/ionstream/fp4,openrouter/deepseek/fp8,openrouter/digitalocean}"
DW_PROVIDERS="${TB_DW_PROVIDERS:-doubleword/realtime,doubleword/flex}"
TASKS="${TB_TASKS:-count-dataset-tokens,create-bucket,csv-to-parquet,extract-safely,fix-permissions,chess-best-move,conda-env-conflict-resolution,crack-7z-hash,crack-7z-hash.easy,crack-7z-hash.hard,cron-broken-network,decommissioning-service-with-sensitive-data,configure-git-webserver,git-multibranch}"
LOCAL_OUT="${TB_LOCAL_OUT:-artifacts/tb-dsflash-cloud}"

# Pull keys from .env if not already in the environment.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${DOUBLEWORD_API_KEY:?set DOUBLEWORD_API_KEY}"

gssh() { gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --command="$1" -- -o StrictHostKeyChecking=no -o ConnectTimeout=15; }

cleanup() {
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
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh || true
mkdir -p /opt/compound && chmod 777 /opt/compound
touch /tmp/setup-done'

echo "== creating $VM ($MACHINE, 100GB, 4h auto-delete) =="
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --provisioning-model=STANDARD \
  --max-run-duration=14400s --instance-termination-action=DELETE \
  --metadata=startup-script="$STARTUP"

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

echo "== running the sweep on the VM (OpenRouter hosts) =="
gssh "cd /opt/compound && OPENROUTER_API_KEY='$OPENROUTER_API_KEY' \
  bash scripts/cloud/run-terminal-bench.sh '$MODEL' '$OR_PROVIDERS' '$TASKS' '$LOCAL_OUT'"

echo "== running the sweep on the VM (Doubleword tiers) =="
gssh "cd /opt/compound && DOUBLEWORD_API_KEY='$DOUBLEWORD_API_KEY' \
  bash scripts/cloud/run-terminal-bench.sh '$DW_MODEL' '$DW_PROVIDERS' '$TASKS' '$LOCAL_OUT'"

echo "== pulling results back to $LOCAL_OUT =="
mkdir -p "$LOCAL_OUT"
gcloud compute scp --recurse "$VM":/opt/compound/"$LOCAL_OUT" "$(dirname "$LOCAL_OUT")" \
  --zone="$ZONE" --project="$PROJECT" || true

echo "== done; cleanup trap will delete the VM =="
echo "then report: uv run python -m compound.tb_report $LOCAL_OUT --prices deepinfra-fp4=0.14,0.28"
