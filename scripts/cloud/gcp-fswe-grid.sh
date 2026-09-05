#!/usr/bin/env bash
# One VM per serving host, all started within the same minute, each running the
# same hand-picked tasks from a Harbor task tree that lives in a git checkout
# rather than in the hub (FrontierSWE v2 ships tasks but not its runner).
#
# Why one VM per arm rather than one VM running the arms in turn: serving-host
# congestion moves by the hour, so arms run back to back confound the host with
# the time of day. In the 2026-09-01 routing run the same unpinned arm measured
# a 14.6% stall rate in one window and 42% in another. Concurrent arms share
# whatever the network is doing.
#
# Both models run on the SAME VM, one after the other, because the task images
# are the expensive part and they do not depend on the model: build once, then
# sweep the models over the built environments.
#
# Cost is bounded three ways, as in gcp-harbor.sh: standard (non-preemptible)
# VMs, a server-side --max-run-duration with --instance-termination-action=DELETE
# so each VM reaps itself even if this controller dies, and a cleanup trap.
#
#   GCP_PROJECT=... bash scripts/cloud/gcp-fswe-grid.sh
#
# Verify nothing is left afterwards:  gcloud compute instances list
set -uo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE="${GCP_MACHINE:-e2-standard-4}"
# pd-standard, not pd-balanced: pd-balanced counts against SSD_TOTAL_GB, which is
# 500GB per region and cannot hold eight VMs. DISKS_TOTAL_GB is 4096. 200GB also
# clears the <200GB poor-I/O warning, which matters when the night is mostly
# docker build.
DISK="${GCP_DISK:-200GB}"
DISK_TYPE="${GCP_DISK_TYPE:-pd-standard}"
MAX_DURATION="${GCP_MAX_DURATION:-32400s}"

REPO_URL="${FSWE_REPO:-https://github.com/Proximal-Labs/frontier-swe-v2.git}"
# Pin the commit: the repo is days old and "work in progress", so an arm that
# cloned an hour later must not silently run different tasks.
REPO_COMMIT="${FSWE_COMMIT:-}"
# Three tasks that built and ran on every arm of the 2026-09-03 grid, all in the
# CPU-only tier. lean-4-kernel-type-checker-in-pascal fails its compose build on
# every arm; qubit-routing declares exactly 16 GB, leaving a 16 GB VM none.
TASKS="${FSWE_TASKS:-crash-proof-flash-filesystem,verilog-simulator-in-swift,libexpat-optimization}"
# "<label>:<openrouter id>:<doubleword id>" per model, space separated.
MODELS="${FSWE_MODELS:-glm53flash:z-ai/glm-5.3-flash:zai-org/GLM-5.3-Flash deepseekv4flash:deepseek/deepseek-v4-flash-0731:deepseek-ai/DeepSeek-V4-Flash-0731}"
# Probe before trusting this list: without our own upstream keys these arms sit
# on OpenRouter's shared rate-limit pool, which moves hourly and is per
# (upstream, model). On 2026-09-03 deepinfra and baseten were listed as up and
# returned 429 on every call for one model while serving the other, which cost
# two arms mid-grid. `compound-bench providers <model> --probe --go` settles it.
ARMS="${FSWE_ARMS:-openrouter/auto openrouter/novita/fp8 openrouter/siliconflow/fp8 openrouter/gmicloud/fp8 openrouter/parasail/fp8 openrouter/together doubleword/realtime doubleword/flex}"
# Which models the DOUBLEWORD arms may run. `dw usage` reports billed cost by
# model but never by tier, and it truncates its window to the calendar day, so
# the realtime/flex split is only recoverable when a UTC day's Doubleword traffic
# involves ONE model: then the window's estimated_realtime_cost is attributable
# and scripts/dw_tier_cost.py can separate the tiers. Defaults to the first model;
# run the grid again the next UTC day with the second to price both.
# The OpenRouter arms are unaffected and still run every model.
DW_MODELS="${FSWE_DW_MODELS:-$(echo "$MODELS" | awk '{print $1}')}"
ATTEMPTS="${FSWE_ATTEMPTS:-1}"
MAX_TURNS="${FSWE_TURNS:-40}"
# Tasks declare a 20-hour agent budget, so this multiplier is the per-task
# ceiling: 0.05 is 60 minutes on every task in the CPU-only tier.
#
# It was 0.02 (24 minutes) on the first grid, and that turned out to bind: 24 of
# 64 trials on the working tasks ended in AgentTimeoutError rather than finishing
# their turns. A truncated agent cannot be scored, so the clock has to sit above
# what max_turns needs. max_turns stays the real control, because an equal turn
# budget gives every host the same work while an equal clock quietly hands the
# faster host more of it.
AGENT_MULT="${FSWE_AGENT_MULT:-0.05}"
STAMP="$(date +%s)"
OUT_ROOT="${FSWE_OUT:-artifacts/fswe-$STAMP}"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${DOUBLEWORD_API_KEY:?set DOUBLEWORD_API_KEY}"

mkdir -p "$OUT_ROOT"
echo "== grid: $(echo "$ARMS" | wc -w | tr -d ' ') arms x $(echo "$MODELS" | wc -w | tr -d ' ') models x $(echo "$TASKS" | tr ',' ' ' | wc -w | tr -d ' ') tasks"
echo "== tasks: $TASKS"
echo "== doubleword arms run only: $(echo "$DW_MODELS" | tr ' ' ',' )"
echo "== out:   $OUT_ROOT"

vm_name() { echo "fswe-$(echo "$1" | tr '/' '-' | tr '[:upper:]' '[:lower:]')-$STAMP"; }

gssh() {
  gcloud compute ssh "$1" --zone="$ZONE" --project="$PROJECT" --command="$2" \
    -- -n -o StrictHostKeyChecking=no -o ConnectTimeout=15
}

STARTUP='#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
mkdir -p /opt/compound && chmod 777 /opt/compound
apt-get update -y
apt-get install -y git curl ca-certificates tar rsync
# Docker from the official repo, not distro docker.io: Harbor task environments
# are compose-based and docker.io ships no Compose v2 plugin.
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sh /tmp/get-docker.sh
systemctl enable --now docker
for i in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 3; done
chmod 666 /var/run/docker.sock || true
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh || true
docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && touch /tmp/setup-done'

STARTUP_FILE="$(mktemp)"; printf '%s\n' "$STARTUP" > "$STARTUP_FILE"
SRC_TGZ="/tmp/compound-src-$STAMP.tgz"
tar czf "$SRC_TGZ" src pyproject.toml uv.lock compound.yaml benchmarks scripts 2>/dev/null

declare -a VMS=()
for ARM in $ARMS; do
  VM="$(vm_name "$ARM")"
  VMS+=("$VM:$ARM")
  echo "== creating $VM for $ARM =="
  gcloud compute instances create "$VM" \
    --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
    --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
    --boot-disk-size="$DISK" --boot-disk-type="$DISK_TYPE" \
    --provisioning-model=STANDARD \
    --max-run-duration="$MAX_DURATION" --instance-termination-action=DELETE \
    --metadata-from-file=startup-script="$STARTUP_FILE" >/dev/null 2>&1 &
done
wait
rm -f "$STARTUP_FILE"
echo "== all VMs requested; waiting for docker on each =="

for entry in "${VMS[@]}"; do
  VM="${entry%%:*}"
  for i in $(seq 1 60); do
    gssh "$VM" "test -f /tmp/setup-done" >/dev/null 2>&1 && break
    sleep 15
  done
  gssh "$VM" "test -f /tmp/setup-done" >/dev/null 2>&1 \
    && echo "  ready: $VM" \
    || echo "  NOT READY: $VM (will still try)"
done

echo "== staging source and launching arms =="
for entry in "${VMS[@]}"; do
  VM="${entry%%:*}"; ARM="${entry#*:}"
  (
    gcloud compute scp "$SRC_TGZ" "$VM":/opt/compound/src.tgz --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
    # Doubleword arms run a single model so the day's billing window stays
    # attributable; every other arm runs the full model list.
    case "$ARM" in
      doubleword/*) ARM_MODELS="$DW_MODELS" ;;
      *)            ARM_MODELS="$MODELS" ;;
    esac
    RUNNER="$(mktemp)"
    cat > "$RUNNER" <<EOF
#!/bin/bash
cd /opt/compound
rm -f RUN_DONE RUN_FAIL
{
  set -x
  tar xzf src.tgz
  [ -d repo ] || git clone --depth 1 "$REPO_URL" repo || { echo CLONE_FAIL; touch /opt/compound/RUN_FAIL; }
  if [ -n "$REPO_COMMIT" ]; then (cd repo && git fetch --depth 1 origin "$REPO_COMMIT" && git checkout "$REPO_COMMIT"); fi
  (cd repo && git rev-parse HEAD > /opt/compound/REPO_COMMIT.txt)
  uv sync --extra dev
  # Give Harbor the separate-verifier build context FrontierSWE ships no
  # equivalent of. Without this the agent phase runs fine and every task then
  # fails at scoring, which is how the first grid produced no quality signal.
  for TASK in \$(echo "$TASKS" | tr ',' ' '); do
    python3 scripts/fswe_prepare.py repo --task "\$TASK" || touch /opt/compound/RUN_FAIL
  done
  for MODELSPEC in $ARM_MODELS; do
    MKEY="\${MODELSPEC%%:*}"; REST="\${MODELSPEC#*:}"
    ORID="\${REST%%:*}"; DWID="\${REST#*:}"
    for TASK in \$(echo "$TASKS" | tr ',' ' '); do
      echo "===== ARM=$ARM MODEL=\$MKEY TASK=\$TASK \$(date -u +%H:%M:%S) ====="
      OPENROUTER_API_KEY='$OPENROUTER_API_KEY' DOUBLEWORD_API_KEY='$DOUBLEWORD_API_KEY' \
      timeout 9000 uv run python -m compound.bench harbor \
        --task-path "repo/tasks/\$TASK" \
        --model "\$ORID" --host-model "doubleword=\$DWID" \
        --providers '$ARM' \
        --attempts $ATTEMPTS --n-concurrent 1 \
        --ak max_turns=$MAX_TURNS \
        --agent-timeout-multiplier $AGENT_MULT \
        --jobs-dir "out/\$MKEY/\$TASK/jobs" \
        --ledger-dir "out/\$MKEY/\$TASK/ledger" \
        --go
      echo "===== exit \$? df: \$(df -h / | tail -1) ====="
    done
  done
} > /opt/compound/run.log 2>&1
tar czf /opt/compound/results.tgz out REPO_COMMIT.txt run.log 2>/dev/null
touch /opt/compound/RUN_DONE
EOF
    gcloud compute scp "$RUNNER" "$VM":/opt/compound/runall.sh --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
    rm -f "$RUNNER"
    gssh "$VM" "chmod +x /opt/compound/runall.sh && sudo systemd-run --unit=fswe-arm --collect /opt/compound/runall.sh" >/dev/null 2>&1
    echo "  launched: $ARM on $VM"
  ) &
done
wait
rm -f "$SRC_TGZ"

echo "== all arms launched at $(date -u +%H:%M:%SZ). Poll with:"
echo "   gcloud compute instances list --project=$PROJECT"
printf '%s\n' "${VMS[@]}" > "$OUT_ROOT/vms.txt"
echo "== VM list written to $OUT_ROOT/vms.txt"
