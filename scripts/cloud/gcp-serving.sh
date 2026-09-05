#!/usr/bin/env bash
# Serving comparison from one cloud VM: same network position for every host.
#
# Latency is measured relative to wherever the client sits, so a comparison run
# from a laptop adds that laptop's round trip to every host equally. That is fine
# for ranking hosts and wrong for comparing against someone else's published
# numbers. Telnyx's DeepSeek V4 Flash comparison ran from us-east-2, so this runs
# from us-east too and says so, rather than quietly reporting numbers that cannot
# be read against theirs.
#
# No Docker: the serving harness only makes HTTPS calls, so this VM is a bare
# python box and starts in about a minute.
#
# Four passes, because cold and warm cannot share an invocation: warm cells have
# to run serially so the first repetition populates the prefix cache the rest are
# meant to read, and a mixed run would race that write and understate the hit
# rate. Small and large profiles are split so the expensive 100k rows can take a
# smaller sample without dragging the cheap rows down with them.
#
#   GCP_PROJECT=... bash scripts/cloud/gcp-serving.sh
#
# Verify nothing is left afterwards:  gcloud compute instances list
set -uo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
ZONE="${GCP_ZONE:-us-east1-b}"
MACHINE="${GCP_MACHINE:-e2-standard-4}"
MAX_DURATION="${GCP_MAX_DURATION:-21600s}"

MODEL_OR="${SERVING_MODEL_OR:-deepseek/deepseek-v4-flash-0731}"
MODEL_DW="${SERVING_MODEL_DW:-deepseek-ai/DeepSeek-V4-Flash-0731}"

# Probed live before the run. Two of the three hosts Telnyx compared against
# (baseten, fireworks) return 429 on OpenRouter's shared pool, so they are absent
# here: that is our access path, not their capacity. The list deliberately spans
# the three precisions OpenRouter labels for this model, plus DeepSeek's own
# first-party endpoint as the reference, so "some providers quantize" is a
# measured variable rather than an assumption.
ROUTES="${SERVING_ROUTES:-openrouter/deepseek,openrouter/morph/bf16,openrouter/together,openrouter/novita/fp8,openrouter/siliconflow/fp8,openrouter/gmicloud/fp8,openrouter/parasail/fp8,openrouter/coreweave/fp8,openrouter/relace/fp4,openrouter/atlas-cloud/fp4,openrouter/auto,direct/telnyx,doubleword/realtime,doubleword/flex}"

REPS_SMALL="${SERVING_REPS_SMALL:-100}"   # 1k and 10k profiles: cheap, buy a real p90
REPS_LARGE="${SERVING_REPS_LARGE:-30}"    # 100k profiles: ~90% of the token bill
LARGE_OUTPUTS="${SERVING_LARGE_OUTPUTS:-100,1000}"
# Semicolon-separated "<shapes> <cache-mode> <reps>" passes. The default is the
# full grid; a single pass reruns one cell, e.g. calls a router refused for a
# reason that was ours (a 402 on our balance) rather than the host's.
PASSES="${SERVING_PASSES:-small cold $REPS_SMALL;small warm $REPS_SMALL;large cold $REPS_LARGE;large warm $REPS_LARGE}"
STAMP="$(date +%s)"
OUT_ROOT="${SERVING_OUT:-artifacts/serving-$STAMP}"
VM="serving-$STAMP"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${DOUBLEWORD_API_KEY:?set DOUBLEWORD_API_KEY}"
: "${TELNYX_API_KEY:?set TELNYX_API_KEY}"

mkdir -p "$OUT_ROOT"
echo "== serving comparison from $ZONE"
echo "== model:  $MODEL_OR"
echo "== routes: $(echo "$ROUTES" | tr ',' '\n' | wc -l | tr -d ' ')"
echo "== out:    $OUT_ROOT"

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
apt-get install -y git curl ca-certificates tar python3 python3-pip
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh || true
touch /tmp/setup-done'

STARTUP_FILE="$(mktemp)"; printf '%s\n' "$STARTUP" > "$STARTUP_FILE"

echo "== creating $VM =="
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB --boot-disk-type=pd-standard \
  --provisioning-model=STANDARD \
  --max-run-duration="$MAX_DURATION" --instance-termination-action=DELETE \
  --metadata-from-file=startup-script="$STARTUP_FILE" >/dev/null 2>&1
rm -f "$STARTUP_FILE"

echo "== waiting for the box =="
for i in $(seq 1 40); do
  gssh "test -f /tmp/setup-done" >/dev/null 2>&1 && break
  sleep 10
done
gssh "test -f /tmp/setup-done" >/dev/null 2>&1 && echo "  ready" || echo "  NOT READY (trying anyway)"

SRC_TGZ="/tmp/compound-serving-$STAMP.tgz"
tar czf "$SRC_TGZ" src pyproject.toml uv.lock compound.yaml scripts 2>/dev/null
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
  uv sync
  # The grid, generated on the box so the bytes are identical to what the
  # calibration verified rather than whatever a local copy happens to hold.
  uv run python scripts/make_profile_shapes.py --out small.json --inputs 1000,10000 --outputs 100,1000
  uv run python scripts/make_profile_shapes.py --out large.json --inputs 100000 --outputs $LARGE_OUTPUTS

  IFS=';' read -ra PASSES <<< '$PASSES'
  for pass in "\${PASSES[@]}"; do
    set -- \$pass
    SHAPES=\$1; CMODE=\$2; REPS=\$3
    echo "===== PASS \$SHAPES/\$CMODE reps=\$REPS \$(date -u +%H:%M:%S) ====="
    OPENROUTER_API_KEY='$OPENROUTER_API_KEY' DOUBLEWORD_API_KEY='$DOUBLEWORD_API_KEY' \
    TELNYX_API_KEY='$TELNYX_API_KEY' \
    uv run python -m compound.bench serving --go \
      --providers '$ROUTES' \
      --shapes "\$SHAPES.json" \
      --model-or '$MODEL_OR' --model '$MODEL_DW' \
      --reps "\$REPS" --cache-mode "\$CMODE" \
      --reasoning-modes off --temperature 0 \
      --out "out/\$SHAPES-\$CMODE"
    echo "===== exit \$? \$(date -u +%H:%M:%S) ====="
  done
} > /opt/compound/serving.log 2>&1
tar czf /opt/compound/results.tgz out serving.log small.json large.json 2>/dev/null
touch /opt/compound/RUN_DONE
EOF
gcloud compute scp "$RUNNER" "$VM":/opt/compound/run.sh --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
rm -f "$RUNNER"

echo "== launching =="
gssh "chmod +x /opt/compound/run.sh && sudo systemd-run --unit=serving --collect /opt/compound/run.sh" >/dev/null 2>&1

echo "== waiting (poll every 60s) =="
for i in $(seq 1 240); do
  gssh "test -f /opt/compound/RUN_DONE" >/dev/null 2>&1 && { echo "  done after ~$i min"; break; }
  sleep 60
done

echo "== collecting =="
gcloud compute scp "$VM":/opt/compound/results.tgz "$OUT_ROOT/results.tgz" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
tar xzf "$OUT_ROOT/results.tgz" -C "$OUT_ROOT" 2>/dev/null
cat "$OUT_ROOT"/out/*/results.jsonl > "$OUT_ROOT/all.jsonl" 2>/dev/null
echo "== $(wc -l < "$OUT_ROOT/all.jsonl" 2>/dev/null || echo 0) calls -> $OUT_ROOT/all.jsonl"
echo "== analyze with: python3 scripts/analyze_serving.py $OUT_ROOT/all.jsonl"
