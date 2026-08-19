#!/usr/bin/env bash
# Run a terminal-bench provider sweep on a fresh Docker-capable VM.
#
# terminal-bench spawns a Docker container per task, which needs a real Docker
# daemon and tens of GB of disk -- more than a laptop comfortably spares when
# many task images build at once. This script provisions the run on a VM that
# has both, then leaves the results (small JSON) to copy back.
#
# Assumes: Ubuntu 22.04+, Docker installed and running, this repo checked out,
# and OPENROUTER_API_KEY / DOUBLEWORD_API_KEY exported. Idempotent: safe to
# re-run; terminal-bench resumes finished tasks.
#
# Usage (on the VM, from the repo root):
#   OPENROUTER_API_KEY=... DOUBLEWORD_API_KEY=... \
#     scripts/cloud/run-terminal-bench.sh <model> <providers> <tasks> [out_dir]
#
# Example:
#   scripts/cloud/run-terminal-bench.sh \
#     deepseek/deepseek-v4-flash-0731 \
#     openrouter/deepinfra/fp4,openrouter/ionstream/fp4,openrouter/deepseek/fp8,openrouter/digitalocean \
#     create-bucket,csv-to-parquet,fix-permissions,chess-best-move,conda-env-conflict-resolution
set -euo pipefail

MODEL="${1:?model, e.g. deepseek/deepseek-v4-flash-0731}"
PROVIDERS="${2:?comma-separated provider tokens}"
TASKS="${3:?comma-separated terminal-bench task ids}"
OUT="${4:-artifacts/tb-cloud}"

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v uv >/dev/null || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
# A fresh VM's daemon may still be starting, or its socket may not be reachable
# by this (non-docker-group) login user. Wait, opening the socket if we can.
for i in $(seq 1 30); do
  docker info >/dev/null 2>&1 && break
  sudo chmod 666 /var/run/docker.sock 2>/dev/null || true
  echo "waiting for docker daemon ($i/30)"; sleep 3
done
docker info >/dev/null 2>&1 || { echo "docker daemon not responding after ~90s"; exit 1; }
# Ubuntu's docker.io ships WITHOUT the Compose v2 plugin, but terminal-bench
# drives every task via `docker compose build/up`. Drop the plugin in if missing.
if ! docker compose version >/dev/null 2>&1; then
  echo "installing docker compose v2 plugin"
  mkdir -p "$HOME/.docker/cli-plugins"
  curl -sSL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
    -o "$HOME/.docker/cli-plugins/docker-compose"
  chmod +x "$HOME/.docker/cli-plugins/docker-compose"
fi

# One-time: fetch the terminal-bench task dataset and build the manifest.
if [ ! -d .compound/sources/terminal-bench-core ]; then
  echo "== downloading terminal-bench-core dataset =="
  uvx --with 'litellm<1.95' terminal-bench@0.2.18 datasets download \
    --name terminal-bench-core --version 0.1.1 \
    --output-dir .compound/sources/terminal-bench-core
fi
uv run python -m compound.bench prepare terminal_bench >/dev/null 2>&1 || true

echo "== running sweep: model=$MODEL providers=$PROVIDERS trials=${TB_TRIALS:-1} =="
# Each behind its own pinning proxy. TB_TRIALS repeats the whole sweep into a
# per-trial subdir (terminal-bench is high-variance, so multiple trials give a
# resolve-rate rather than a single noisy pass/fail).
for t in $(seq 1 "${TB_TRIALS:-1}"); do
  echo "== trial $t/${TB_TRIALS:-1} =="
  PYTHONPATH=src uv run python -m compound.bench run terminal_bench \
    --model "$MODEL" --providers "$PROVIDERS" --tasks "$TASKS" \
    --tb-concurrent "${TB_CONCURRENT:-2}" --output "$OUT/trial-$t" --go
  # Prune after EVERY trial, not just at the end: a multi-trial sweep otherwise
  # accumulates task images/containers until the disk fills and the run stalls.
  echo "== reclaiming Docker space after trial $t =="
  docker container prune -f >/dev/null 2>&1 || true
  docker image prune -af >/dev/null 2>&1 || true
done

echo "== results under $OUT (copy back: gcloud compute scp --recurse ...:$OUT ./) =="
# `find | head` can return non-zero under `pipefail` (SIGPIPE when head exits),
# which would abort the caller before the next sweep / the results copy-back.
find "$OUT" -name results.json 2>/dev/null | head -20 || true
exit 0
