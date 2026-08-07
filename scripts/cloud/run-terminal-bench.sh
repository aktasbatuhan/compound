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
docker info >/dev/null 2>&1 || { echo "docker daemon not responding"; exit 1; }
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

echo "== running sweep: model=$MODEL providers=$PROVIDERS =="
# Hosts run in parallel; each behind its own pinning proxy. Disk on a VM is
# ample, so the default per-host concurrency is fine.
PYTHONPATH=src uv run python -m compound.bench run terminal_bench \
  --model "$MODEL" --providers "$PROVIDERS" --tasks "$TASKS" \
  --tb-concurrent 2 --output "$OUT" --go

echo "== reclaiming Docker space between sweeps =="
docker container prune -f >/dev/null 2>&1 || true
docker image prune -af >/dev/null 2>&1 || true

echo "== results under $OUT (copy back: gcloud compute scp --recurse ...:$OUT ./) =="
find "$OUT" -name results.json | head
