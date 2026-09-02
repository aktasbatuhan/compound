title: Terminal-Bench 4.0
order: 40

# Terminal-Bench 4.0 through Harbor

Terminal-Bench 4.0 is distributed as a [Harbor](https://harborframework.com)
dataset. `compound-bench harbor` runs it across pinned hosts: one Harbor job per
host, each behind its own proxy, with a call ledger per host.

```bash
compound-bench harbor \
    --model z-ai/glm-5.3-flash \
    --providers openrouter/auto,openrouter/deepinfra/fp8,doubleword/realtime,doubleword/flex \
    --host-model doubleword=zai-org/GLM-5.3-Flash --reasoning on \
    --dataset terminal-bench@4.0.0 \
    --n-tasks 10 --attempts 2 --n-concurrent 4 \
    --ak max_turns=100 \
    --ledger-dir artifacts/tb4/ledger --go
```

Without `--go` it prints the grid and the exact Harbor command per host.

## Flags that matter

- `--dataset name@version`. Pin a version, never `@latest`, so the task set cannot shift between arms of one experiment.
- `--agent`. Defaults to `terminus-2`. Must be a terminus-family agent when pinning: those run in the harness process and can reach the proxy. In-sandbox agents cannot, and the CLI refuses to pin them.
- `--tasks` accepts glob patterns. Harbor namespaces task names (`terminal-bench/hello-world`), so a bare name is widened to `*/name`.
- `--ak max_turns=N`. Give every host the same amount of work. An equal wall clock hands a faster host more turns and records a slower host's truncation as a failure.
- `--agent-timeout-multiplier`. Scales only how long the agent may work. TB4 allows 8 hours per task by default.
- `--timeout-multiplier`. Scales every phase, including environment build. Usually not what you want: a small value can kill containers before they finish building.
- `--host-model HOST=MODEL`, repeatable. Hosts name the same weights differently: OpenRouter serves `z-ai/glm-5.3-flash`, Doubleword serves `zai-org/GLM-5.3-Flash`. The proxy rewrites the model id for that host; `HOST` is a provider token, label, or kind (`doubleword=zai-org/GLM-5.3-Flash`).
- `--reasoning on|off`. Pin the reasoning mode on every arm. Hosts disagree on the default, and an unpinned comparison mixes modes.
- `--env docker|modal|daytona`. Harbor's sandbox backend.

## Requirements

- Docker with Compose v2. On Ubuntu the `docker.io` package lacks Compose; install from `get.docker.com`.
- Disk. TB4 task images are large and several run at once. Budget 100 GB for a five-task grid.
- Python 3.12 or newer for Harbor itself (`uvx --from harbor`).

## Running on a cloud VM

TB4 grids are a poor fit for a laptop. `scripts/cloud/gcp-harbor.sh` provisions a
GCP VM, installs Docker, runs one arm, copies results back, and deletes the VM.
It bounds cost three ways: a standard VM type, a server-side maximum run
duration with delete-on-termination, and a cleanup trap.

```bash
GCP_PROJECT=<project> bash scripts/cloud/gcp-harbor.sh   # one arm per VM
gcloud compute instances list                            # verify nothing is left
```

To compare hosts fairly, run every arm at the same time on separate VMs.
Serving-host congestion moves by the hour, so arms run one after another
confound host with time of day.

## Reading the results

Harbor writes a job directory per arm with a `result.json` per trial. The
ledger directory gets `<host>.jsonl`. Then:

```bash
compound-bench ledger artifacts/tb4/ledger/deepinfra-fp8.jsonl --hosts
python3 scripts/analyze_arms.py artifacts/tb4          # arms side by side
```

See [Reports and ledgers](../reports/) for what each column means.
