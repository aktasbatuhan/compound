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
- `--task-path <dir>`. Run a Harbor task or dataset directory that sits on disk instead of one resolved from the hub. A benchmark that ships Harbor-schema tasks but no runner of its own is still runnable this way: clone it, pin a commit, and point at a task directory. Mutually exclusive with `--dataset`.
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

## Task trees that ship no runner

`--task-path` runs a Harbor-schema task directory straight from a git checkout,
which is how a benchmark that publishes tasks but not its runner can still be
run. FrontierSWE v2 is the case that motivated it.

One thing to check before trusting such a run: whether the tasks can actually be
**graded**. A task whose `[verifier]` sets `environment_mode = "separate"` has
Harbor build the verifier in its own container, using `<task>/tests` as the build
context, and in that mode Harbor uploads nothing into the container, so the image
must already contain `/tests/test.sh`. Benchmarks whose own uploader performs that
step ship no `<task>/tests` at all, and the failure surfaces only at scoring
time:

```
FileNotFoundError: Task environment directory <task>/tests has no environment definition.
```

The agent phase is unaffected, so the run looks healthy and yields no reward on
any task. `scripts/fswe_prepare.py` closes the gap for FrontierSWE v2 by
materializing `<task>/tests` as a copy of `<task>/environment` plus a shim that
forwards to the grader the image already carries. It changes no scoring code.

```bash
python3 scripts/fswe_prepare.py /path/to/frontier-swe-v2 --all-cpu-tasks
```

Validate before spending model tokens. Each task ships a reference solution and
an `oracle_reward_threshold`, and Harbor's `oracle` agent runs that solution
instead of a model, so one VM tells you whether grading works end to end:

```bash
GCP_PROJECT=<project> bash scripts/cloud/gcp-fswe-oracle.sh
```

One catch specific to FrontierSWE: its `solve.sh` and its verifier both key off
`HARBOR_ORACLE_FLAG`, which is px-eval's variable and appears nowhere in Harbor.
Without it `solve.sh` exits immediately and the oracle scores 0 however healthy
the pipeline is, so the script generates a random flag per run and passes it to
both phases with `--ae` and `--ve`. Never set that variable for a scored model
run: an agent can read its own environment, and the verifier treats a marker
matching the flag as proof of an oracle rollout.

Two other things to size up front, both of which silently produce zero graded
trials:

- **The agent clock.** `--agent-timeout-multiplier` is a fraction of the task's
  own budget. FrontierSWE tasks declare 20 hours, so `0.02` is 24 minutes, and
  an agent cut off mid-task cannot be scored at all. Keep the clock above what
  `max_turns` needs and let turns be the binding control.
- **The turn budget, and whether the score can discriminate at all.** A reward
  that is 0 for every host is real but useless for ranking them, and a turn
  budget too small for the task guarantees exactly that. Measured on
  2026-09-03: a terminus-2 agent spends `max_turns=40` in about 9 minutes, and
  every host then scored 0 on all three tasks, with the Verilog task reporting
  `build: 0.0` and the libexpat optimization task `unit_pass_rate: 0.0` even
  though it starts from working code. Before reading anything into a quality
  column, check that the scores vary; if they do not, the budget or the
  benchmark is wrong for the question, not the hosts.
- **Resources.** 11 of FrontierSWE v2's 34 tasks require a GPU and two ask for
  128 GB of RAM. `--all-cpu-tasks` selects the ones a normal VM can hold.

## Reading the results

Harbor writes a job directory per arm with a `result.json` per trial. The
ledger directory gets `<host>.jsonl`. Then:

```bash
compound-bench ledger artifacts/tb4/ledger/deepinfra-fp8.jsonl --hosts
python3 scripts/analyze_arms.py artifacts/tb4          # arms side by side
```

See [Reports and ledgers](../reports/) for what each column means.
