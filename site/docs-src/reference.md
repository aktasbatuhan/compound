title: CLI reference
order: 90

# CLI reference

Generated from `--help` at build time, so it matches the checked-in CLI. `compound-bench` is installed by `uv sync --extra dev`; `python -m compound.bench` is the same entry point.

## compound-bench list

```text
usage: compound.bench list [-h]

options:
  -h, --help  show this help message and exit
```

## compound-bench prepare

```text
usage: compound.bench prepare [-h] [--per-subject PER_SUBJECT]
                              {tau2,mmlu,terminal_bench}

positional arguments:
  {tau2,mmlu,terminal_bench}

options:
  -h, --help            show this help message and exit
  --per-subject PER_SUBJECT
                        mmlu: test questions per subject
```

## compound-bench providers

```text
usage: compound.bench providers [-h] [--json] [--probe] model

positional arguments:
  model       OpenRouter model slug, e.g. deepseek/deepseek-v4-flash-0731

options:
  -h, --help  show this help message and exit
  --json      machine-readable output
  --probe     send one tiny pinned call to every host and report what it
              actually did. OpenRouter's 'up' is its own belief; without your
              own upstream key you sit on its shared rate-limit pool, where a
              listed-up host can 429 every call.
```

## compound-bench tasks

```text
usage: compound.bench tasks [-h] [--partition PARTITION] [--contains CONTAINS]
                            {bfcl,ds1000,mmlu,tau2,terminal_bench}

positional arguments:
  {bfcl,ds1000,mmlu,tau2,terminal_bench}

options:
  -h, --help            show this help message and exit
  --partition PARTITION
                        filter to one partition
  --contains CONTAINS   case-insensitive substring filter
```

## compound-bench run

```text
usage: compound.bench run [-h] --model MODEL [--tasks TASKS]
                          [--partition PARTITION] [--manifest MANIFEST]
                          [--contains CONTAINS] [--trials TRIALS] [--go]
                          [--providers PROVIDERS] [--provider PROVIDER]
                          [--api-base API_BASE] [--api-key-env API_KEY_ENV]
                          [--upstream UPSTREAM] [--tier TIER]
                          [--max-steps MAX_STEPS] [--max-tokens MAX_TOKENS]
                          [--user-model USER_MODEL] [--output OUTPUT]
                          [--cap CAP] [--tb-agent TB_AGENT]
                          [--tb-concurrent TB_CONCURRENT]
                          [--reasoning {on,off,default}]
                          [--cache-optin | --no-cache-optin]
                          [--call-ledger PATH]
                          [--tb-timeout-mult TB_TIMEOUT_MULT]
                          {bfcl,ds1000,mmlu,tau2,terminal_bench}

positional arguments:
  {bfcl,ds1000,mmlu,tau2,terminal_bench}

options:
  -h, --help            show this help message and exit
  --model MODEL         model id as the provider knows it
  --tasks TASKS         comma-separated case ids (see `tasks`)
  --partition PARTITION
                        or: every case in one partition
  --manifest MANIFEST   override the benchmark's task manifest (more tasks)
  --contains CONTAINS   or: every case id matching a substring
  --trials TRIALS
  --go                  actually spend; default is a dry run
  --providers PROVIDERS
                        comma-separated provider tokens to sweep, e.g. openrou
                        ter/deepinfra,openrouter/baseten,doubleword/flex
  --provider PROVIDER   tau2: openrouter, doubleword, or a label for --api-
                        base
  --api-base API_BASE   tau2: custom OpenAI-compatible endpoint
  --api-key-env API_KEY_ENV
                        tau2: env var holding the key for --api-base
  --upstream UPSTREAM   tau2: pin one OpenRouter upstream (fallbacks disabled)
  --tier TIER           tau2: service tier flag (e.g. doubleword flex)
  --max-steps MAX_STEPS
  --max-tokens MAX_TOKENS
  --user-model USER_MODEL
                        tau2: user simulator (OpenRouter)
  --output OUTPUT       tau2: episode output dir
  --cap CAP             bfcl/ds1000: per-run USD cap
  --tb-agent TB_AGENT   terminal_bench: harness agent (default terminus)
  --tb-concurrent TB_CONCURRENT
                        terminal_bench: tasks per host
  --reasoning {on,off,default}
                        terminal_bench: pin the model's reasoning mode via the
                        proxy (on/off), or 'default' to inject nothing. Given,
                        the flag wins over a pre-set COMPOUND_REASONING;
                        omitted, that env var is honored.
  --cache-optin, --no-cache-optin
                        terminal_bench: inject explicit prompt-cache markers
                        for explicit_marker providers (e.g. doubleword). ON by
                        default, because a marker-gated host otherwise re-
                        bills the whole transcript every turn; pass --no-
                        cache-optin to measure that unmarked path on purpose.
                        COMPOUND_DW_CACHE overrides when neither flag is
                        given.
  --call-ledger PATH    record one JSONL row per model call (route, provider
                        echo, tokens, cached tokens, cost, status, latency).
                        The per-call record is what supports cache-hit and
                        routing claims; episode results cannot.
  --tb-timeout-mult TB_TIMEOUT_MULT
                        terminal_bench: multiply every task's
                        max_agent_timeout_sec by N (extended-limits mode;
                        results are labeled non-official). A pre-set
                        COMPOUND_TB_TIMEOUT_MULT wins over this flag.
```

## compound-bench harbor

```text
usage: compound.bench harbor [-h] --providers PROVIDERS --model MODEL
                             [--task-path TASK_PATH] [--host-model HOST=MODEL]
                             [--dataset DATASET] [--agent AGENT]
                             [--tasks TASKS] [--n-tasks N_TASKS]
                             [--attempts ATTEMPTS]
                             [--n-concurrent N_CONCURRENT]
                             [--timeout-multiplier TIMEOUT_MULTIPLIER]
                             [--agent-timeout-multiplier AGENT_TIMEOUT_MULTIPLIER]
                             [--ak KEY=VALUE] [--env ENV]
                             [--jobs-dir JOBS_DIR] [--ledger-dir LEDGER_DIR]
                             [--reasoning {on,off,default}]
                             [--cache-optin | --no-cache-optin] [--go]

options:
  -h, --help            show this help message and exit
  --providers PROVIDERS
                        comma-separated provider tokens, e.g.
                        openrouter/auto,openrouter/deepinfra
  --model MODEL         model id as the upstream knows it
  --task-path TASK_PATH
                        run a Harbor task or dataset DIRECTORY on disk instead
                        of a hub dataset (harbor --path). Lets a benchmark
                        whose own runner is unreleased still run, as long as
                        its tasks carry a Harbor task.toml.
  --host-model HOST=MODEL
                        model id to send to one host when it names the weights
                        differently, repeatable; HOST is a provider token,
                        label, or kind (e.g. doubleword=zai-org/GLM-5.3-Flash)
  --dataset DATASET     Harbor dataset name@version (pinned, not @latest, so
                        the task set cannot shift between arms of one
                        experiment)
  --agent AGENT         Harbor agent. Must be a terminus-family agent when
                        pinning: an in-sandbox agent cannot reach a localhost
                        proxy.
  --tasks TASKS         comma-separated task names (glob patterns allowed)
  --n-tasks N_TASKS     cap tasks after filtering
  --attempts ATTEMPTS, -k ATTEMPTS
                        attempts per task
  --n-concurrent N_CONCURRENT
                        concurrent trials
  --timeout-multiplier TIMEOUT_MULTIPLIER
                        scale EVERY phase's time limit, environment build
                        included (Harbor-native; runs are non-official when
                        set)
  --agent-timeout-multiplier AGENT_TIMEOUT_MULTIPLIER
                        scale only how long the agent may work, leaving
                        environment build and verification alone. This is the
                        flag for bounding a run: TB4 tasks allow the agent 8
                        hours by default.
  --ak KEY=VALUE, --agent-kwarg KEY=VALUE
                        agent constructor kwarg, repeatable. Use max_turns=N
                        to give every host the same work: an equal wall clock
                        hands a faster host more turns.
  --env ENV             Harbor environment backend
  --jobs-dir JOBS_DIR   where jobs land
  --ledger-dir LEDGER_DIR
                        per-host call ledger directory
  --reasoning {on,off,default}
                        pin the model's reasoning mode via the proxy
  --cache-optin, --no-cache-optin
                        inject explicit prompt-cache markers for
                        explicit_marker providers (e.g. doubleword). ON by
                        default; --no-cache-optin measures the unmarked path.
  --go                  execute (default is a dry run)
```

## compound-bench serving

```text
usage: compound.bench serving [-h] --providers PROVIDERS --shapes SHAPES
                              [--model-or MODEL_OR] [--model MODEL]
                              [--host-model HOST=MODEL] [--rounds ROUNDS]
                              [--interval INTERVAL] [--reps REPS]
                              [--cache-mode {cold,warm,both}]
                              [--reasoning-modes {on,off,both}]
                              [--temperature TEMPERATURE] [--out OUT]

options:
  -h, --help            show this help message and exit
  --providers PROVIDERS
                        comma-separated provider tokens, e.g.
                        openrouter/deepinfra,doubleword/flex,openrouter/auto
  --shapes SHAPES       JSON file mapping name -> {messages, response_format}
  --model-or MODEL_OR   model slug for OpenRouter routes
  --model MODEL         model slug for Doubleword/direct routes
  --host-model HOST=MODEL
                        model id one host should receive instead of --model /
                        --model-or, keyed by token, label or kind
                        (openai=gpt-5.4-mini, anthropic=claude-sonnet-5).
                        Repeatable; a first-party grid needs one per host
                        since no single slug names the same weights
                        everywhere.
  --rounds ROUNDS       scheduled rounds (time-of-day variance)
  --interval INTERVAL   seconds between rounds
  --reps REPS           repetitions per (route, mode, shape) cell
  --cache-mode {cold,warm,both}
                        cold prepends a per-call nonce so no prefix is ever
                        served warm, isolating raw serving speed; warm sends a
                        byte-identical prompt every rep so the host's prompt
                        cache can hit. The cold/warm delta is the cache
                        measurement, and warm cells run serially so rep 0 can
                        populate the cache the rest read.
  --reasoning-modes {on,off,both}
                        which reasoning pinning to sweep; 'off' matches a
                        vendor latency benchmark that disables reasoning
  --temperature TEMPERATURE
                        sampling temperature. Use 0 to compare hosts token for
                        token: at temperature 0 a divergence between two hosts
                        serving the same weights is a difference in numerics,
                        not in sampling.
  --out OUT             output dir for results.jsonl
```

## compound-bench ledger

```text
usage: compound.bench ledger [-h] [--hosts] path

positional arguments:
  path        path to a calls.jsonl written by --call-ledger

options:
  -h, --help  show this help message and exit
  --hosts     also list which upstreams answered each route, with counts
```

## compound (trace pipeline, TypeScript)

```text
compound — turn production traces into gated optimization evidence

Usage:
  compound init [--config PATH] [--db PATH] [--force]
  compound validate [--config PATH]
  compound providers [name]                        (known providers; a name prints a paste-ready block)
  compound import <file> [--importer langfuse|json|otel] [--db PATH] [--config PATH] [--project-id ID] [--unsafe-no-redaction]
  compound curate <task_key> [--split train:val:cal:dec] [--db PATH]
  compound suggest-assertions <task_key> [--db PATH] [--config PATH]
  compound experiment <task_key> <model> [--partition P] [--paid --cap USD]
  compound gate <task_key> --candidate M --reference M --reason "..." [--margin 0.05] [--monthly-volume N] [--paid --cap USD]
  compound eval <task_key> --candidate M --reference M [--reason "..."]   (CI gate: exit 0 meets / 1 regresses / 2 undecidable)
  compound judge calibrate <task_key> [--paid --cap USD]
  compound judge grade <task_key> <experiment_id> [--paid --cap USD]
  compound optimize <task_key> --candidate M [--reflection M] [--max-calls N] [--force]
  compound telemetry [task_key] [--json] [--db PATH]
  compound view [gate|case|trace|experiment] [id] [--full] [--db PATH]   (read-only browser; overview if no args)
  compound view compare [task_key] [--priority quality=0.5,cost=0.3,latency=0.2] [--monthly-volume N] [--db PATH]
                                                                         (cost vs score per model; --priority adds a weighted
                                                                          ranking + Pareto frontier; axes: quality, cost,
                                                                          latency, throughput; per-task default via
                                                                          compound.yaml task_keys.<task>.priority)
  compound status [--db PATH]
  compound serve [--port N] [--host HOST] [--db PATH] [--config PATH]
  compound help

Importers: langfuse, json, otel
```
