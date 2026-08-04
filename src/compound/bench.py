"""One front door for the benchmark library: list, inspect, and run tasks.

Every benchmark Compound ships lives behind the same three commands, so a task
subset on any of them is one CLI call away:

    python -m compound.bench list
    python -m compound.bench tasks tau2 [--partition optimizer_train] [--contains retail]
    python -m compound.bench run tau2 --model zai-org/GLM-5.2-FP8 --tasks retail:10,airline:3 --go
    python -m compound.bench run bfcl --model kimi-k3 --tasks simple_20,simple_21 --go

Money safety: ``run`` without ``--go`` never spends — it prints what would
execute and exits. BFCL and DS-1000 additionally enforce the paid-run budget and
per-run cap from compound.yaml; tau runs bill whatever the episodes cost, so
size the subset first.

Any provider, any model (tau2): the agent can run on OpenRouter (optionally
pinned to one upstream via --upstream), on Doubleword, or on ANY
OpenAI-compatible host you point at with --api-base/--api-key-env (vLLM,
Fireworks direct, Groq, a local server, ...). BFCL and DS-1000 draw models and
providers from compound.yaml, where any ``openai_compatible`` provider block
works the same way.

Adding a benchmark is one ``Benchmark`` entry: a manifest of partitioned case
ids plus a runner that accepts (models, case_ids). See issue #33.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Benchmark:
    name: str
    manifest: Path
    description: str
    runnable: str  # one-line note on what `run` executes for this benchmark


BENCHMARKS: dict[str, Benchmark] = {
    "tau2": Benchmark(
        name="tau2",
        manifest=Path("benchmarks/manifests/tau_bench.json"),
        description=(
            "tau2-bench: interactive tool-calling customer support "
            "(airline/retail/telecom) with a live user simulator and official reward"
        ),
        runnable="full interactive protocol per episode; any provider via --api-base",
    ),
    "bfcl": Benchmark(
        name="bfcl",
        manifest=Path("benchmarks/manifests/bfcl.json"),
        description=(
            "Berkeley Function-Calling Leaderboard: single-turn function-call "
            "generation graded by the official AST checker"
        ),
        runnable="single-turn cases; models/providers come from compound.yaml",
    ),
    "ds1000": Benchmark(
        name="ds1000",
        manifest=Path("benchmarks/manifests/ds1000_baseline25.json"),
        description=(
            "DS-1000: data-science code generation graded by executing the "
            "official tests in a pinned container"
        ),
        runnable="needs the evaluator docker image; models come from compound.yaml",
    ),
    "mmlu": Benchmark(
        name="mmlu",
        manifest=Path("benchmarks/manifests/mmlu.json"),
        description=(
            "MMLU: multiple-choice knowledge across 57 subjects, graded by "
            "exact letter match (self-contained manifest, no judge)"
        ),
        runnable="one completion per case; any provider via --api-base",
    ),
    "terminal_bench": Benchmark(
        name="terminal_bench",
        manifest=Path("benchmarks/manifests/terminal_bench.json"),
        description=(
            "Terminal-Bench core: agentic terminal tasks executed by the "
            "official harness in Docker with the official tests"
        ),
        runnable="delegates to terminal-bench via uvx; --model is a litellm name",
    ),
}


def _load_cases(bench: Benchmark) -> list[dict]:
    if not bench.manifest.exists():
        raise SystemExit(
            f"error: manifest {bench.manifest} not found — "
            f"run `compound prepare-manifests` first"
        )
    return json.loads(bench.manifest.read_text())["cases"]


def cmd_list() -> int:
    for bench in BENCHMARKS.values():
        try:
            cases = _load_cases(bench)
        except SystemExit:
            print(f"{bench.name:8s} (manifest missing — run `compound prepare-manifests`)")
            continue
        parts = Counter(c["partition"] for c in cases)
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(parts.items()))
        print(f"{bench.name:8s} {len(cases):4d} cases ({breakdown})")
        print(f"{'':8s} {bench.description}")
        print(f"{'':8s} run: {bench.runnable}")
    return 0


def select_case_ids(
    cases: list[dict],
    *,
    partition: str | None = None,
    contains: str | None = None,
    explicit: list[str] | None = None,
) -> list[str]:
    """Resolve a task subset; explicit ids are validated against the manifest."""
    ids = [c["case_id"] for c in cases]
    if explicit is not None:
        known = set(ids)
        missing = [t for t in explicit if t not in known]
        if missing:
            raise SystemExit(f"error: unknown case ids: {', '.join(missing)}")
        return explicit
    if partition:
        ids = [c["case_id"] for c in cases if c["partition"] == partition]
    if contains:
        ids = [i for i in ids if contains.lower() in i.lower()]
    return ids


def cmd_tasks(name: str, partition: str | None, contains: str | None) -> int:
    bench = BENCHMARKS[name]
    ids = select_case_ids(_load_cases(bench), partition=partition, contains=contains)
    for case_id in ids:
        print(case_id)
    print(f"# {len(ids)} case(s)", file=sys.stderr)
    return 0


def _run_tau(args: argparse.Namespace, case_ids: list[str]) -> int:
    from compound.adapters.tau import TauModel, run_tau_partition

    agent = TauModel(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        openrouter_provider=args.upstream,
        service_tier=args.tier,
        max_tokens=args.max_tokens,
    )
    agent.litellm_name()  # fail fast on inconsistent provider flags
    agent.resolve_api_key_env()
    user = TauModel(provider="openrouter", model=args.user_model)
    print(f"agent: {agent.litellm_name()}  ({agent.slug()})")
    print(f"user simulator: {user.litellm_name()}")
    print(f"cases: {len(case_ids)} x {args.trials} trial(s), max_steps={args.max_steps}")
    if not args.go:
        print("\ndry run (no spend). Add --go to execute; episodes bill at provider rates.")
        return 0
    output = Path(args.output or "artifacts/bench/tau2")
    run_tau_partition(
        manifest_path=BENCHMARKS["tau2"].manifest,
        partition=None,
        agent_model=agent,
        user_model=user,
        candidate_instruction="",
        trials=args.trials,
        max_steps=args.max_steps,
        output_dir=output / "episodes",
        telemetry_path=(output / "telemetry.jsonl").resolve(),
        case_ids=set(case_ids),
    )
    print(f"episodes -> {output}/episodes")
    return 0


def _run_delegated(name: str, args: argparse.Namespace, case_ids: list[str]) -> int:
    if not args.go:
        print(f"{name}: {len(case_ids)} case(s) for model {args.model}")
        print(f"cap: ${args.cap:.2f} (budget rules from compound.yaml apply)")
        print("\ndry run (no spend). Add --go to execute.")
        return 0
    if name == "bfcl":
        from compound.bfcl_matrix import run_bfcl_baseline_matrix

        path = run_bfcl_baseline_matrix(
            models=[args.model],
            case_ids=case_ids,
            trials_per_case=args.trials,
            experiment_cap_usd=args.cap,
        )
    else:
        from compound.baseline_matrix import run_ds1000_baseline_matrix

        path = run_ds1000_baseline_matrix(
            manifest_path=BENCHMARKS["ds1000"].manifest,
            models=[args.model],
            case_ids=case_ids,
            trials_per_case=args.trials,
            experiment_cap_usd=args.cap,
        )
    print(path)
    return 0


def _endpoint_for(args: argparse.Namespace) -> tuple[str, str]:
    """(base_url, api_key_env) for benchmarks that call the chat API directly."""
    if args.api_base:
        if not args.api_key_env:
            raise SystemExit("error: --api-base needs --api-key-env")
        return args.api_base, args.api_key_env
    known = {
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "doubleword": ("https://api.doubleword.ai/v1", "DOUBLEWORD_API_KEY"),
    }
    if args.provider not in known:
        raise SystemExit(
            f"error: provider {args.provider!r} needs --api-base/--api-key-env"
        )
    return known[args.provider]


def _run_mmlu(args: argparse.Namespace, case_ids: list[str]) -> int:
    from compound.adapters.mmlu import run_mmlu

    base_url, api_key_env = _endpoint_for(args)
    cases = [
        c for c in _load_cases(BENCHMARKS["mmlu"]) if c["case_id"] in set(case_ids)
    ]
    print(f"mmlu: {len(cases)} case(s), model {args.model} via {base_url}")
    if not args.go:
        print("\ndry run (no spend). Add --go to execute; one short completion per case.")
        return 0
    run_mmlu(
        cases,
        model=args.model,
        base_url=base_url,
        api_key_env=api_key_env,
        upstream=args.upstream,
        output_path=Path(args.output or "artifacts/bench/mmlu") / "results.json",
    )
    return 0


def _run_terminal_bench(args: argparse.Namespace, case_ids: list[str]) -> int:
    from compound.adapters.terminal_bench import DEFAULT_AGENT, run_terminal_bench

    print(f"terminal_bench: {len(case_ids)} task(s), model {args.model}")
    print("harness: official terminal-bench via uvx (Docker required)")
    if not args.go:
        print("\ndry run (no spend). Add --go to execute; agentic episodes bill per step.")
        return 0
    return run_terminal_bench(
        case_ids,
        model=args.model,
        agent=args.tb_agent or DEFAULT_AGENT,
        output_dir=Path(args.output or "artifacts/bench/terminal-bench"),
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    if args.benchmark == "mmlu":
        from compound.adapters.mmlu import build_manifest

        path = build_manifest(
            BENCHMARKS["mmlu"].manifest, per_subject=args.per_subject
        )
    elif args.benchmark == "terminal_bench":
        from compound.adapters.terminal_bench import build_manifest

        path = build_manifest(BENCHMARKS["terminal_bench"].manifest)
    else:
        raise SystemExit(
            f"error: {args.benchmark} manifests come from `compound prepare-manifests`"
        )
    cases = json.loads(path.read_text())["cases"]
    print(f"{len(cases)} cases -> {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bench = BENCHMARKS[args.benchmark]
    explicit = args.tasks.split(",") if args.tasks else None
    case_ids = select_case_ids(
        _load_cases(bench),
        partition=args.partition,
        contains=args.contains,
        explicit=explicit,
    )
    if not case_ids:
        raise SystemExit("error: selection matches no cases")
    if args.benchmark == "tau2":
        return _run_tau(args, case_ids)
    if args.benchmark == "mmlu":
        return _run_mmlu(args, case_ids)
    if args.benchmark == "terminal_bench":
        return _run_terminal_bench(args, case_ids)
    return _run_delegated(args.benchmark, args, case_ids)


def main() -> int:
    parser = argparse.ArgumentParser(prog="compound.bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="benchmarks, case counts, and how each one runs")

    prepare = sub.add_parser("prepare", help="build a benchmark's partitioned manifest")
    prepare.add_argument("benchmark", choices=("mmlu", "terminal_bench"))
    prepare.add_argument("--per-subject", type=int, default=5, help="mmlu: test questions per subject")

    tasks = sub.add_parser("tasks", help="print case ids for a benchmark")
    tasks.add_argument("benchmark", choices=sorted(BENCHMARKS))
    tasks.add_argument("--partition", help="filter to one partition")
    tasks.add_argument("--contains", help="case-insensitive substring filter")

    run = sub.add_parser("run", help="run a task subset (dry run unless --go)")
    run.add_argument("benchmark", choices=sorted(BENCHMARKS))
    run.add_argument("--model", required=True, help="model id as the provider knows it")
    run.add_argument("--tasks", help="comma-separated case ids (see `tasks`)")
    run.add_argument("--partition", help="or: every case in one partition")
    run.add_argument("--contains", help="or: every case id matching a substring")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--go", action="store_true", help="actually spend; default is a dry run")
    # tau2 routing (any provider, any model)
    run.add_argument("--provider", default="openrouter", help="tau2: openrouter, doubleword, or a label for --api-base")
    run.add_argument("--api-base", help="tau2: custom OpenAI-compatible endpoint")
    run.add_argument("--api-key-env", help="tau2: env var holding the key for --api-base")
    run.add_argument("--upstream", help="tau2: pin one OpenRouter upstream (fallbacks disabled)")
    run.add_argument("--tier", help="tau2: service tier flag (e.g. doubleword flex)")
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--user-model", default="openai/gpt-5.6-luna", help="tau2: user simulator (OpenRouter)")
    run.add_argument("--output", help="tau2: episode output dir")
    # bfcl / ds1000 delegation
    run.add_argument("--cap", type=float, default=4.0, help="bfcl/ds1000: per-run USD cap")
    # terminal_bench delegation
    run.add_argument("--tb-agent", help="terminal_bench: harness agent (default terminus)")

    args = parser.parse_args()
    if args.command == "list":
        return cmd_list()
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "tasks":
        return cmd_tasks(args.benchmark, args.partition, args.contains)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
