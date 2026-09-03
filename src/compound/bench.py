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
        runnable="full interactive protocol per episode; any provider via --api-base"
        " (needs `prepare tau2` once)",
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


def _prepare_hint(name: str) -> str:
    # mmlu / terminal_bench manifests are built locally; the rest ship with the repo
    # but can be rebuilt from source datasets.
    if name in ("mmlu", "terminal_bench"):
        return f"compound-bench prepare {name}"
    return "compound prepare-manifests"


def _load_cases(bench: Benchmark) -> list[dict]:
    if not bench.manifest.exists():
        raise SystemExit(
            f"error: manifest {bench.manifest} not found — "
            f"run `{_prepare_hint(bench.name)}` first"
        )
    return json.loads(bench.manifest.read_text())["cases"]


def cmd_list() -> int:
    for bench in BENCHMARKS.values():
        try:
            cases = _load_cases(bench)
        except SystemExit:
            print(f"{bench.name:8s} (manifest missing — run `{_prepare_hint(bench.name)}`)")
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


def cmd_providers(model: str, as_json: bool, probe: bool = False) -> int:
    """List the OpenRouter upstreams that serve a model, as --providers tokens."""
    import os
    from dataclasses import asdict

    from compound.openrouter_discovery import fetch_endpoints, format_table, probe_endpoints

    try:
        endpoints = fetch_endpoints(model)
    except Exception as exc:  # network/HTTP/JSON — surface, do not traceback
        raise SystemExit(
            f"error: could not fetch endpoints for {model!r}: {exc}"
        ) from exc

    probed: dict[str, tuple[int | str, float, str]] = {}
    if probe:
        _require_keys({"OPENROUTER_API_KEY"})
        probed = probe_endpoints(endpoints, model, os.environ["OPENROUTER_API_KEY"])

    if as_json:
        rows = []
        for endpoint in endpoints:
            row = asdict(endpoint)
            if endpoint.tag in probed:
                status, seconds, detail = probed[endpoint.tag]
                row["probe"] = {
                    "status": status,
                    "seconds": round(seconds, 2),
                    "answered": status == 200,
                    "detail": detail,
                }
            rows.append(row)
        print(json.dumps(rows, indent=2))
        return 0

    print(format_table(endpoints))
    if probe:
        # OpenRouter's own "up" is its belief about the endpoint; this column is
        # one real pinned call. They disagree often enough to matter: a host can
        # be listed up and 429 every call for one model while serving another.
        print("\nPROBE: one pinned call per host, just now")
        print(f"{'PROVIDER TOKEN':<30s} {'STATUS':>7s} {'SECONDS':>8s}  DETAIL")
        answered = 0
        for endpoint in endpoints:
            if endpoint.tag not in probed:
                continue
            status, seconds, detail = probed[endpoint.tag]
            answered += status == 200
            print(f"{endpoint.token:<30s} {str(status):>7s} {seconds:>8.1f}  {detail[:60]}")
        print(f"\n{answered} of {len(probed)} answered. Only these can carry an arm right now.")
    return 0


def _require_keys(env_vars: set[str]) -> None:
    """Fail before any spend if a required credential is missing from the env."""
    import os

    missing = sorted(v for v in env_vars if not os.getenv(v))
    if missing:
        raise SystemExit(
            "error: missing required API key(s): "
            + ", ".join(missing)
            + " — set them in .env or the environment before a --go run"
        )


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
    # User simulator and the nl-assertions judge always route via OpenRouter; the
    # agent adds its own key when it is not an OpenRouter host.
    needed = {"OPENROUTER_API_KEY"}
    agent_key = agent.resolve_api_key_env()
    if agent_key:
        needed.add(agent_key)
    _require_keys(needed)
    # Only enforce the engine on real execution, so dry runs preview without it.
    from compound.adapters.tau_setup import ensure_tau2

    ensure_tau2()  # clear pointer to `prepare tau2` instead of a bare ImportError
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
    _require_keys({api_key_env})
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
    print(_tb_pin_line())
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
    if args.benchmark == "tau2":
        # tau2 needs its engine installed, not a manifest built. The partitioned
        # task manifest already ships in the repo; this clones + installs tau2.
        from compound.adapters.tau_setup import prepare_tau2

        path = prepare_tau2()
        print(f"tau2-bench ready at {path}. Now: python -m compound.bench run tau2 ...")
        return 0
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


def _load_providers_config() -> dict | None:
    """The ``providers`` block of compound.yaml, for direct/<name> tokens."""
    path = Path("compound.yaml")
    if not path.exists():
        return None
    import yaml

    return (yaml.safe_load(path.read_text()) or {}).get("providers")


def _run_sweep(args: argparse.Namespace, case_ids: list[str]) -> int:
    """Run one benchmark across a provider-token list (same model, many hosts)."""
    from compound import provider_sweep
    from compound.providers_registry import parse_providers

    specs = parse_providers(args.providers, providers_config=_load_providers_config())
    print(
        f"{args.benchmark} sweep: {len(specs)} host(s) x {len(case_ids)} task(s), "
        f"model {args.model}, {args.trials} trial(s)"
    )
    for line in provider_sweep.plan(specs, args.model):
        print(line)
    if args.benchmark == "terminal_bench":
        print(_tb_pin_line())
    if not args.go:
        print("\ndry run (no spend). Add --go to execute; each host bills at its own rate.")
        return 0
    needed = {s.required_key_env() for s in specs}
    if args.benchmark == "tau2":
        needed.add("OPENROUTER_API_KEY")  # user simulator + nl-assertions judge
    _require_keys(needed)
    output = Path(args.output or f"artifacts/bench/{args.benchmark}-sweep")
    if args.benchmark == "tau2":
        from compound.adapters.tau_setup import ensure_tau2

        ensure_tau2()
        provider_sweep.sweep_tau2(
            specs,
            model=args.model,
            case_ids=case_ids,
            manifest_path=Path(args.manifest) if args.manifest else BENCHMARKS["tau2"].manifest,
            trials=args.trials,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            user_model_name=args.user_model,
            output_dir=output,
        )
    elif args.benchmark == "terminal_bench":
        from compound.adapters.terminal_bench import DEFAULT_AGENT

        provider_sweep.sweep_terminal_bench(
            specs,
            model=args.model,
            case_ids=case_ids,
            output_dir=output,
            agent=args.tb_agent or DEFAULT_AGENT,
            n_concurrent=args.tb_concurrent,
        )
    else:
        raise SystemExit(
            f"error: --providers supports tau2 and terminal_bench, not {args.benchmark}"
        )
    print(f"sweep output -> {output}")
    _auto_report(args.benchmark, output)
    return 0


def _auto_report(benchmark: str, output: Path) -> None:
    """Build the report (tables + charts incl. provider radars) after a sweep.

    Best-effort by design: a finished paid run must never be marked failed
    because chart generation hiccuped. Cost axes appear when the run recorded
    provider-reported cost; add ``--prices`` via the report CLI to fill gaps.
    """
    try:
        if benchmark == "tau2":
            from compound.bench_report import build_report

            build_report(output, {})
        elif benchmark == "terminal_bench":
            from compound.tb_report import build_report

            build_report(output, {})
        else:
            return
        print(f"report -> {output}/report/charts.html (profiles, context, cost)")
    except Exception as exc:  # noqa: BLE001
        print(f"report generation skipped ({exc}); run the report module manually")


def _manifest_cases(args: argparse.Namespace, bench: Benchmark) -> list[dict]:
    """Cases from a --manifest override if given, else the benchmark's own."""
    if args.manifest:
        path = Path(args.manifest)
        if not path.exists():
            raise SystemExit(f"error: --manifest {path} not found")
        return json.loads(path.read_text())["cases"]
    return _load_cases(bench)


#: Mirrored from :mod:`compound.adapters.harbor` so building the parser does not
#: import the adapter (and its uvx assumptions) on every CLI invocation.
_HARBOR_DEFAULT_DATASET = "terminal-bench/terminal-bench@4.0.0"
_HARBOR_DEFAULT_AGENT = "terminus-2"


def cmd_serving(args: argparse.Namespace) -> int:
    """Serving-metrics harness: TTFT / decode TPS / cost per host per reasoning mode."""
    from compound import serving_metrics as sm
    from compound.providers_registry import parse_providers

    specs = parse_providers(args.providers, providers_config=_load_providers_config())
    shapes = sm.load_shapes(Path(args.shapes))
    try:
        for spec in specs:
            sm.model_for(spec, args.model_or, args.model)  # fail fast on a missing model
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    _require_keys({s.required_key_env() for s in specs})
    cells = len(specs) * len(shapes) * len(sm.MODES) * args.reps * args.rounds
    print(
        f"serving: {len(specs)} route(s) x {len(shapes)} shape(s) x {len(sm.MODES)} mode(s) "
        f"x {args.reps} rep(s) x {args.rounds} round(s) = {cells} calls"
    )
    out = sm.run_serving(
        specs,
        args.model_or,
        args.model,
        shapes,
        out_dir=Path(args.out),
        rounds=args.rounds,
        interval=args.interval,
        reps=args.reps,
    )
    print(f"results -> {out}")
    return 0


def _parse_agent_kwargs(pairs: list[str] | None) -> dict[str, str]:
    """``KEY=VALUE`` strings into a dict, failing loudly on a malformed pair."""
    out: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"error: --ak expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def cmd_harbor(args: argparse.Namespace) -> int:
    """Terminal-Bench 4.0 (or any Harbor dataset) across hosts, each pinned."""
    from compound import provider_sweep
    from compound.adapters import harbor
    from compound.providers_registry import apply_host_models, parse_providers

    specs = parse_providers(args.providers, providers_config=_load_providers_config())
    host_models: dict[str, str] = {}
    for item in args.host_model:
        if "=" not in item:
            raise SystemExit(f"--host-model expects HOST=MODEL, got {item!r}")
        key, value = item.split("=", 1)
        host_models[key.strip()] = value.strip()
    specs = apply_host_models(specs, host_models)
    tasks = args.tasks.split(",") if args.tasks else None
    agent_kwargs = _parse_agent_kwargs(args.agent_kwargs)
    # An on-disk task tree and a hub dataset are alternative sources; naming a
    # path means the pinned-version default no longer describes the run.
    dataset = None if args.task_path else args.dataset
    source = f"path {args.task_path}" if args.task_path else f"dataset {args.dataset}"
    print(f"harbor: {source}, agent {args.agent}, model {args.model}")
    for line in provider_sweep.plan(specs, args.model):
        print(line)
    # Report the pinning this run will apply, not the ambient env: the flags are
    # threaded into the proxy per host, and the ledger is per host under
    # --ledger-dir rather than the single COMPOUND_CALL_LEDGER path.
    _apply_tb_env(args)
    print(_tb_pin_line().splitlines()[0])
    print(f"call ledger: {args.ledger_dir or 'off'}")
    if tasks:
        print(f"tasks: {', '.join(tasks)}")
    if args.n_tasks:
        print(f"task cap: {args.n_tasks}")
    print(
        f"grid: {len(specs)} host(s) x {args.n_tasks or len(tasks or []) or 'all'} task(s) "
        f"x {args.attempts} attempt(s)"
    )
    if not args.go:
        # Show the exact argv rather than describing it: a dry run should let a
        # reviewer check the command that money would be spent on.
        example = harbor.build_command(
            dataset=dataset,
            task_path=args.task_path,
            model=f"openai/{args.model}",
            agent=args.agent,
            jobs_dir=args.jobs_dir,
            job_name="<host>-<ts>",
            include_tasks=tasks,
            n_tasks=args.n_tasks,
            attempts=args.attempts,
            n_concurrent=args.n_concurrent,
            timeout_multiplier=args.timeout_multiplier,
            agent_timeout_multiplier=args.agent_timeout_multiplier,
            agent_kwargs=agent_kwargs,
            env_type=args.env,
            proxied=True,
        )
        print("\ncommand per host:\n  " + " ".join(example))
        print("\ndry run (no spend). Add --go to execute.")
        return 0
    summaries = provider_sweep.sweep_harbor(
        specs,
        model=args.model,
        jobs_dir=Path(args.jobs_dir),
        dataset=dataset,
        task_path=args.task_path,
        agent=args.agent,
        include_tasks=tasks,
        n_tasks=args.n_tasks,
        attempts=args.attempts,
        n_concurrent=args.n_concurrent,
        timeout_multiplier=args.timeout_multiplier,
        agent_timeout_multiplier=args.agent_timeout_multiplier,
        agent_kwargs=agent_kwargs,
        env_type=args.env,
        ledger_dir=Path(args.ledger_dir) if args.ledger_dir else None,
    )
    if not summaries:
        print("no host produced a job result")
        return 1
    print(f"\n{'host':<24s} {'trials':>7s} {'verdicts':>9s} {'resolved':>9s} "
          f"{'rate':>7s} {'unverified':>11s} {'errored':>8s}")
    for label, summary in summaries.items():
        rate = summary["resolve_rate"]
        print(
            f"{label:<24s} {summary['trials']:>7d} {summary['verdicts']:>9d} "
            f"{summary['resolved']:>9d} "
            f"{('—' if rate is None else f'{rate * 100:.1f}'):>7s} "
            f"{summary['unverified']:>11d} {summary.get('errored_trials', 0):>8d}"
        )
    print(
        "\nrate is over trials that returned a verdict; unverified trials "
        "(harness or infrastructure failures) are counted apart, never as model failures."
    )
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    """Per-route read of a call ledger: cache hits, routing spread, cost."""
    from compound import call_ledger

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"error: no call ledger at {path}")
    records = call_ledger.load_records(path)
    if not records:
        raise SystemExit(f"error: {path} has no readable rows")
    rows = call_ledger.summarize(records)
    print(f"call ledger: {len(records)} call(s) over {len(rows)} route(s) — {path}\n")
    print(call_ledger.format_summary(rows))
    if args.hosts:
        print("\nupstreams that answered, per route:")
        for row in rows:
            spread = ", ".join(
                f"{host} {n}" for host, n in sorted(row["upstreams"].items(), key=lambda kv: -kv[1])
            )
            print(f"  {row['route']}: {spread or '—'}")
    return 0


def _apply_tb_env(args: argparse.Namespace) -> None:
    """Thread the terminal_bench pinning flags into the env the proxy reads.

    The pinning proxy and adapter read ``COMPOUND_REASONING`` /
    ``COMPOUND_DW_CACHE`` / ``COMPOUND_TB_TIMEOUT_MULT`` at request time, so a
    flag takes effect by setting its env var in this process before either the
    single-host or the sweep path is dispatched. Precedence per issue:

    * ``--reasoning`` wins over a pre-set ``COMPOUND_REASONING``; ``default``
      clears it so nothing is injected. Omitted, the env var is left untouched.
    * ``--cache-optin`` / ``--no-cache-optin`` set ``COMPOUND_DW_CACHE`` to ``1``/``0``.
      Neither given, the env var is left alone and markers default to ON
      (see :func:`compound.orproxy.cache_optin_enabled`).
    * ``--call-ledger PATH`` sets ``COMPOUND_CALL_LEDGER``, turning on the
      per-call record; unset, the proxy does no extra work.
    * ``--tb-timeout-mult`` sets ``COMPOUND_TB_TIMEOUT_MULT`` only when it is not
      already set, so a shell-exported multiplier wins over the flag.
    """
    import os

    if args.reasoning is not None:
        if args.reasoning == "default":
            os.environ.pop("COMPOUND_REASONING", None)
        else:
            os.environ["COMPOUND_REASONING"] = args.reasoning
    if args.cache_optin is not None:
        os.environ["COMPOUND_DW_CACHE"] = "1" if args.cache_optin else "0"
    if args.call_ledger:
        os.environ["COMPOUND_CALL_LEDGER"] = args.call_ledger
    if args.tb_timeout_mult is not None and "COMPOUND_TB_TIMEOUT_MULT" not in os.environ:
        os.environ["COMPOUND_TB_TIMEOUT_MULT"] = str(args.tb_timeout_mult)


def _tb_pin_line() -> str:
    """One line naming the effective reasoning / cache / timeout pinning."""
    import os

    from compound.orproxy import cache_optin_enabled

    reasoning = os.getenv("COMPOUND_REASONING", "").lower()
    reasoning = reasoning if reasoning in ("on", "off") else "default"
    cache = cache_optin_enabled()
    mult = os.getenv("COMPOUND_TB_TIMEOUT_MULT", "").strip() or "1"
    extended = " (extended limits, non-official)" if mult not in ("", "1", "1.0") else ""
    ledger = os.getenv("COMPOUND_CALL_LEDGER", "").strip() or "off"
    return (
        f"pinning: reasoning={reasoning} cache_optin={cache} timeout_mult={mult}{extended}\n"
        f"call ledger: {ledger}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    bench = BENCHMARKS[args.benchmark]
    explicit = args.tasks.split(",") if args.tasks else None
    case_ids = select_case_ids(
        _manifest_cases(args, bench),
        partition=args.partition,
        contains=args.contains,
        explicit=explicit,
    )
    if not case_ids:
        raise SystemExit("error: selection matches no cases")
    if args.benchmark == "terminal_bench":
        # Set before dispatch so both the single-host path and the provider
        # sweep (both proxied in-process) see the same pinning.
        _apply_tb_env(args)
    if args.providers:
        return _run_sweep(args, case_ids)
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

    prepare = sub.add_parser("prepare", help="install a benchmark's engine or build its manifest")
    prepare.add_argument("benchmark", choices=("tau2", "mmlu", "terminal_bench"))
    prepare.add_argument(
        "--per-subject", type=int, default=5, help="mmlu: test questions per subject"
    )

    providers = sub.add_parser(
        "providers",
        help="list the OpenRouter upstreams that serve a model, as --providers tokens",
    )
    providers.add_argument(
        "model", help="OpenRouter model slug, e.g. deepseek/deepseek-v4-flash-0731"
    )
    providers.add_argument(
        "--json", action="store_true", dest="as_json", help="machine-readable output"
    )
    providers.add_argument(
        "--probe",
        action="store_true",
        help="send one tiny pinned call to every host and report what it actually did. "
             "OpenRouter's 'up' is its own belief; without your own upstream key you sit "
             "on its shared rate-limit pool, where a listed-up host can 429 every call.",
    )

    tasks = sub.add_parser("tasks", help="print case ids for a benchmark")
    tasks.add_argument("benchmark", choices=sorted(BENCHMARKS))
    tasks.add_argument("--partition", help="filter to one partition")
    tasks.add_argument("--contains", help="case-insensitive substring filter")

    serving = sub.add_parser(
        "serving",
        help="serving-metrics harness: TTFT/decode/cost per host per reasoning mode",
    )
    serving.add_argument(
        "--providers",
        required=True,
        help="comma-separated provider tokens, e.g. "
        "openrouter/deepinfra,doubleword/flex,openrouter/auto",
    )
    serving.add_argument(
        "--shapes",
        required=True,
        help="JSON file mapping name -> {messages, response_format}",
    )
    serving.add_argument(
        "--model-or", dest="model_or", help="model slug for OpenRouter routes"
    )
    serving.add_argument("--model", help="model slug for Doubleword/direct routes")
    serving.add_argument(
        "--rounds", type=int, default=1, help="scheduled rounds (time-of-day variance)"
    )
    serving.add_argument(
        "--interval", type=float, default=3600.0, help="seconds between rounds"
    )
    serving.add_argument(
        "--reps", type=int, default=2, help="repetitions per (route, mode, shape) cell"
    )
    serving.add_argument(
        "--out",
        default="artifacts/bench/serving-metrics",
        help="output dir for results.jsonl",
    )

    harbor_p = sub.add_parser(
        "harbor",
        help="run Terminal-Bench 4.0 / any Harbor dataset across pinned hosts",
    )
    harbor_p.add_argument(
        "--providers",
        required=True,
        help="comma-separated provider tokens, e.g. openrouter/auto,openrouter/deepinfra",
    )
    harbor_p.add_argument("--model", required=True, help="model id as the upstream knows it")
    harbor_p.add_argument(
        "--task-path",
        default=None,
        help="run a Harbor task or dataset DIRECTORY on disk instead of a hub dataset "
             "(harbor --path). Lets a benchmark whose own runner is unreleased still run, "
             "as long as its tasks carry a Harbor task.toml.",
    )
    harbor_p.add_argument(
        "--host-model", action="append", default=[], metavar="HOST=MODEL",
        help="model id to send to one host when it names the weights differently, "
             "repeatable; HOST is a provider token, label, or kind "
             "(e.g. doubleword=zai-org/GLM-5.3-Flash)",
    )
    harbor_p.add_argument(
        "--dataset", default=_HARBOR_DEFAULT_DATASET,
        help="Harbor dataset name@version (pinned, not @latest, so the task set "
        "cannot shift between arms of one experiment)",
    )
    harbor_p.add_argument(
        "--agent", default=_HARBOR_DEFAULT_AGENT,
        help="Harbor agent. Must be a terminus-family agent when pinning: an "
        "in-sandbox agent cannot reach a localhost proxy.",
    )
    harbor_p.add_argument("--tasks", help="comma-separated task names (glob patterns allowed)")
    harbor_p.add_argument("--n-tasks", type=int, default=None, help="cap tasks after filtering")
    harbor_p.add_argument("--attempts", "-k", type=int, default=1, help="attempts per task")
    harbor_p.add_argument("--n-concurrent", type=int, default=4, help="concurrent trials")
    harbor_p.add_argument(
        "--timeout-multiplier", type=float, default=None,
        help="scale EVERY phase's time limit, environment build included "
        "(Harbor-native; runs are non-official when set)",
    )
    harbor_p.add_argument(
        "--agent-timeout-multiplier", type=float, default=None,
        help="scale only how long the agent may work, leaving environment build "
        "and verification alone. This is the flag for bounding a run: TB4 tasks "
        "allow the agent 8 hours by default.",
    )
    harbor_p.add_argument(
        "--ak", "--agent-kwarg", dest="agent_kwargs", action="append", default=None,
        metavar="KEY=VALUE",
        help="agent constructor kwarg, repeatable. Use max_turns=N to give every "
        "host the same work: an equal wall clock hands a faster host more turns.",
    )
    harbor_p.add_argument("--env", default="docker", help="Harbor environment backend")
    harbor_p.add_argument("--jobs-dir", default="artifacts/harbor", help="where jobs land")
    harbor_p.add_argument("--ledger-dir", default=None, help="per-host call ledger directory")
    harbor_p.add_argument("--reasoning", choices=("on", "off", "default"), default=None,
                          help="pin the model's reasoning mode via the proxy")
    harbor_p.add_argument(
        "--cache-optin", action=argparse.BooleanOptionalAction, default=None,
        help="inject explicit prompt-cache markers for explicit_marker providers "
        "(e.g. doubleword). ON by default; --no-cache-optin measures the unmarked path.",
    )
    harbor_p.add_argument("--call-ledger", default=None, help=argparse.SUPPRESS)
    harbor_p.add_argument("--tb-timeout-mult", default=None, help=argparse.SUPPRESS)
    harbor_p.add_argument("--go", action="store_true", help="execute (default is a dry run)")

    ledger = sub.add_parser(
        "ledger", help="summarize a call ledger (cache hits, routing spread, cost)"
    )
    ledger.add_argument("path", help="path to a calls.jsonl written by --call-ledger")
    ledger.add_argument(
        "--hosts",
        action="store_true",
        help="also list which upstreams answered each route, with counts",
    )

    run = sub.add_parser("run", help="run a task subset (dry run unless --go)")
    run.add_argument("benchmark", choices=sorted(BENCHMARKS))
    run.add_argument("--model", required=True, help="model id as the provider knows it")
    run.add_argument("--tasks", help="comma-separated case ids (see `tasks`)")
    run.add_argument("--partition", help="or: every case in one partition")
    run.add_argument("--manifest", help="override the benchmark's task manifest (more tasks)")
    run.add_argument("--contains", help="or: every case id matching a substring")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--go", action="store_true", help="actually spend; default is a dry run")
    # provider sweep: same model across many hosts (tau2, terminal_bench)
    run.add_argument(
        "--providers",
        help="comma-separated provider tokens to sweep, e.g. "
        "openrouter/deepinfra,openrouter/baseten,doubleword/flex",
    )
    # tau2 routing (any provider, any model)
    run.add_argument(
        "--provider",
        default="openrouter",
        help="tau2: openrouter, doubleword, or a label for --api-base",
    )
    run.add_argument("--api-base", help="tau2: custom OpenAI-compatible endpoint")
    run.add_argument("--api-key-env", help="tau2: env var holding the key for --api-base")
    run.add_argument("--upstream", help="tau2: pin one OpenRouter upstream (fallbacks disabled)")
    run.add_argument("--tier", help="tau2: service tier flag (e.g. doubleword flex)")
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument(
        "--user-model", default="openai/gpt-5.6-luna", help="tau2: user simulator (OpenRouter)"
    )
    run.add_argument("--output", help="tau2: episode output dir")
    # bfcl / ds1000 delegation
    run.add_argument("--cap", type=float, default=4.0, help="bfcl/ds1000: per-run USD cap")
    # terminal_bench delegation
    run.add_argument("--tb-agent", help="terminal_bench: harness agent (default terminus)")
    run.add_argument("--tb-concurrent", type=int, default=2, help="terminal_bench: tasks per host")
    run.add_argument(
        "--reasoning",
        choices=("on", "off", "default"),
        default=None,
        help="terminal_bench: pin the model's reasoning mode via the proxy "
        "(on/off), or 'default' to inject nothing. Given, the flag wins over a "
        "pre-set COMPOUND_REASONING; omitted, that env var is honored.",
    )
    run.add_argument(
        "--cache-optin",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="terminal_bench: inject explicit prompt-cache markers for "
        "explicit_marker providers (e.g. doubleword). ON by default, because a "
        "marker-gated host otherwise re-bills the whole transcript every turn; "
        "pass --no-cache-optin to measure that unmarked path on purpose. "
        "COMPOUND_DW_CACHE overrides when neither flag is given.",
    )
    run.add_argument(
        "--call-ledger",
        default=None,
        metavar="PATH",
        help="record one JSONL row per model call (route, provider echo, tokens, "
        "cached tokens, cost, status, latency). The per-call record is what "
        "supports cache-hit and routing claims; episode results cannot.",
    )
    run.add_argument(
        "--tb-timeout-mult",
        type=float,
        default=None,
        help="terminal_bench: multiply every task's max_agent_timeout_sec by N "
        "(extended-limits mode; results are labeled non-official). A pre-set "
        "COMPOUND_TB_TIMEOUT_MULT wins over this flag.",
    )

    args = parser.parse_args()
    if args.command == "list":
        return cmd_list()
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "providers":
        return cmd_providers(args.model, args.as_json, args.probe)
    if args.command == "tasks":
        return cmd_tasks(args.benchmark, args.partition, args.contains)
    if args.command == "serving":
        return cmd_serving(args)
    if args.command == "ledger":
        return cmd_ledger(args)
    if args.command == "harbor":
        return cmd_harbor(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
