"""Run one benchmark across many serving hosts from a provider-token list.

This is the "same model, many hosts" loop. Given a resolved list of
:class:`ProviderSpec` (see :mod:`compound.providers_registry`), it runs the
chosen benchmark once per host and lands each host's output under
``<output>/<label>/`` so the shared reporting step can diff them.

Two execution paths, one behaviour:

* **in-process** benchmarks (tau2) get the pinning injected directly by
  :meth:`ProviderSpec.to_tau_model`; and
* **external harness** benchmarks (terminal-bench) get the identical pinning
  through a per-host :mod:`compound.orproxy` the harness is pointed at.

Both refuse to spend on a dry run: :func:`plan` prints what each host would do.
"""

from __future__ import annotations

import os
from pathlib import Path

from compound.orproxy import serve_provider
from compound.providers_registry import ProviderSpec


def plan(specs: list[ProviderSpec], model: str) -> list[str]:
    """Human-readable dry-run plan, one line per host."""
    lines = []
    for spec in specs:
        pin = spec.proxy_injection() or "(host default)"
        lines.append(f"  {spec.label:22s} {model} -> {spec.forward_base_url}  pin={pin}")
    return lines


def sweep_tau2(
    specs: list[ProviderSpec],
    *,
    model: str,
    case_ids: list[str],
    manifest_path: str | Path,
    trials: int,
    max_steps: int,
    max_tokens: int | None,
    user_model_name: str,
    output_dir: Path,
) -> list[str]:
    """Run tau2 across hosts in-process. Returns the labels that completed."""
    from compound.adapters.tau import TauModel, run_tau_partition

    user = TauModel(provider="openrouter", model=user_model_name)
    done: list[str] = []
    for spec in specs:
        key_env = spec.required_key_env()
        if not os.getenv(key_env):
            print(f"skip {spec.label}: {key_env} not set")
            continue
        agent = spec.to_tau_model(model, max_tokens=max_tokens)
        host_dir = output_dir / spec.safe_label
        print(f"[tau2] {spec.label}: {agent.litellm_name()} ({spec.forward_base_url})")
        run_tau_partition(
            manifest_path=manifest_path,
            partition=None,
            agent_model=agent,
            user_model=user,
            candidate_instruction="",
            trials=trials,
            max_steps=max_steps,
            output_dir=host_dir / "episodes",
            telemetry_path=(host_dir / "telemetry.jsonl").resolve(),
            case_ids=set(case_ids),
        )
        done.append(spec.label)
    return done


def sweep_terminal_bench(
    specs: list[ProviderSpec],
    *,
    model: str,
    case_ids: list[str],
    output_dir: Path,
    agent: str,
    n_concurrent: int = 2,
) -> list[str]:
    """Run terminal-bench across hosts, each behind its own pinning proxy.

    ``model`` is the model id the upstream knows (e.g.
    ``deepseek/deepseek-v4-flash-0731``); the harness is invoked as
    ``openai/<model>`` against the local proxy, so litellm forwards the bare id
    and the proxy adds the host pinning.
    """
    import time

    from compound.adapters.terminal_bench import run_terminal_bench

    done: list[str] = []
    for spec in specs:
        key_env = spec.required_key_env()
        if not os.getenv(key_env):
            print(f"skip {spec.label}: {key_env} not set")
            continue
        host_dir = output_dir / spec.safe_label
        # Namespace the harness run per host so concurrent hosts running the same
        # task cannot collide on a Docker container name.
        run_id = f"{spec.safe_label}-{int(time.time())}"
        print(f"[terminal_bench] {spec.label}: pin -> {spec.forward_base_url} (run-id {run_id})")
        with serve_provider(spec) as base_url:
            code = run_terminal_bench(
                case_ids,
                model=f"openai/{model}",
                output_dir=host_dir,
                agent=agent,
                n_concurrent=n_concurrent,
                run_id=run_id,
                extra_env={
                    "OPENAI_API_BASE": base_url,
                    "OPENAI_API_KEY": "proxy",  # real key is injected by the proxy
                },
            )
        if code == 0:
            done.append(spec.label)
        else:
            print(f"terminal_bench {spec.label} exited {code}")
    return done


def sweep_harbor(
    specs: list[ProviderSpec],
    *,
    model: str,
    jobs_dir: Path,
    dataset: str,
    agent: str,
    include_tasks: list[str] | None = None,
    n_tasks: int | None = None,
    attempts: int = 1,
    n_concurrent: int = 4,
    timeout_multiplier: float | None = None,
    agent_timeout_multiplier: float | None = None,
    agent_kwargs: dict[str, str] | None = None,
    env_type: str = "docker",
    ledger_dir: Path | None = None,
) -> dict[str, dict]:
    """Run a Harbor dataset (Terminal-Bench 4.0) across hosts, each pinned.

    One Harbor job per host, named for the host so results stay separable, with
    each job's agent pointed at that host's proxy. The pinning, the reasoning
    mode and the call ledger all live in the proxy, so they apply here exactly
    as they do on the TB1 path without Harbor knowing anything about them.

    When ``ledger_dir`` is given each host also gets its own ledger file, since
    a shared one would still be correct (every row names its route) but is far
    less convenient to hand to a reviewer per arm.

    Returns the per-host summary keyed by label, so a caller can decide the next
    arm from the previous one, which is how the unpinned control run's routing
    distribution selects the hosts worth pinning.
    """
    import time

    from compound.adapters import harbor

    summaries: dict[str, dict] = {}
    for spec in specs:
        key_env = spec.required_key_env()
        if not os.getenv(key_env):
            print(f"skip {spec.label}: {key_env} not set")
            continue
        job_name = f"{spec.safe_label}-{int(time.time())}"
        print(f"[harbor] {spec.label}: pin -> {spec.forward_base_url} (job {job_name})")
        command = harbor.build_command(
            dataset=dataset,
            model=f"openai/{model}",
            agent=agent,
            jobs_dir=jobs_dir,
            job_name=job_name,
            include_tasks=include_tasks,
            n_tasks=n_tasks,
            attempts=attempts,
            n_concurrent=n_concurrent,
            timeout_multiplier=timeout_multiplier,
            agent_timeout_multiplier=agent_timeout_multiplier,
            agent_kwargs=agent_kwargs,
            env_type=env_type,
            proxied=True,
        )
        # The proxy runs in THIS process, so its signals must be set here. Only
        # the endpoint and key belong in the subprocess env, which is where the
        # agent reads them. Putting the ledger path in the subprocess env would
        # leave the proxy recording nothing at all.
        previous = {k: os.environ.get(k) for k in ("COMPOUND_CALL_LEDGER", "COMPOUND_RUN_LABEL")}
        if ledger_dir is not None:
            os.environ["COMPOUND_CALL_LEDGER"] = str(
                Path(ledger_dir) / f"{spec.safe_label}.jsonl"
            )
        os.environ["COMPOUND_RUN_LABEL"] = spec.label
        try:
            with serve_provider(spec) as base_url:
                code = harbor.run_harbor(command, extra_env=harbor.proxy_env(base_url))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if code != 0:
            print(f"harbor {spec.label} exited {code}")
            continue
        summaries[spec.label] = harbor.load_job_summary(jobs_dir, job_name)
    return summaries
