"""Run Terminal-Bench 4.0 (and any Harbor dataset) through the pinning proxy.

Terminal-Bench moved. The ``terminal-bench`` PyPI package our other adapter
drives is now Terminal-Bench 1, frozen at 0.2.18; TB 4.0 (released 2026-08-26)
lives in a new repo and runs on `Harbor <https://harborframework.com>`_, which
resolves datasets from a hub, owns sandboxing, and runs trials concurrently.

Three things we hand-built for TB1 are native here, so this adapter delegates
rather than reimplements them:

* ``--timeout-multiplier`` replaces our ``task.yaml`` patching. Harbor scales
  the limits itself, so nothing on disk is mutated and no base file is needed.
  It scales *every* phase though, including the environment build, so shrinking
  it to bound a run kills the container before it starts
  (``EnvironmentStartTimeoutError``). Capping only how long the agent may work
  is ``agent_timeout_multiplier``, which leaves build and verification alone.
* ``-k`` runs repeated attempts per task, replacing our per-trial subdirectories.
* ``-n`` plus a sandbox backend runs trials concurrently, which is what turns a
  full provider matrix from an overnight VM fan-out into a single job.

What stays ours is the pinning. Harbor picks a model, not a serving host, so
"same weights, different host" still needs the proxy: the agent is pointed at
``openai/<model>`` with ``OPENAI_API_BASE`` on a local :mod:`compound.orproxy`,
and that proxy injects the routing and records every call to the ledger. This
works because ``terminus``-family agents run in the harness process and drive
the sandbox over tmux, so their model calls originate outside the container. An
*installed* agent (claude-code, codex, qwen-code) calls the model from inside
the sandbox instead, where ``127.0.0.1`` is the container: pinning such an agent
needs a proxy address the sandbox can reach, which is why :func:`build_command`
refuses to pretend and raises instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

#: Harbor is invoked through uvx so the project pins nothing: Harbor requires
#: Python >= 3.12 and this project runs on 3.11.
HARBOR_UVX = ["uvx", "--from", "harbor", "harbor"]

#: Pinned rather than ``@latest``: a benchmark that "evolves over time" would
#: otherwise silently change the task set between two arms of one experiment,
#: which is exactly the confound the whole matrix exists to avoid.
DEFAULT_DATASET = "terminal-bench/terminal-bench@4.0.0"

#: The terminus family drives the sandbox over tmux from the harness process,
#: so its model calls leave from here and a localhost proxy can pin them.
DEFAULT_AGENT = "terminus-2"

#: Agents that call the model from inside the sandbox. A localhost proxy is not
#: reachable from there, so pinning them needs a routable proxy address.
IN_SANDBOX_AGENTS = frozenset(
    {
        "aider", "antigravity-cli", "antigravity-sdk", "claude-code", "cline-cli",
        "codex", "copilot-cli", "cortex-code", "cursor-cli", "gemini-cli", "goose",
        "junie", "kimi-cli", "kimi-code", "mini-swe-agent", "opencode", "openhands",
        "openhands-sdk", "qwen-coder", "rovodev-cli", "swe-agent",
    }
)


def build_command(
    *,
    dataset: str = DEFAULT_DATASET,
    model: str,
    agent: str = DEFAULT_AGENT,
    jobs_dir: str | Path,
    job_name: str,
    include_tasks: list[str] | None = None,
    n_tasks: int | None = None,
    attempts: int = 1,
    n_concurrent: int = 4,
    timeout_multiplier: float | None = None,
    agent_timeout_multiplier: float | None = None,
    env_type: str = "docker",
    allow_agent_hosts: list[str] | None = None,
    proxied: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The ``harbor run`` argv for one arm of a sweep.

    Pure, so the exact command a run would issue is asserted in tests and
    printed on a dry run rather than discovered by spending money.

    ``proxied`` marks that the model is served through a local pinning proxy.
    Combined with an agent that calls the model from inside the sandbox, that
    is unsatisfiable, so it raises rather than producing a run whose "pinned"
    arm quietly reaches the public endpoint and reports a host we never chose.
    """
    if proxied and agent in IN_SANDBOX_AGENTS:
        raise ValueError(
            f"agent {agent!r} calls the model from inside the sandbox, where a "
            "localhost pinning proxy is unreachable. Use a terminus-family agent, "
            "or give the proxy an address the sandbox can reach."
        )
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    command = [
        *HARBOR_UVX,
        "run",
        "--dataset", dataset,
        "--agent", agent,
        "--model", model,
        "--jobs-dir", str(jobs_dir),
        "--job-name", job_name,
        "--n-attempts", str(attempts),
        "--n-concurrent", str(n_concurrent),
        "--env", env_type,
        # Non-interactive: a sweep must never block on a host-access prompt.
        "--yes",
    ]
    for task in include_tasks or []:
        command += ["--include-task-name", qualify_task(task)]
    if n_tasks is not None:
        command += ["--n-tasks", str(n_tasks)]
    if timeout_multiplier is not None and timeout_multiplier != 1.0:
        command += ["--timeout-multiplier", str(timeout_multiplier)]
    if agent_timeout_multiplier is not None and agent_timeout_multiplier != 1.0:
        command += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]
    for host in allow_agent_hosts or []:
        command += ["--allow-agent-host", host]
    command += extra_args or []
    return command


def qualify_task(name: str) -> str:
    """Match a bare task name against Harbor's namespaced task ids.

    Harbor identifies a task by its dataset-qualified name
    (``terminal-bench/data-anonymization``), and ``--include-task-name`` matches
    that whole string, so a bare name silently matches nothing and the job dies
    with "No tasks matched the filter(s)". A name with no separator is widened
    to a ``*/`` glob so it matches under whichever dataset supplies it; a name
    that already carries a namespace, or its own glob, is passed through.
    """
    if "/" in name or name.startswith("*"):
        return name
    return f"*/{name}"


def proxy_env(base_url: str) -> dict[str, str]:
    """Environment that points a harness-side agent at the pinning proxy.

    Both spellings are set because Harbor accepts either for the ``openai``
    provider. The key is a placeholder: the proxy holds the real credential and
    injects it upstream, so no provider key is exposed to the agent or written
    into a job config.
    """
    return {
        "OPENAI_API_BASE": base_url,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": "proxy",
    }


def run_harbor(
    command: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> int:
    """Execute a built command, returning its exit code.

    ``extra_env`` overlays this process' environment rather than replacing it,
    which is how a per-host proxy points one arm at a pinned upstream without
    disturbing the parent or any concurrent arm.
    """
    if shutil.which("uvx") is None:
        raise SystemExit("error: harbor is invoked through uvx (install uv)")
    if shutil.which("docker") is None and "--env" in command:
        index = command.index("--env")
        if index + 1 < len(command) and command[index + 1] == "docker":
            raise SystemExit("error: the docker environment needs Docker running")
    env = {**os.environ, **(extra_env or {})}
    return subprocess.call(command, env=env, cwd=str(cwd) if cwd else None)


def job_result_path(jobs_dir: str | Path, job_name: str) -> Path:
    """Where Harbor writes the job-level result for a run."""
    return Path(jobs_dir) / job_name / "result.json"


def _resolved(trial: dict[str, Any]) -> bool | None:
    """Whether one trial passed, or ``None`` when it never produced a verdict.

    Harbor's own pass@k treats a single reward of 0 or 1 as the outcome. A trial
    that errored before verification has no rewards at all: that is not a
    failure of the model, it is a missing measurement, and collapsing the two
    is how infrastructure noise gets reported as a quality difference.
    """
    verifier = trial.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if not isinstance(rewards, dict) or len(rewards) != 1:
        return None
    value = next(iter(rewards.values()))
    if not isinstance(value, (int, float)) or value not in (0, 1):
        return None
    return bool(value)


def trial_rows(job_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a Harbor ``result.json`` into one row per trial.

    ``error`` carries the exception type when a trial died before verification,
    so a run's failures can be split into model failures and harness or
    infrastructure failures instead of being pooled into one pass rate.
    """
    rows: list[dict[str, Any]] = []
    for trial in job_result.get("trial_results") or []:
        exception = trial.get("exception_info") or {}
        agent_info = trial.get("agent_info") or {}
        model_info = agent_info.get("model_info") or {}
        rows.append(
            {
                "task_name": trial.get("task_name"),
                "trial_name": trial.get("trial_name"),
                "resolved": _resolved(trial),
                "error": exception.get("exception_type"),
                "agent": agent_info.get("name"),
                "model": model_info.get("name"),
                "started_at": trial.get("started_at"),
                "finished_at": trial.get("finished_at"),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve rate over a job, with unverified trials reported separately.

    ``resolve_rate`` is computed over trials that actually returned a verdict.
    ``unverified`` is reported next to it rather than folded in, because a run
    where a tenth of the trials never reached the verifier is a different claim
    from one where they ran and failed.
    """
    verdicts = [row["resolved"] for row in rows if row["resolved"] is not None]
    unverified = [row for row in rows if row["resolved"] is None]
    errors: dict[str, int] = {}
    for row in unverified:
        key = row["error"] or "no_verdict"
        errors[key] = errors.get(key, 0) + 1
    return {
        "trials": len(rows),
        "verdicts": len(verdicts),
        "resolved": sum(verdicts),
        "resolve_rate": (sum(verdicts) / len(verdicts)) if verdicts else None,
        "unverified": len(unverified),
        "errors": errors,
    }


def load_trial_results(jobs_dir: str | Path, job_name: str) -> list[dict[str, Any]]:
    """Every trial's own ``result.json`` under a job directory.

    Harbor writes each trial's result into its own subdirectory and does not
    always aggregate them into the job-level ``trial_results``: a finished job
    can carry an empty list there while every trial's outcome sits on disk.
    Reading only the job level therefore reported a completed arm as having run
    nothing, so the trial files are the source of truth and the job level is
    used only as a fallback.
    """
    root = Path(jobs_dir) / job_name
    results: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/result.json")):
        try:
            results.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return results


def load_job_summary(jobs_dir: str | Path, job_name: str) -> dict[str, Any]:
    """Read a finished job and summarize it, keeping Harbor's own totals.

    Harbor tracks its own token and cost totals from the agent's accounting.
    They are carried through as a cross-check against the call ledger, which
    measures the same run at the proxy: the ledger is authoritative for cost
    and cache, since it reads what the provider actually billed and returned.
    """
    path = job_result_path(jobs_dir, job_name)
    if not path.exists():
        raise SystemExit(f"error: no Harbor job result at {path}")
    job_result = json.loads(path.read_text())
    trials = job_result.get("trial_results") or load_trial_results(jobs_dir, job_name)
    rows = trial_rows({"trial_results": trials})
    stats = job_result.get("stats") or {}
    summary = summarize(rows)
    # A trial that died before producing a result is counted by Harbor but has
    # no entry in trial_results. Without this an arm where every trial errored
    # reads as an empty arm rather than a failed one.
    summary["errored_trials"] = stats.get("n_errored_trials") or 0
    summary["total_trials"] = job_result.get("n_total_trials")
    summary["harbor_stats"] = {
        "n_input_tokens": stats.get("n_input_tokens"),
        "n_cache_tokens": stats.get("n_cache_tokens"),
        "n_output_tokens": stats.get("n_output_tokens"),
        "cost_usd": stats.get("cost_usd"),
    }
    summary["rows"] = rows
    return summary
