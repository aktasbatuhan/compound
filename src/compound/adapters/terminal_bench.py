"""Terminal-Bench adapter: agentic terminal tasks via the official harness.

Compound does not reimplement the benchmark: episodes execute inside the real
``terminal-bench`` harness (Docker containers, official tests, official
parser), invoked through ``uvx`` with a pinned version. Our side contributes
the partitioned manifest, subset selection, and money discipline.

Setup (one-time, ~80 task definitions):

    uvx --with 'litellm<1.95' terminal-bench@0.2.18 datasets download \
        --name terminal-bench-core --version 0.1.1 \
        --output-dir .compound/sources/terminal-bench-core
    python -m compound.bench prepare terminal_bench
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from compound.adapters.mmlu import stable_partition
from compound.orproxy import cache_optin_enabled

TB_VERSION = "terminal-bench@0.2.18"
# 0.2.18 resolves to a litellm whose native extension fails to build on macOS;
# any earlier litellm satisfies the harness' >=1.67.5 floor.
TB_UVX = ["uvx", "--with", "litellm<1.95", TB_VERSION]
DATASET_DIR = Path(".compound/sources/terminal-bench-core")
DEFAULT_AGENT = "terminus"
#: Sidecar recording each task's shipped ``max_agent_timeout_sec`` so an
#: extended-limits patch stays idempotent and reversible (see
#: :func:`apply_timeout_mult`).
TIMEOUT_BASE_FILE = ".compound_timeout_base.json"


def build_manifest(output_path: str | Path, dataset_dir: str | Path = DATASET_DIR) -> Path:
    source = Path(dataset_dir)
    task_files = sorted(source.glob("*/task.yaml"))
    if not task_files:
        raise SystemExit(
            f"error: no tasks under {source} — download terminal-bench-core first "
            "(see compound.adapters.terminal_bench docstring)"
        )
    cases = []
    for task_file in task_files:
        task = yaml.safe_load(task_file.read_text())
        case_id = task_file.parent.name
        cases.append(
            {
                "case_id": case_id,
                "partition": stable_partition(f"terminal_bench:{case_id}"),
                "metadata": {
                    "difficulty": task.get("difficulty", "unknown"),
                    "category": task.get("category", "unknown"),
                    "max_agent_timeout_sec": task.get("max_agent_timeout_sec"),
                },
            }
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "benchmark": "terminal_bench",
                "source": "terminal-bench-core==0.1.1",
                "cases": cases,
            },
            indent=1,
        )
    )
    return destination


def timeout_mult_from_env() -> float | None:
    """The extended-limits multiplier for this run, or ``None`` when unset.

    Reads ``COMPOUND_TB_TIMEOUT_MULT``, which carries either a value exported by
    the cloud shell wrapper or the ``--tb-timeout-mult`` flag (bench.py threads
    the flag into this env var; a pre-set shell value wins). A non-positive or
    unparseable value is treated as unset, so a typo never silently shrinks the
    task clock.
    """
    raw = os.getenv("COMPOUND_TB_TIMEOUT_MULT", "").strip()
    if not raw:
        return None
    try:
        mult = float(raw)
    except ValueError:
        return None
    return mult if mult > 0 else None


def apply_timeout_mult(dataset_dir: str | Path, mult: float) -> int:
    """Scale every task's ``max_agent_timeout_sec`` by ``mult``, idempotently.

    terminal-bench reads each task's wall-clock limit from its ``task.yaml``. On
    slow-decoding hosts a thinking-mode episode burns that limit on API time
    alone, so an extended-limits run separates capability from serving speed.
    Each task's shipped limit is recorded once in :data:`TIMEOUT_BASE_FILE` and
    every patch recomputes from that base, so re-running never compounds and
    ``mult == 1.0`` restores the shipped limits. Returns the number of tasks
    patched.

    This is the Python twin of the ``COMPOUND_TB_TIMEOUT_MULT`` shell patch.
    Runs made this way deviate from official terminal-bench limits and must be
    labeled as such (run_metadata.json), never compared to official numbers.
    """
    root = Path(dataset_dir)
    base_path = root / TIMEOUT_BASE_FILE
    base: dict[str, Any] = (
        json.loads(base_path.read_text()) if base_path.exists() else {}
    )
    patched = 0
    for task_file in sorted(root.glob("*/task.yaml")):
        task_id = task_file.parent.name
        task = yaml.safe_load(task_file.read_text())
        if task_id not in base:
            if task.get("max_agent_timeout_sec") is None:
                continue
            base[task_id] = task["max_agent_timeout_sec"]
        original = base[task_id]
        if original is None:
            continue
        task["max_agent_timeout_sec"] = original * mult
        task_file.write_text(yaml.safe_dump(task, sort_keys=False))
        patched += 1
    base_path.write_text(json.dumps(base, indent=1))
    return patched


def write_run_metadata(
    output_dir: str | Path, *, model: str, agent: str, timeout_mult: float | None
) -> Path:
    """Record how a run was pinned so reports can label it (issues #42/#43/#44).

    ``reasoning_mode`` and ``cache_optin`` are read from the same env vars the
    pinning proxy reads (``COMPOUND_REASONING`` / ``COMPOUND_DW_CACHE``), so the
    metadata reflects exactly what was injected whether the flag or the env var
    set it. ``extended_limits`` marks a run whose task clock was stretched: its
    numbers must never be presented as official-limit terminal-bench results.
    Returns the path to the written ``run_metadata.json``.
    """
    reasoning = os.getenv("COMPOUND_REASONING", "").lower()
    reasoning_mode = reasoning if reasoning in ("on", "off") else "default"
    cache_optin = cache_optin_enabled()
    mult = timeout_mult if timeout_mult else 1.0
    meta = {
        "model": model,
        "agent": agent,
        "reasoning_mode": reasoning_mode,
        "cache_optin": cache_optin,
        "tb_timeout_mult": mult,
        "extended_limits": mult != 1.0,
        "official_limits": mult == 1.0,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def run_terminal_bench(
    case_ids: list[str],
    *,
    model: str,
    output_dir: str | Path,
    agent: str = DEFAULT_AGENT,
    n_concurrent: int = 2,
    dataset_dir: str | Path = DATASET_DIR,
    extra_env: dict[str, str] | None = None,
    run_id: str | None = None,
) -> int:
    """Run a task subset through the official harness. Returns its exit code.

    ``model`` is a litellm-style name (e.g. ``openrouter/moonshotai/kimi-k3``);
    the harness bills through the matching provider env key. A custom
    OpenAI-compatible host works via ``openai/<model>`` plus OPENAI_API_BASE and
    OPENAI_API_KEY in the environment. ``extra_env`` overlays the subprocess
    environment, which is how a per-host proxy (:mod:`compound.orproxy`) points
    the harness at a pinned upstream without touching this process' env.
    """
    if shutil.which("docker") is None:
        raise SystemExit("error: terminal-bench needs Docker running")
    if shutil.which("uvx") is None:
        raise SystemExit("error: terminal-bench is invoked through uvx (install uv)")
    if not Path(dataset_dir).exists():
        raise SystemExit(f"error: dataset missing at {dataset_dir} (see adapter docstring)")
    # Extended-limits mode (issue #44): honor COMPOUND_TB_TIMEOUT_MULT whether it
    # came from the cloud shell or the --tb-timeout-mult flag. Patch before the
    # harness reads task.yaml; the patch is idempotent so a resumed run is safe.
    mult = timeout_mult_from_env()
    if mult is not None and mult != 1.0:
        patched = apply_timeout_mult(dataset_dir, mult)
        print(f"terminal_bench: extended agent time limits x{mult:g} on {patched} task(s)")
    write_run_metadata(output_dir, model=model, agent=agent, timeout_mult=mult)
    command = [
        *TB_UVX,
        "run",
        "--dataset-path",
        str(dataset_dir),
        "--agent",
        agent,
        "--model",
        model,
        "--output-path",
        str(output_dir),
        "--n-concurrent",
        str(n_concurrent),
        "--no-livestream",
    ]
    # A unique run-id namespaces the harness' Docker compose project names, so
    # several hosts can run the same task concurrently without colliding on a
    # container name (the names are otherwise `<task>-<trial>-<timestamp>`).
    if run_id:
        command.extend(["--run-id", run_id])
    for case_id in case_ids:
        command.extend(["--task-id", case_id])
    print("exec:", " ".join(command))
    env = {**os.environ, **extra_env} if extra_env else None
    return subprocess.call(command, env=env)
