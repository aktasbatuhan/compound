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

import yaml

from compound.adapters.mmlu import stable_partition

TB_VERSION = "terminal-bench@0.2.18"
# 0.2.18 resolves to a litellm whose native extension fails to build on macOS;
# any earlier litellm satisfies the harness' >=1.67.5 floor.
TB_UVX = ["uvx", "--with", "litellm<1.95", TB_VERSION]
DATASET_DIR = Path(".compound/sources/terminal-bench-core")
DEFAULT_AGENT = "terminus"


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
