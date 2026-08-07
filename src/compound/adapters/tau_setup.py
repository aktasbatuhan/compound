"""Make tau2-bench self-serve, so `compound.bench run tau2` works from a fresh
clone the same way `mmlu` and `terminal_bench` do.

tau2 is Sierra's public benchmark (github.com/sierra-research/tau2-bench,
Apache-2.0). Our tau adapter imports it in-process, so it must be installed in
the same environment that runs `compound.bench`. `prepare_tau2()` clones the
repo at a pinned commit and installs it editable; `ensure_tau2()` is the
preflight `run tau2` calls to fail with a clear pointer instead of a bare
ImportError. The nl-assertions judge reroute is a runtime monkeypatch (see
compound.adapters.tau.route_nl_judge_via_openrouter), so no source patch of the
clone is needed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

TAU2_REPO = "https://github.com/sierra-research/tau2-bench.git"
# Pinned to the commit our published numbers were produced against.
TAU2_COMMIT = "07d5b33"
TAU2_DIR = Path(".compound/sources/tau2-bench")

PREPARE_HINT = "python -m compound.bench prepare tau2"


def tau2_installed() -> bool:
    """True if `import tau2` would succeed in the current interpreter."""
    return importlib.util.find_spec("tau2") is not None


def ensure_tau2() -> None:
    """Preflight for `run tau2`: raise a pointed error if tau2 is missing."""
    if not tau2_installed():
        raise SystemExit(
            "error: tau2-bench is not installed in this environment.\n"
            f"       Run `{PREPARE_HINT}` once to clone and install it "
            "(github.com/sierra-research/tau2-bench)."
        )


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepare_tau2(
    *,
    dir: Path = TAU2_DIR,
    commit: str = TAU2_COMMIT,
    installer: object = None,
) -> Path:
    """Clone tau2-bench at the pinned commit (if absent) and install it editable.

    `installer` is an injection seam for tests; it defaults to running git and
    pip as subprocesses. Returns the checkout path.
    """
    run = installer or _run  # type: ignore[assignment]
    dir = Path(dir)
    if not (dir / ".git").exists():
        dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", TAU2_REPO, str(dir)])
        run(["git", "-C", str(dir), "checkout", commit])
    else:
        print(f"tau2-bench already checked out at {dir}")
    # Install into the interpreter running compound.bench so `import tau2` works.
    run([sys.executable, "-m", "pip", "install", "-e", str(dir)])
    return dir
