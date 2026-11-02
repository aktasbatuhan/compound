"""Regenerate the docs' CLI reference from the live ``--help`` text.

The reference page is written from the commands themselves, so it cannot drift
from the code; CI runs this and fails if the committed page differs.

    uv run python scripts/build_cli_reference.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website/src/content/docs/reference/cli.md"

SUBCOMMANDS = [
    "estimate", "prices", "shape", "gateway-import", "gateway-activity", "replay", "simulate",
    "serving", "providers", "list", "prepare", "tasks", "run", "harbor", "ledger",
    "cache-study", "serving-report", "validate-config",
]


def cli_help(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(could not run {' '.join(cmd)}: {exc})"
    return (out.stdout or out.stderr).strip()


def reference_markdown() -> str:
    parts = [
        "---",
        "title: CLI reference",
        "description: Every compound command and flag, generated from --help.",
        "sidebar:",
        "  order: 1",
        "---",
        "",
        "Generated from `--help` by `scripts/build_cli_reference.py`, so it matches the "
        "checked-in CLI. `compound` is installed by `uv sync --extra dev`, and "
        "`python -m compound.cli` is the same entry point. Do not edit this page by hand.",
        "",
    ]
    for sub in SUBCOMMANDS:
        text = cli_help(["uv", "run", "compound", sub, "--help"])
        parts += [f"## compound {sub}", "", "```text", text, "```", ""]
    text = cli_help(["bun", "run", "compound", "--help"])
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = "\n".join(line for line in text.splitlines() if not line.startswith("$ bun run"))
    parts += ["## Trace pipeline (TypeScript)", "", "`bun run compound`, until it moves into "
              "the Python package.", "", "```text", text.strip(), "```", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(reference_markdown())
    print(f"wrote {OUT.relative_to(ROOT)}")
