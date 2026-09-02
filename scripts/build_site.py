#!/usr/bin/env python3
"""Render the site's docs pages from Markdown.

Sources live in ``site/docs-src/*.md``. Each file starts with a small header::

    title: Providers
    order: 20
    out: docs/providers        # optional; default docs/<stem>
    nav: docs                  # docs (sidebar) | top (no sidebar entry)

and renders to ``site/<out>/index.html`` inside a shared shell that links the
site stylesheet. ``reference.md`` is regenerated from the live ``--help`` text
of every ``compound-bench`` subcommand so the reference cannot drift from the
CLI.

Run from the repo root::

    python3 scripts/build_site.py
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

try:
    import markdown  # type: ignore
except ImportError:  # pragma: no cover
    sys.exit("python-markdown is required: pip install markdown")

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SRC = SITE / "docs-src"

BENCH_SUBCOMMANDS = ["list", "prepare", "providers", "tasks", "run", "harbor", "serving", "ledger"]


def parse(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text()
    meta: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines) and re.match(r"^[a-z]+: ", lines[i]):
        key, value = lines[i].split(": ", 1)
        meta[key] = value.strip()
        i += 1
    return meta, "\n".join(lines[i:]).lstrip("\n")


def cli_help(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(could not run {' '.join(cmd)}: {exc})"
    return (out.stdout or out.stderr).strip()


def reference_markdown() -> str:
    parts = [
        "title: CLI reference",
        "order: 90",
        "",
        "# CLI reference",
        "",
        "Generated from `--help` at build time, so it matches the checked-in CLI. "
        "`compound-bench` is installed by `uv sync --extra dev`; "
        "`python -m compound.bench` is the same entry point.",
        "",
    ]
    for sub in BENCH_SUBCOMMANDS:
        text = cli_help(["uv", "run", "compound-bench", sub, "--help"])
        parts += [f"## compound-bench {sub}", "", "```text", text, "```", ""]
    text = cli_help(["bun", "run", "compound", "--help"])
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = "\n".join(l for l in text.splitlines() if not l.startswith("$ bun run"))
    parts += ["## compound (trace pipeline, TypeScript)", "", "```text", text.strip(), "```", ""]
    return "\n".join(parts)


SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Compound</title>
<meta name="description" content="{description}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anybody:wght@400;500;600;700;800&family=Public+Sans:wght@400;500;600&family=Spline+Sans+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{root}assets/site.css">
</head>
<body>
<header>
  <div class="wrap nav">
    <a class="logo" href="{root}"><span class="glyph"></span>compound</a>
    <nav class="nav-links">
      <a href="{root}docs/"{docs_cur}>docs</a>
      <a href="{root}examples/"{ex_cur}>examples</a>
      <a class="gh-btn" href="https://github.com/aktasbatuhan/compound">GitHub</a>
    </nav>
  </div>
</header>
<main class="wrap docs-layout">
{sidebar}
  <article class="docs-body">
{body}
  </article>
</main>
<footer>
  <div class="wrap foot">
    <span>compound (Apache-2.0)</span>
    <span><a href="{root}docs/">docs</a> <a href="{root}examples/">examples</a> <a href="https://github.com/aktasbatuhan/compound/issues">issues</a></span>
  </div>
</footer>
</body>
</html>
"""


def build() -> None:
    (SRC / "reference.md").write_text(reference_markdown())
    pages = []
    for path in sorted(SRC.glob("*.md")):
        meta, body = parse(path)
        out = meta.get("out", f"docs/{path.stem}" if path.stem != "index" else "docs")
        pages.append((int(meta.get("order", 50)), meta, body, out, path))
    pages.sort(key=lambda p: p[0])
    nav_pages = [p for p in pages if p[1].get("nav", "docs") == "docs"]

    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc"])
    for _, meta, body, out, path in pages:
        depth = len(Path(out).parts)
        root = "../" * depth
        md.reset()
        rendered = md.convert(body)
        first_p = re.search(r"<p>(.*?)</p>", rendered, re.S)
        description = html.escape(re.sub(r"<[^>]+>", "", first_p.group(1)))[:300] if first_p else ""
        items = []
        for _, m2, _, out2, _ in nav_pages:
            cur = ' class="cur"' if out2 == out else ""
            items.append(f'    <a href="{root}{out2}/"{cur}>{html.escape(m2["title"])}</a>')
        in_docs = meta.get("nav", "docs") == "docs"
        sidebar = "  <nav class=\"docs-nav\">\n" + "\n".join(items) + "\n  </nav>" if in_docs else ""
        page = SHELL.format(
            title=html.escape(meta["title"]),
            description=description,
            root=root,
            docs_cur=' class="cur"' if in_docs else "",
            ex_cur=' class="cur"' if out == "examples" else "",
            sidebar=sidebar,
            body=rendered,
        )
        target = SITE / out / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page)
        print(f"{path.name} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
