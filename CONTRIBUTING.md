# Contributing

Thanks for looking. Compound measures how the hosts serving an open model differ
on a workload. Contributions that add a host, a benchmark, or a measurement are
the most useful, and so are bug reports with a dry-run command attached.

## Setup

```bash
git clone https://github.com/aktasbatuhan/compound
cd compound
uv sync --extra dev        # Python: benchmark engine, proxy, ledger
bun install                # TypeScript: trace pipeline (optional)
cp .env.example .env       # keys; never committed
```

## Before you open a pull request

```bash
uv run ruff check src tests scripts
uv run pytest -q
bun run lint && bun run typecheck && bun test packages
python3 scripts/build_site.py      # only if you touched site/docs-src
```

CI runs the same commands. The site job fails if `site/docs/` is out of date
with `site/docs-src/`, so commit the rebuilt pages with the source change.

## What makes a change easy to merge

- **Every number has a record.** A change that reports a new metric should say
  where it comes from (a usage block, a timer, a declared price) and whether it
  is measured or derived. `None` means the host did not report it; do not
  substitute zero.
- **Dry run first.** Anything that can spend money stays behind `--go` or
  `--paid`, and the dry run prints what would be executed.
- **Pinning is verified, not assumed.** A new provider kind must record the
  served host on every call so a pinned run can be checked afterwards.
- **Tests next to the code.** Python tests live in `tests/`, TypeScript tests
  next to the package they cover. A bug fix comes with the test that failed.
- **One thing per pull request.** A benchmark adapter, a provider kind, or a
  report column each stand on their own.

## Adding a benchmark

A benchmark is one registry entry: a manifest of partitioned case ids plus a
runner that accepts `(models, case_ids)`. Read `src/compound/adapters/` for the
interface and the five shipped adapters, and keep the official grader.

## Adding a provider

Most hosts need no code. An OpenAI-compatible endpoint is a `providers.<name>`
block in `compound.yaml`, addressed as `direct/<name>`. Open an issue if a host
needs a different pinning dialect, cache marker, or usage format; those go in
`src/compound/providers_registry.py` and `src/compound/orproxy.py`.

## Reporting a problem

Include the exact command, the dry-run output, and the ledger row or result
file that looks wrong. Do not paste API keys or full transcripts of private
traces.
