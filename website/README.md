# Compound docs

The documentation site, built with [Starlight](https://starlight.astro.build).
Pages are Markdown under `src/content/docs/`, one folder per sidebar section.

```bash
npm ci
npm run dev       # local preview with live reload
npm run build     # static site in dist/
```

Both commands first copy `../site/report` into `public/report`, so the
generated reports keep their `/report/` addresses. The CLI reference page is
generated: run `uv run python scripts/build_cli_reference.py` from the repo
root instead of editing it.
