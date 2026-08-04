"""tau-bench adapter for `compound.viz`: sweep episodes -> route rows -> report.

Reads the episode results a `compound.tau_sweep` run produced, joins them with
the sweep spec's declared prices, and emits the generic viz row contract. The
rendering itself lives in `compound.viz` and is benchmark-agnostic; a new
benchmark needs only an extractor like this one (see issue #33).

    PYTHONPATH=src python -m compound.tau_report --output artifacts/tau-sweep/report.html
    PYTHONPATH=src python -m compound.tau_report --rows-out rows.json   # just the data
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from compound.viz import Route, write_report

SCORING_DOMAINS = ("airline", "retail")
DEFAULT_SWEEPS = [
    (Path("benchmarks/sweeps/tau-provider-sweep.json"), Path("artifacts/tau-sweep")),
    (Path("benchmarks/sweeps/tau-doubleword-flex.json"), Path("artifacts/tau-sweep-dwflex")),
]


def _model_key(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-FP8", "").replace("-0731", "").lower()


def _slug(model: str, upstream: str, tier: str | None) -> str:
    parts = [model.replace("/", "--")]
    if upstream.startswith("doubleword"):
        parts.append("at-doubleword-" + ("flex" if tier == "flex" else "realtime"))
    else:
        parts.append("at-" + upstream.replace("/", "-"))
    return "--".join(parts)


def extract(sweeps: list[tuple[Path, Path]]) -> list[Route]:
    rows: list[Route] = []
    for spec_path, base in sweeps:
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        for c in spec["configs"]:
            tier = c.get("service_tier")
            slug = _slug(c["model"], c["upstream"], tier)
            solved = n = 0
            lats: list[float] = []
            tps: list[float] = []
            tok_in = tok_out = 0
            served: set[str] = set()
            for dom in SCORING_DOMAINS:
                results = base / "episodes" / f"{dom}-all-{slug}.json" / "results.json"
                if not results.exists():
                    continue
                for sim in json.loads(results.read_text()).get("simulations", []):
                    if sim.get("termination_reason") == "infrastructure_error":
                        continue
                    n += 1
                    if (sim.get("reward_info") or {}).get("reward") == 1.0:
                        solved += 1
                    for m in sim.get("messages", []):
                        if m.get("role") != "assistant" or not m.get("usage"):
                            continue
                        u = m["usage"]
                        raw = m.get("raw_data") or {}
                        tok_in += u.get("prompt_tokens", 0) or 0
                        out = u.get("completion_tokens", 0) or 0
                        tok_out += out
                        g = m.get("generation_time_seconds")
                        if g and g > 0:
                            lats.append(g)
                            tps.append(out / g)
                        if raw.get("provider"):
                            served.add(raw["provider"])
            if n == 0:
                continue
            usd = (tok_in * c["usd_in_per_m"] + tok_out * c["usd_out_per_m"]) / 1e6
            rows.append(Route(
                model=_model_key(c["model"]), host=c["upstream"],
                quality=solved / n, quality_num=solved, quality_den=n,
                cost=usd / n,
                lat_p50=round(statistics.median(lats), 2) if lats else None,
                lat_p95=round(statistics.quantiles(lats, n=20)[18], 2) if len(lats) >= 20
                else (round(max(lats), 2) if lats else None),
                tps=round(statistics.median(tps), 1) if tps else None,
                quant=c.get("quant") or "unknown", served=sorted(served),
                flagged=bool(tier),
            ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="append", nargs=2, metavar=("SPEC", "DIR"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tau-sweep/report.html"))
    parser.add_argument("--rows-out", type=Path, help="also write the raw rows JSON")
    args = parser.parse_args()
    sweeps = ([(Path(s), Path(d)) for s, d in args.sweep] if args.sweep else DEFAULT_SWEEPS)
    rows = extract(sweeps)
    if not rows:
        print("no sweep results found; run compound.tau_sweep first")
        return 1
    if args.rows_out:
        args.rows_out.write_text(json.dumps([asdict(r) for r in rows], indent=1))
        print(f"{len(rows)} rows -> {args.rows_out}")
    write_report(
        rows, args.output,
        title=f"Same model, different host: quality, cost and speed across {len(rows)} routes",
        note="Interactive tau-bench episodes (agent, user simulator, live domain tools, official "
             "reward), each route pinned to one serving host with fallbacks disabled and the "
             "served host verified per call.",
        cost_label="cost per episode (USD, log scale, declared prices)",
        latency_label="median time per model call (seconds, log scale)",
    )
    print(f"{len(rows)} routes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
