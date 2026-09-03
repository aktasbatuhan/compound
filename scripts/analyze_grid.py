#!/usr/bin/env python3
"""Compare serving hosts across several models and tasks from per-call ledgers.

Layout this reads (what ``gcp-fswe-grid.sh`` produces, one tree per arm):

    <root>/<arm>/out/<model>/<task>/ledger/<route>.jsonl

The single-model analyzer (``analyze_arms.py``) names an arm after its ledger
file, which is right when one ledger is one arm. Here the same route appears
once per (model, task), so rows are grouped by (model, route) and pooled across
tasks, with a per-task breakdown underneath.

Two questions this answers that a single-model run cannot:

* Within a model, does a pinned host differ from the router on the rate of calls
  that never complete? Tested against the unpinned arm with a two-proportion
  z-test, Holm-corrected across the pinned arms of that model.
* Does the host ranking survive changing the model? A host that is cheapest and
  most reliable for one model and mid-pack for another is a property of the
  host-model pair, not of the host, and a reader picking a host for their own
  workload needs to know which one they are looking at.

Usage:  python3 scripts/analyze_grid.py artifacts/fswe-<stamp>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_arms import load_arm, summarize_arm, two_proportion_z  # noqa: E402


def collect(root: Path) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    """(model, route) -> task -> rows."""
    out: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for led in sorted(root.glob("**/ledger/*.jsonl")):
        # .../out/<model>/<task>/ledger/<route>.jsonl
        parts = led.parts
        try:
            task = parts[-3]
            model = parts[-4]
        except IndexError:  # pragma: no cover - malformed tree
            continue
        rows = load_arm(led)
        if rows:
            out[(model, led.stem)][task].extend(rows)
    return out


def table(title: str, arms: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    head = (
        f"{'route':<22s} {'calls':>6s} {'hang%':>7s} {'95% CI':>13s} {'aband':>6s} "
        f"{'cost$':>9s} {'$/1M ptok':>10s} {'cache%':>7s} {'p50s':>6s} {'p90s':>6s} {'hosts':>5s}"
    )
    print(head)
    print("-" * len(head))
    for a in sorted(arms, key=lambda x: x["hang_rate"]):
        lo, hi = a["hang_ci"]
        cache = "--" if a["cache_ratio"] is None else f"{a['cache_ratio'] * 100:.1f}"
        # A host that reports no cost per call (Doubleword) must print as
        # unreported, never as $0.0000: a null is not a measured zero, and a
        # zero here would read as "this host is free". See dw_cost_attribution.
        priced = a["priced_calls"] > 0
        cost = f"{a['cost_usd']:.4f}" if priced else "--"
        # Per MILLION prompt tokens: these runs move millions of tokens, and a
        # per-1k figure prints as a wall of zeroes.
        cpm = (
            f"{a['cost_per_1k_prompt'] * 1000:.4f}"
            if priced and a["cost_per_1k_prompt"] is not None
            else "--"
        )
        print(
            f"{a['arm']:<22s} {a['calls']:>6d} {a['hang_rate'] * 100:>6.1f}% "
            f"{f'{lo * 100:.1f}-{hi * 100:.1f}':>13s} {a['abandoned']:>6d} "
            f"{cost:>9s} {cpm:>10s} {cache:>7s} "
            f"{(a['p50_s'] or 0):>6.0f} {(a['p90_s'] or 0):>6.0f} {len(a['hosts']):>5d}"
        )


def holm(arms: list[dict[str, Any]]) -> None:
    base = next((a for a in arms if "auto" in a["arm"]), None)
    if base is None or base["calls"] == 0:
        print("  (no unpinned arm recorded calls; no baseline test)")
        return
    tests = []
    for a in arms:
        if a is base or a["calls"] == 0:
            continue
        z, p = two_proportion_z(a["hangs"], a["calls"], base["hangs"], base["calls"])
        tests.append([a, z, p, False])
    m = len(tests)
    for rank, t in enumerate(sorted(tests, key=lambda t: t[2])):
        t[3] = t[2] < 0.05 / (m - rank) if m - rank > 0 else False
    print(f"\n  vs {base['arm']} ({base['hang_rate'] * 100:.1f}% of "
          f"{base['calls']} calls), two-proportion z, Holm-corrected over {m}:")
    for a, z, p, sig in sorted(tests, key=lambda t: t[2]):
        mark = "SIGNIFICANT" if sig else "not significant"
        print(f"    {a['arm']:<22s} {a['hang_rate'] * 100:>5.1f}%  z={z:+6.2f}  p={p:.4f}  {mark}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
    data = collect(root)
    if not data:
        print(f"no ledgers under {root}")
        return 1

    models = sorted({m for m, _ in data})
    per_model: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        arms = []
        for (m, route), tasks in data.items():
            if m != model:
                continue
            pooled = [r for rows in tasks.values() for r in rows]
            arms.append(summarize_arm(route, pooled))
        per_model[model] = arms
        n_tasks = len({t for (m, _), ts in data.items() if m == model for t in ts})
        table(f"MODEL {model} — pooled over {n_tasks} task(s)", arms)
        holm(arms)

    if len(models) > 1:
        print("\n\nHOST RANKING ACROSS MODELS (by rate of calls that never completed)")
        print("A host whose rank moves is a host-model pair, not a host property.")
        header = f"{'route':<22s}" + "".join(f"{m:>26s}" for m in models)
        print(header)
        print("-" * len(header))
        routes = sorted({r for _, r in data})
        for route in routes:
            line = f"{route:<22s}"
            for model in models:
                a = next((x for x in per_model[model] if x["arm"] == route), None)
                if a is None or a["calls"] == 0:
                    line += f"{'--':>26s}"
                else:
                    rank = sorted(per_model[model], key=lambda x: x["hang_rate"]).index(a) + 1
                    cell = "#{} {:.1f}% n={}".format(rank, a["hang_rate"] * 100, a["calls"])
                    line += f"{cell:>26s}"
            print(line)

    print("\n\nPER-TASK BREAKDOWN (calls / hang% / cache%)")
    tasks = sorted({t for ts in data.values() for t in ts})
    for model in models:
        print(f"\n  {model}")
        header = f"    {'route':<22s}" + "".join(f"{t[:20]:>22s}" for t in tasks)
        print(header)
        for (m, route), ts in sorted(data.items()):
            if m != model:
                continue
            line = f"    {route:<22s}"
            for t in tasks:
                rows = ts.get(t)
                if not rows:
                    line += f"{'--':>22s}"
                    continue
                a = summarize_arm(route, rows)
                cache = "--" if a["cache_ratio"] is None else f"{a['cache_ratio'] * 100:.0f}%"
                cell = "{}/{:.0f}%/{}".format(a["calls"], a["hang_rate"] * 100, cache)
                line += f"{cell:>22s}"
            print(line)

    print(
        "\nCost is a LOWER BOUND on any arm with abandoned calls: those tokens were "
        "billed but their usage block never arrived. '--' in a cost column means the "
        "host reports no cost per call (Doubleword); run scripts/dw_cost_attribution.py "
        "for its billed total. cache% is the host's own reported cached/prompt ratio; "
        "read it beside $/1M ptok, not instead of it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
