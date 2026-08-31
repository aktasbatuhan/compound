"""Compare provider arms from a time-matched Harbor sweep.

Reads one call ledger per arm plus each arm's Harbor job results, and reports
the comparison with the uncertainty attached. Three things it deliberately does
not do:

* It does not pool arms that ran at different times. Provider congestion drifts
  over an evening, so a sequential grid confounds the host with the hour it ran
  in. Arms are only comparable when they ran concurrently, and the header says
  which run is which.
* It does not treat a self-reported ``cached_tokens`` field as truth. One host
  reported ~100% cached on a prompt it had never seen, so the cache column is
  printed beside the billed cost rather than instead of it.
* It does not count an abandoned call's cost as zero. Those tokens were billed
  and the response that would have reported them never arrived, so every cost
  figure here is a *lower bound* on the arm that abandoned calls, and the
  report says so rather than quietly understating the slowest host.

Usage:
    uv run python scripts/analyze_arms.py artifacts/tb4-par
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because arm rates here sit far from
    0.5 with modest n, where the normal interval misbehaves (and can run past
    0 or 1, which a rate cannot).
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test; returns (z, two-sided p).

    Used for "does this host hang more often than that one", which is a
    comparison of two rates over independent calls.
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # Two-sided p from the normal CDF via erfc, so no scipy dependency.
    return (z, math.erfc(abs(z) / math.sqrt(2)))


def load_arm(ledger: Path) -> list[dict[str, Any]]:
    rows = []
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_arm(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    hangs = sum(1 for r in rows if r.get("error") == "hang_timeout")
    abandoned = sum(1 for r in rows if r.get("abandoned"))
    errors = sum(1 for r in rows if r.get("status") not in (200, None))
    priced = [r for r in rows if r.get("cost_usd") is not None]
    cost = sum(r["cost_usd"] for r in priced)
    ptok = sum(r.get("prompt_tokens") or 0 for r in rows)
    ctok = sum(r.get("cached_tokens") or 0 for r in rows)
    lat = sorted(r["latency_ms"] / 1000 for r in rows if r.get("latency_ms") is not None)
    ok = [r for r in rows if not r.get("abandoned") and r.get("status") == 200]
    ok_lat = sorted(r["latency_ms"] / 1000 for r in ok if r.get("latency_ms") is not None)
    hosts: dict[str, int] = {}
    for r in rows:
        echo = r.get("provider_echo")
        if echo:
            hosts[echo] = hosts.get(echo, 0) + 1
    lo, hi = wilson_interval(hangs, n)
    return {
        "arm": name,
        "calls": n,
        "completed": len(ok),
        "hangs": hangs,
        "hang_rate": hangs / n if n else 0.0,
        "hang_ci": (lo, hi),
        "abandoned": abandoned,
        "errors": errors,
        "cost_usd": cost,
        "priced_calls": len(priced),
        "prompt_tokens": ptok,
        "cached_tokens": ctok,
        "cache_ratio": (ctok / ptok) if ptok else None,
        "cost_per_1k_prompt": (cost / (ptok / 1000)) if ptok else None,
        "p50_s": ok_lat[len(ok_lat) // 2] if ok_lat else None,
        "p90_s": ok_lat[int(len(ok_lat) * 0.9)] if ok_lat else None,
        "max_s": lat[-1] if lat else None,
        "hosts": hosts,
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/tb4-par")
    ledgers = sorted(root.glob("*/ledger/*.jsonl")) or sorted(root.glob("**/ledger/*.jsonl"))
    if not ledgers:
        print(f"no ledgers under {root}")
        return 1

    arms = []
    for led in ledgers:
        # The ledger is named for the route it recorded, which is the arm's
        # identity in both layouts. The containing directory is not: a
        # sequential sweep writes every route's ledger into one job directory,
        # so naming arms after it labels them all identically.
        arm = led.stem
        rows = load_arm(led)
        if rows:
            arms.append(summarize_arm(arm, rows))
    arms.sort(key=lambda a: a["arm"])

    concurrent = len({led.parent.parent.name for led in ledgers}) == len(ledgers)
    design = (
        "arms ran CONCURRENTLY: congestion is common to all"
        if concurrent
        else "WARNING: arms share one sweep directory, so they ran SEQUENTIALLY; "
        "differences below are confounded with time of day"
    )
    print(f"Arms from {root}\n{design}\n")
    head = (
        f"{'arm':<16s} {'calls':>6s} {'hang%':>7s} {'95% CI':>14s} {'aband':>6s} "
        f"{'cost$':>9s} {'$/1k ptok':>10s} {'cache%':>7s} {'p50s':>6s} {'p90s':>6s} {'hosts':>5s}"
    )
    print(head)
    print("-" * len(head))
    for a in arms:
        lo, hi = a["hang_ci"]
        cache = "—" if a["cache_ratio"] is None else f"{a['cache_ratio'] * 100:.1f}"
        cpk = "—" if a["cost_per_1k_prompt"] is None else f"{a['cost_per_1k_prompt']:.5f}"
        print(
            f"{a['arm']:<16s} {a['calls']:>6d} {a['hang_rate'] * 100:>6.1f}% "
            f"{f'{lo * 100:.1f}-{hi * 100:.1f}':>14s} {a['abandoned']:>6d} "
            f"{a['cost_usd']:>9.4f} {cpk:>10s} {cache:>7s} "
            f"{(a['p50_s'] or 0):>6.0f} {(a['p90_s'] or 0):>6.0f} {len(a['hosts']):>5d}"
        )

    base = next((a for a in arms if "auto" in a["arm"]), None)
    if base:
        print(f"\nHang rate vs {base['arm']} (two-proportion z-test):")
        for a in arms:
            if a is base:
                continue
            z, p = two_proportion_z(a["hangs"], a["calls"], base["hangs"], base["calls"])
            verdict = "significant" if p < 0.05 else "not significant"
            print(
                f"  {a['arm']:<16s} {a['hang_rate'] * 100:5.1f}% vs {base['hang_rate'] * 100:5.1f}%"
                f"   z={z:+6.2f}  p={p:.4f}  {verdict}"
            )

    print("\nUpstreams that answered, per arm:")
    for a in arms:
        spread = ", ".join(f"{h} {n}" for h, n in sorted(a["hosts"].items(), key=lambda kv: -kv[1]))
        print(f"  {a['arm']:<16s} {spread or '—'}")

    print(
        "\nCost figures are LOWER BOUNDS on any arm with abandoned calls: those "
        "tokens were billed but their usage block never arrived. cache% is the "
        "host's self-reported cached/prompt ratio, which at least one host "
        "populates unreliably; read it beside $/1k ptok, not instead of it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
