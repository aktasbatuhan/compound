#!/usr/bin/env python3
"""Compare serving hosts on speed, cost, cache, reliability, and agreement.

Reads the ``results.jsonl`` written by ``compound-bench serving`` and reports the
five things a vendor latency benchmark leaves out.

**Speed** is the part everyone publishes: TTFT and decode rate, per profile.
Reported here at p50 and p90, with the sample size next to them, because a p95
over ten runs is the second-worst sample rather than an estimate of anything.

**Reliability is a column, not a filter.** A host that returns 429 on the long
prompts and succeeds on the short ones will look fast if its failures are
dropped before the percentiles are taken. Failures are counted here and shown
with a Wilson interval, and the latency columns say how many calls they cover.

**Cost is per profile.** A host's effective rate moves with context length,
because what gets cached moves with context length, so one $/1M figure for a
host is a rate card rather than a bill. Hosts that return a per-call cost
(OpenRouter) are measured; hosts that return none (OpenAI, Anthropic, Telnyx)
are priced from the rate cards in ``compound.yaml`` and shown with a ``~``
prefix, because a derived number and a billed number must never sit in one
column looking alike. Doubleword's cost comes from its billing meter, see
``scripts/dw_snapshot.py``.

**Cache is the cold/warm delta.** Cold cells carry a per-call nonce so nothing
can be served warm; warm cells repeat a byte-identical prompt. The difference
between them is what the host's prompt cache is actually worth on this shape.

**Agreement is the one nobody publishes**, and it has to be done in two steps
or it is worthless. At temperature 0 the decode is nominally deterministic, so
two hosts serving the same weights on the same bytes should emit the same
tokens. Before comparing hosts to each other, though, ask whether a host can
reproduce its *own* output: batched serving makes floating-point reduction order
depend on who else is in the batch, and a host that varies run to run sets the
floor for what any cross-host difference can mean. So step 1 measures
self-consistency, step 2 compares only the hosts that passed it, and reports the
character offset where they first split.

Both steps use warm cells only. A cold cell carries a per-call nonce, so its
prompts differ by construction and "divergence" there is just the nonce.

A caveat that belongs in any writeup of the last section: divergence shows the
hosts are not bit-identical. It does not by itself say which one is right, and it
does not prove quantization specifically. Batching, kernel choice, speculative
decoding and attention implementation all produce it too.

    python3 scripts/analyze_serving.py artifacts/serving/results.jsonl
    python3 scripts/analyze_serving.py results.jsonl --rates compound.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def percentile(values: list[float], p: float) -> float | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return float(xs[lo]) if lo == hi else float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def wilson(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson interval, which stays sane at 0 successes unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ok(row: dict[str, Any]) -> bool:
    return row.get("status") == 200 and not row.get("error")


def load_rates(path: Path | None) -> dict[str, Any]:
    """The ``serving_rates_usd_per_million_tokens`` block, or ``{}``."""
    if path is None or not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("rates need pyyaml; derived cost skipped")
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("serving_rates_usd_per_million_tokens") or {}


def speed_table(rows: list[dict[str, Any]], rates: dict[str, Any]) -> None:
    from compound.serving_metrics import derived_cost_usd

    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r.get("shape", "?"), r.get("cache_mode", "?"), r.get("route", "?"))].append(r)

    for (shape, cmode) in sorted({(s, c) for s, c, _ in cells}):
        print(f"\n{shape}  [{cmode}]")
        header = (
            f"  {'route':<22}{'n':>5}{'fail%':>7}{'95% CI':>14}"
            f"{'ttft50':>8}{'ttft90':>8}{'dec50':>8}{'tot50':>8}"
            f"{'cache%':>8}{'$/1M in':>9}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for route in sorted({rt for s, c, rt in cells if s == shape and c == cmode}):
            rs = cells[(shape, cmode, route)]
            good = [r for r in rs if ok(r)]
            fails = len(rs) - len(good)
            lo, hi = wilson(fails, len(rs))
            ttft50 = percentile([r.get("ttft_s") for r in good], 50)
            ttft90 = percentile([r.get("ttft_s") for r in good], 90)
            dec50 = percentile([r.get("decode_tps") for r in good], 50)
            tot50 = percentile([r.get("total_s") for r in good], 50)
            prompt = sum(r.get("prompt_tokens") or 0 for r in good)
            cached = sum(r.get("cached_tokens") or 0 for r in good)
            costs = [r["cost_usd"] for r in good if r.get("cost_usd") is not None]
            cache_pct = (cached / prompt * 100) if prompt else None
            per_m = (sum(costs) / (prompt / 1e6)) if costs and prompt else None
            cost_col = _f(per_m, 4)
            if per_m is None and prompt:
                derived = [derived_cost_usd(r, rates) for r in good]
                derived = [d for d in derived if d is not None]
                if derived:
                    cost_col = "~" + _f(sum(derived) / (prompt / 1e6), 4)
            print(
                f"  {route:<22}{len(rs):>5}{fails / len(rs) * 100:>6.0f}%"
                f"{f'{lo * 100:.0f}-{hi * 100:.0f}':>14}"
                f"{_f(ttft50):>8}{_f(ttft90):>8}{_f(dec50, 0):>8}{_f(tot50):>8}"
                f"{_f(cache_pct, 0):>8}{cost_col:>9}"
            )


def _f(v: float | None, nd: int = 2) -> str:
    return "--" if v is None else f"{v:.{nd}f}"


def agreement(rows: list[dict[str, Any]]) -> None:
    """Self-consistency per host, then which hosts agree with each other."""
    # A route recorded with temperature None could not be pinned (Anthropic's
    # current models reject sampling parameters). It is set aside here, not
    # counted as a host that failed to reproduce itself.
    unpinnable = sorted({str(r.get("route")) for r in rows if r.get("temperature") is None})
    rows = [r for r in rows if r.get("temperature") is not None]
    temps = {r.get("temperature") for r in rows}
    if temps - {0}:
        print(f"\n\nAGREEMENT: skipped, this run used temperature {sorted(temps)}.")
        print("Sampling noise makes divergence uninformative above temperature 0.")
        return

    print("\n\nAGREEMENT AT TEMPERATURE 0")
    print("Same weights, same bytes, deterministic decode: hosts should match.")
    if unpinnable:
        print(f"  Not compared, temperature cannot be pinned: {', '.join(unpinnable)}")

    # Only warm cells can be compared. A cold cell carries a fresh per-call
    # nonce, so its prompts differ by construction and any "divergence" there is
    # just the nonce doing its job.
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if ok(r) and r.get("text_sha256") and r.get("cache_mode") == "warm":
            by_cell[(r.get("shape", "?"), r.get("route", "?"))].append(r)
    if not by_cell:
        print("\n  No warm cells: agreement needs byte-identical prompts, so run")
        print("  --cache-mode warm (or both). Cold cells nonce every prompt.")
        return

    print("\n  Step 1: does a host agree with ITSELF? (warm cells, identical bytes)")
    print("  This is the floor. A host that cannot reproduce its own output makes")
    print("  cross-host comparison uninterpretable, so it has to be measured first.")
    print(f"  {'shape':<16}{'route':<22}{'reps':>5}{'distinct':>9}  verdict")
    print("  " + "-" * 70)
    self_consistent: set[tuple[str, str]] = set()
    for (shape, route), rs in sorted(by_cell.items()):
        distinct = len({r["text_sha256"] for r in rs})
        if distinct == 1 and len(rs) > 1:
            self_consistent.add((shape, route))
            verdict = "deterministic"
        elif len(rs) < 2:
            verdict = "only 1 rep, cannot tell"
        else:
            verdict = f"NON-DETERMINISTIC ({distinct} outputs in {len(rs)} reps)"
        print(f"  {shape:<16}{route:<22}{len(rs):>5}{distinct:>9}  {verdict}")

    print("\n  Step 2: do hosts agree with EACH OTHER?")
    print("  Only hosts that passed step 1 are compared: between two hosts that")
    print("  each vary run to run, a mismatch says nothing about the hosts.")
    for shape in sorted({s for s, _ in by_cell}):
        texts: dict[str, list[str]] = defaultdict(list)
        for (sh, route), rs in by_cell.items():
            if sh != shape or (sh, route) not in self_consistent:
                continue
            for r in rs:
                texts[route].append(r.get("text") or "")
        if len(texts) < 2:
            n_self = len({rt for sh, rt in self_consistent if sh == shape})
            print(f"\n  {shape}: {n_self} host(s) reproduce their own output; "
                  "need 2+ to compare.")
            continue
        # The reference is the most common first-response across hosts, so no
        # single host is privileged by being listed first.
        firsts = [v[0] for v in texts.values() if v]
        if not firsts:
            continue
        reference = Counter(firsts).most_common(1)[0][0]
        agree = sum(1 for f in firsts if f == reference)
        print(f"\n  {shape}: {agree} of {len(firsts)} hosts match the modal output")
        print(f"    {'route':<22}{'matches':>9}{'first split':>13}")
        for route in sorted(texts):
            sample = texts[route][0]
            if sample == reference:
                print(f"    {route:<22}{'yes':>9}{'-':>13}")
            else:
                idx = _first_diff(reference, sample)
                print(f"    {route:<22}{'no':>9}{idx:>13,}")


def _first_diff(a: str, b: str) -> int:
    """Character offset where two strings first differ."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="a results.jsonl from compound-bench serving")
    parser.add_argument("--no-agreement", action="store_true")
    parser.add_argument(
        "--rates",
        type=Path,
        default=Path("compound.yaml"),
        help="config carrying serving_rates_usd_per_million_tokens, for hosts "
        "that bill no per-call cost (shown as ~derived)",
    )
    args = parser.parse_args()

    rows = load(args.results)
    if not rows:
        print(f"no records in {args.results}")
        return 1
    routes = sorted({r.get("route") for r in rows})
    print(f"{len(rows)} calls, {len(routes)} routes: {', '.join(map(str, routes))}")
    speed_table(rows, load_rates(args.rates))
    if not args.no_agreement:
        agreement(rows)
    print(
        "\nProvenance: latency and token counts are measured per call from the "
        "stream.\nCost without a ~ is the provider-reported figure (OpenRouter). "
        "A ~ marks cost derived\nfrom the rate cards in compound.yaml (OpenAI, "
        "Anthropic, Telnyx); Doubleword's comes\nfrom its billing meter, see "
        "scripts/dw_snapshot.py.\n"
        "Divergence shows hosts are not bit-identical. It does not say which is "
        "correct, and\nit is not by itself evidence of quantization: batching, kernels "
        "and speculative\ndecoding produce it too."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
