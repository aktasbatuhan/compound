#!/usr/bin/env python3
"""Per-tier Doubleword cost, separated properly rather than by token share.

The problem
-----------
Doubleword's inference API returns no per-call cost, so a Doubleword ledger row
has ``cost_usd = None``. Billing lives in ``dw usage``, which reports a window's
billed total broken down **by model but never by tier**. A grid that runs the
realtime and flex tiers of the same model therefore gets one number covering two
arms.

The obvious workaround is to split that number by each tier's share of the
tokens, and it is wrong in a way that is easy to miss: dividing a single total by
token share necessarily hands both tiers the *same* effective rate, so the output
says the tiers cost the same no matter what they actually cost. Doubleword's
published flex rate is below its realtime rate, so that is not a rounding
artifact, it is the entire question being assumed away.

Two things that do not work
---------------------------
* **Sub-day windows.** ``dw usage --since/--until`` documents ISO 8601 but
  truncates to the calendar day: measured on 2026-09-03, the windows
  ``00:00-02:00Z``, ``02:00-04:00Z`` and the whole day all returned an identical
  632 requests and $1.9908. So the tiers cannot be separated by running them at
  different times of the same day.
* **Per-request billing.** ``dw requests`` would give per-call rows, but it
  requires the ``RequestViewer`` role and returns
  ``Forbidden: Insufficient permissions to Read Requests`` on this account.

What does work
--------------
``dw usage`` also reports ``estimated_realtime_cost`` for the window: what those
same tokens would have cost billed entirely at the realtime tier. With a window
covering exactly one model, that gives two independent measured numbers over the
same traffic::

    E = estimated_realtime_cost = r_realtime * T_total
    C = total_cost              = r_realtime * T_realtime + r_flex * T_flex

so the realtime rate ``r_realtime = E / T_total`` is **fully measured**, with no
assumption at all, and the flex rate follows from the actual bill. The tiers may
still run concurrently; only the *models* have to be separated, one per UTC day,
so that the window's estimate is attributable to a single model.

The one assumption left, stated plainly: splitting ``T_total`` into its realtime
and flex parts uses the token share our own call ledger recorded, because
``dw usage`` reports tokens per model and not per tier. That share is measured
from every call we made; what it assumes is that our ledger's tokens are
proportional to Doubleword's meters within a model, which is far weaker than
assuming the two tiers cost the same.

Usage
-----
::

    python3 scripts/dw_tier_cost.py artifacts/fswe-<stamp> \\
        --model-day glm53flash=2026-09-03 \\
        --model-day deepseekv4flash=2026-09-04

Each ``--model-day`` names a UTC date whose Doubleword traffic involved only that
model. The script refuses to derive rates for a day that carried more than one
model, and says so, rather than emitting a number that cannot be trusted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compound.dw_usage import (  # noqa: E402
    FLEX_LABEL,
    REALTIME_LABEL,
    derive_tier_rates,
    parse_usage,
)

#: The grid's model label -> the id Doubleword bills under.
MODEL_IDS = {
    "glm53flash": "zai-org/GLM-5.3-Flash",
    "deepseekv4flash": "deepseek-ai/DeepSeek-V4-Flash-0731",
}

#: Ledger stem -> the tier label derive_tier_rates keys its output by.
TIER_OF_STEM = {
    "doubleword-realtime": REALTIME_LABEL,
    "doubleword-flex": FLEX_LABEL,
}


def ledger_tokens(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    """(model label, tier label) -> token totals, Doubleword routes only.

    Only ledgers under an arm's ``out/`` tree are read, so preserved evidence
    trees (a fully rate-limited host kept as ``out-429-evidence/``) cannot leak
    into a cost figure.
    """
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"prompt": 0, "completion": 0, "cached": 0, "calls": 0}
    )
    for ledger in sorted(root.glob("**/ledger/*.jsonl")):
        tier = TIER_OF_STEM.get(ledger.stem)
        if tier is None or ledger.parts[-5] != "out":
            continue
        key = (ledger.parts[-4], tier)
        for line in ledger.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            totals[key]["calls"] += 1
            totals[key]["prompt"] += row.get("prompt_tokens") or 0
            totals[key]["completion"] += row.get("completion_tokens") or 0
            totals[key]["cached"] += row.get("cached_tokens") or 0
    return totals


def dw_usage_day(day: str, dw_bin: str = "dw") -> dict:
    """Raw ``dw usage`` payload for one UTC calendar day."""
    argv = [dw_bin, "usage", "--since", day, "--until", day, "--output", "json"]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"`{' '.join(argv)}` failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="a grid artifacts root")
    parser.add_argument(
        "--model-day", action="append", default=[], metavar="LABEL=YYYY-MM-DD",
        help="a UTC day whose Doubleword traffic was only this model, repeatable",
    )
    parser.add_argument("--dw-bin", default="dw")
    args = parser.parse_args()

    if not args.model_day:
        parser.error("pass at least one --model-day LABEL=YYYY-MM-DD")

    tokens = ledger_tokens(args.root)
    if not tokens:
        print(f"no Doubleword ledgers under {args.root}")
        return 1

    print(f"Doubleword per-tier cost from {args.root}\n")
    exit_code = 0
    for pair in args.model_day:
        label, _, day = pair.partition("=")
        model_id = MODEL_IDS.get(label, label)
        payload = dw_usage_day(day, args.dw_bin)
        rows = payload.get("by_model", [])

        print(f"== {label} ({model_id})  UTC day {day}")
        print(f"   window billed total: ${float(payload.get('total_cost', 0)):.4f} "
              f"over {payload.get('total_request_count', 0)} requests, "
              f"{len(rows)} model(s)  [MEASURED]")

        if len(rows) != 1:
            names = ", ".join(r.get("model", "?") for r in rows) or "none"
            print(f"   REFUSING to derive tier rates: the day carried {len(rows)} models "
                  f"({names}).\n   dw usage reports estimated_realtime_cost per WINDOW, not per "
                  "model, so with\n   more than one model it cannot be attributed and the tiers "
                  "cannot be separated.\n   Re-run so this day's Doubleword traffic uses one "
                  "model only.\n")
            exit_code = 1
            continue

        usage = parse_usage(payload, model_id)
        rt = tokens.get((label, REALTIME_LABEL), {})
        flex = tokens.get((label, FLEX_LABEL), {})
        rt_tok = rt.get("prompt", 0) + rt.get("completion", 0)
        flex_tok = flex.get("prompt", 0) + flex.get("completion", 0)
        if rt_tok + flex_tok == 0:
            print("   no Doubleword tokens in the ledger for this model; skipping\n")
            exit_code = 1
            continue

        print(f"   billed ${usage.total_cost:.4f}, all-realtime estimate "
              f"${usage.estimated_realtime_cost:.4f}  [both MEASURED by dw]")
        print(f"   dw meters {usage.total_tokens:,} tokens; our ledger recorded "
              f"{rt_tok + flex_tok:,} across both tiers")
        if usage.request_count != rt.get("calls", 0) + flex.get("calls", 0):
            print(f"   NOTE: dw counted {usage.request_count} requests but our ledger has "
                  f"{rt.get('calls', 0) + flex.get('calls', 0)}; the day includes traffic "
                  "from outside this grid,\n         so these rates cover more than the grid's "
                  "calls.")

        rates = derive_tier_rates(usage, realtime_tokens=rt_tok, flex_tokens=flex_tok)
        header = (f"   {'tier':<22}{'calls':>7}{'tokens':>14}{'$/1M':>10}"
                  f"{'cost$':>10}{'cache%':>8}")
        print(header)
        print("   " + "-" * (len(header) - 3))
        for tier, counts, tok in ((REALTIME_LABEL, rt, rt_tok), (FLEX_LABEL, flex, flex_tok)):
            if tok == 0:
                continue
            rate = rates.get(tier, 0.0)
            cost = rate * tok / 1e6
            prompt = counts.get("prompt", 0)
            cache = (counts.get("cached", 0) / prompt * 100) if prompt else 0.0
            print(f"   {tier:<22}{counts.get('calls', 0):>7}{tok:>14,}"
                  f"{rate:>10.4f}{cost:>10.4f}{cache:>7.1f}%")
        if rt_tok and flex_tok:
            base = rates.get(REALTIME_LABEL) or 0.0
            if base:
                ratio = rates.get(FLEX_LABEL, 0.0) / base
                print(f"   flex is {ratio:.2f}x the realtime rate on this model.")
        print()

    print("Provenance: the billed total and the all-realtime estimate are MEASURED by "
          "Doubleword's\nown meters. The realtime rate is measured (estimate / total tokens). "
          "The flex rate is\nDERIVED: it takes the remainder of the actual bill, using our call "
          "ledger's token share to\nsplit the meter between tiers, because dw usage reports "
          "tokens per model and not per tier.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
