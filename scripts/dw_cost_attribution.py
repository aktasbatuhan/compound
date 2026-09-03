#!/usr/bin/env python3
"""Attribute Doubleword's billed cost across the tiers a grid ran.

Doubleword does not report cost per call, so a ledger row from a Doubleword
route has ``cost_usd = None``. What it does report, through ``dw usage``, is the
**billed** total for a time window, broken down by model but not by tier. A grid
that runs the realtime and flex tiers of the same model therefore has one billed
number covering two arms.

This splits that number by each tier's share of the tokens our own ledger
recorded, which is the only split the data supports. It is an attribution, not a
measurement, and the output says so: the billed total is measured, the per-tier
figures are derived from it.

    python3 scripts/dw_cost_attribution.py artifacts/fswe-<stamp> --since 2026-09-03

Reads ledgers at ``<root>/<arm>/out/<model>/<task>/ledger/<route>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def ledger_tokens(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    """(model, route) -> token totals, for Doubleword routes only."""
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"prompt": 0, "completion": 0, "cached": 0, "calls": 0}
    )
    for ledger in sorted(root.glob("**/ledger/*.jsonl")):
        if "doubleword" not in ledger.stem:
            continue
        model = ledger.parts[-4]
        key = (model, ledger.stem)
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


def dw_usage(since: str, until: str | None) -> dict[str, dict]:
    """model -> billed usage, from the dw CLI. Billed, not derived."""
    cmd = ["dw", "usage", "--since", since, "--output", "json"]
    if until:
        cmd += ["--until", until]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"dw usage failed: {out.stderr.strip()[:300]}")
    payload = json.loads(out.stdout)
    return {row["model"]: row for row in payload.get("by_model", [])}


#: The grid's model label -> the id Doubleword bills under.
MODEL_IDS = {
    "glm53flash": "zai-org/GLM-5.3-Flash",
    "deepseekv4flash": "deepseek-ai/DeepSeek-V4-Flash-0731",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--since", required=True, help="ISO date, e.g. 2026-09-03")
    parser.add_argument("--until", default=None)
    args = parser.parse_args()

    totals = ledger_tokens(args.root)
    if not totals:
        print("no Doubleword ledgers found")
        return 1
    billed = dw_usage(args.since, args.until)

    print(f"Doubleword billed usage for {args.since}..{args.until or args.since} (measured):")
    for model_id, row in billed.items():
        print(
            f"  {model_id:36s} ${float(row['cost']):.4f}  "
            f"in={row['input_tokens']:,}  out={row['output_tokens']:,}  reqs={row['request_count']}"
        )

    print("\nPer-tier attribution by our ledger's prompt-token share (derived):")
    header = (
        f"{'model':<16s} {'route':<22s} {'calls':>6s} {'ptok':>12s} "
        f"{'share':>6s} {'cost$':>9s} {'$/1M ptok':>10s} {'cache%':>7s}"
    )
    print(header)
    print("-" * len(header))
    for model_label, model_id in MODEL_IDS.items():
        rows = {r: v for (m, r), v in totals.items() if m == model_label}
        if not rows or model_id not in billed:
            continue
        total_prompt = sum(v["prompt"] for v in rows.values()) or 1
        model_cost = float(billed[model_id]["cost"])
        for route, v in sorted(rows.items()):
            share = v["prompt"] / total_prompt
            cost = model_cost * share
            cache = (v["cached"] / v["prompt"] * 100) if v["prompt"] else 0.0
            per_m = cost / (v["prompt"] / 1e6) if v["prompt"] else 0.0
            print(
                f"{model_label:<16s} {route:<22s} {v['calls']:>6d} {v['prompt']:>12,d} "
                f"{share * 100:>5.1f}% {cost:>9.4f} {per_m:>10.4f} {cache:>6.1f}%"
            )

    print(
        "\nThe billed total is measured by Doubleword's own meters. The split "
        "between tiers is derived from the prompt tokens our ledger recorded, "
        "because dw usage reports by model and not by tier. The window may also "
        "include calls made outside this grid; check request counts against the "
        "ledger's call counts before quoting a per-tier figure."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
