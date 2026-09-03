#!/usr/bin/env python3
"""Measure Doubleword's realtime and flex rates for one model, in one sitting.

``scripts/dw_tier_cost.py`` can separate the two tiers, but only from a billing
window that covers a single model, because ``dw usage`` reports its
``estimated_realtime_cost`` per window and not per model. A grid that ran two
models through Doubleword on the same UTC day can never be split after the fact,
and the window truncates to the calendar day so there is no way to carve it
finer.

Rather than re-run a whole grid under that constraint, calibrate once: send a
small, known workload to each tier for one model, on a day whose Doubleword
traffic is only that model, and read the rates straight off the bill. Rates are a
property of the (model, tier) pair, not of the workload, so they then apply to
any run's measured tokens.

This refuses to run if the day already carries another model's Doubleword
traffic, since the result would be unattributable, and says which model is in
the way.

    python3 scripts/dw_tier_calibrate.py --model zai-org/GLM-5.3-Flash

Cost is a few cents: the default workload is 8 calls per tier.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compound.dw_usage import (  # noqa: E402
    FLEX_LABEL,
    REALTIME_LABEL,
    derive_tier_rates,
    parse_usage,
)
from compound.orproxy import serve_provider  # noqa: E402
from compound.providers_registry import parse_provider  # noqa: E402

TIERS = (("doubleword/realtime", REALTIME_LABEL), ("doubleword/flex", FLEX_LABEL))


def load_env(path: Path = Path(".env")) -> None:
    """Read keys without sourcing the file, which clobbers PATH in a shell."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def dw_usage_day(day: str) -> dict:
    argv = ["dw", "usage", "--since", day, "--until", day, "--output", "json"]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"`{' '.join(argv)}` failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def drive_tier(provider: str, wire_model: str, calls: int, filler_tokens: int) -> dict[str, int]:
    """Send ``calls`` requests through the pinning proxy. Returns token totals.

    Each call carries a distinct prefix so the prompt cache cannot serve it: a
    cached call is billed differently, and a calibration whose traffic is half
    cache hits measures a blend rather than the tier's rate.
    """
    spec = replace(parse_provider(provider), wire_model=wire_model)
    filler = "The quick brown fox jumps over the lazy dog. " * max(1, filler_tokens // 10)
    totals = {"prompt": 0, "completion": 0, "calls": 0}
    with serve_provider(spec) as base:
        for i in range(calls):
            body = {
                "model": "placeholder",
                "messages": [
                    {"role": "system", "content": f"Run {time.time_ns()}-{i}.\n{filler}"},
                    {"role": "user", "content": f"Reply with the number {i}."},
                ],
                "max_tokens": 32,
            }
            request = urllib.request.Request(
                base.rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer calib"},
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.load(response)
            except Exception as exc:  # noqa: BLE001 - one bad call should not end it
                print(f"    call {i}: ERROR {str(exc)[:90]}")
                continue
            usage = payload.get("usage") or {}
            totals["calls"] += 1
            totals["prompt"] += int(usage.get("prompt_tokens") or 0)
            totals["completion"] += int(usage.get("completion_tokens") or 0)
            print(f"    call {i}: prompt={usage.get('prompt_tokens')} "
                  f"completion={usage.get('completion_tokens')}")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model id as Doubleword bills it")
    parser.add_argument("--calls", type=int, default=8, help="calls per tier")
    parser.add_argument("--filler-tokens", type=int, default=2000)
    parser.add_argument("--day", default=None, help="UTC date, defaults to today")
    parser.add_argument(
        "--force", action="store_true",
        help="calibrate even though the day already carries another model",
    )
    args = parser.parse_args()
    load_env()

    day = args.day or dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    before = dw_usage_day(day)
    others = [r["model"] for r in before.get("by_model", []) if r.get("model") != args.model]
    if others and not args.force:
        print(f"REFUSING: Doubleword traffic on {day} already includes {', '.join(others)}.")
        print("dw usage reports estimated_realtime_cost per window and the window is a whole")
        print("calendar day, so a second model makes the tier rates unrecoverable. Calibrate")
        print("on a day whose Doubleword traffic is only this model.")
        return 1

    # Cache markers off: a cached call bills differently, and this measures a rate.
    os.environ["COMPOUND_DW_CACHE"] = "0"
    measured: dict[str, dict[str, int]] = {}
    for provider, label in TIERS:
        print(f"\n{provider}  {args.model}  {args.calls} calls")
        measured[label] = drive_tier(provider, args.model, args.calls, args.filler_tokens)

    print("\nwaiting 30s for the meters to settle")
    time.sleep(30)
    after = dw_usage_day(day)
    rows = after.get("by_model", [])
    if len(rows) != 1:
        print(f"the window now carries {len(rows)} models; cannot attribute. Rerun on a clean day.")
        return 1

    usage = parse_usage(after, args.model)
    rt = measured[REALTIME_LABEL]
    flex = measured[FLEX_LABEL]
    rt_tok = rt["prompt"] + rt["completion"]
    flex_tok = flex["prompt"] + flex["completion"]
    rates = derive_tier_rates(usage, realtime_tokens=rt_tok, flex_tokens=flex_tok)

    print(f"\nDoubleword rates for {args.model}, UTC day {day}")
    print(f"  billed ${usage.total_cost:.6f}, all-realtime estimate "
          f"${usage.estimated_realtime_cost:.6f}, {usage.total_tokens:,} metered tokens")
    print(f"\n  {'tier':<22}{'calls':>7}{'tokens':>10}{'$/1M tokens':>14}")
    print("  " + "-" * 53)
    for label, counts, tok in ((REALTIME_LABEL, rt, rt_tok), (FLEX_LABEL, flex, flex_tok)):
        print(f"  {label:<22}{counts['calls']:>7}{tok:>10,}{rates.get(label, 0.0):>14.4f}")
    base = rates.get(REALTIME_LABEL) or 0.0
    if base and rates.get(FLEX_LABEL) is not None:
        print(f"\n  flex is {rates[FLEX_LABEL] / base:.2f}x the realtime rate.")
    print("\nThe realtime rate is MEASURED (all-realtime estimate / metered tokens). The flex")
    print("rate is DERIVED: the remainder of the actual bill, split by our ledger's token")
    print("share, because dw usage reports tokens per model and not per tier. Apply these")
    print("rates to any run's measured tokens; they do not depend on this workload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
