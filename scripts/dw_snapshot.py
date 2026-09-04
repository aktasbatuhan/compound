#!/usr/bin/env python3
"""Exact Doubleword cost per run, by differencing the billing meter around it.

Doubleword's API returns no per-call cost, and ``dw usage`` reports a window's
billed total by model but never by tier. Earlier attempts to recover the
realtime/flex split all leaned on some assumption: splitting one total by token
share (which hands both tiers the same rate by construction), or deriving a rate
from ``estimated_realtime_cost`` (which needs a window covering a single model).

None of that is necessary. The day's totals are **live and cumulative**, so the
difference between two readings is exactly the spend between them. Measured
2026-09-03: 632 requests / $1.9908 in the morning, 1027 / $3.1445 later the same
day. Snapshot the meter around a run and its cost is measured, not derived.

Two properties make this precise rather than approximate:

* ``by_model`` breaks each snapshot down per model, so two models running
  concurrently do not contaminate each other's deltas. Only the *tier* has to be
  held constant across an interval, because nothing in the payload distinguishes
  tiers.
* ``estimated_realtime_cost`` is what the same tokens would have cost billed
  entirely at realtime. Over a **realtime-only** interval it must equal the
  billed delta, so every interval carries its own integrity check; over a
  **flex-only** interval the ratio *is* the flex discount, measured directly.

Two requirements. Nothing else may bill this Doubleword account during an
interval, or the delta absorbs it. And the meter books a call up to about a
minute after it completes, so an end snapshot taken the moment a pass finishes
misses its last calls and hands them to the next interval. Measured 2026-09-04:
80 calls sent, 68 booked at the end snapshot, the other 12 booked inside the
following interval. ``snap --expect N --since LABEL`` waits until the model's
request count has risen by N over the snapshot labelled LABEL before reading.

Usage::

    # before anything runs
    python3 scripts/dw_snapshot.py snap --log dw.jsonl --label baseline

    # around each task
    python3 scripts/dw_snapshot.py snap --log dw.jsonl --label 'realtime|glm53flash|libexpat:start'
    ... run the task ...
    python3 scripts/dw_snapshot.py snap --log dw.jsonl --label 'realtime|glm53flash|libexpat:end'

    python3 scripts/dw_snapshot.py report --log dw.jsonl

Labels are ``tier|model|task:start`` / ``:end``; the report pairs them and reads
each interval's delta from that model's own row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

#: Grid model label -> the id Doubleword bills under.
MODEL_IDS = {
    "glm53flash": "zai-org/GLM-5.3-Flash",
    "deepseekv4flash": "deepseek-ai/DeepSeek-V4-Flash-0731",
}


def dw_usage(day: str, dw_bin: str = "dw") -> dict:
    argv = [dw_bin, "usage", "--since", day, "--until", day, "--output", "json"]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"`{' '.join(argv)}` failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def cmd_snap(args: argparse.Namespace) -> int:
    now = dt.datetime.now(dt.UTC)
    day = args.day or now.strftime("%Y-%m-%d")
    payload = dw_usage(day, args.dw_bin)
    if args.expect:
        payload = wait_for_booking(args, day, payload)
    row = {"ts": now.isoformat(), "day": day, "label": args.label, "usage": payload}
    with Path(args.log).open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    by_model = {r["model"]: r for r in payload.get("by_model", [])}
    print(f"{args.label:52s} reqs={payload.get('total_request_count'):>6} "
          f"cost=${float(payload.get('total_cost', 0)):.6f} "
          f"models={len(by_model)}")
    return 0


def wait_for_booking(args: argparse.Namespace, day: str, payload: dict) -> dict:
    """Poll until the meter has booked ``--expect`` more requests than at ``--since``."""
    import time

    model_id = MODEL_IDS.get(args.model, args.model)
    base = None
    for line in Path(args.log).read_text().splitlines():
        row = json.loads(line)
        if row.get("label") == args.since:
            base = int(model_row(row["usage"], model_id).get("request_count", 0))
    if base is None:
        raise SystemExit(f"--since label {args.since!r} not found in {args.log}")
    target = base + args.expect
    deadline = time.time() + args.timeout
    while True:
        have = int(model_row(payload, model_id).get("request_count", 0))
        if have >= target:
            return payload
        if time.time() > deadline:
            print(f"  WARNING: meter shows {have} of expected {target} requests after "
                  f"{args.timeout}s; snapshotting anyway", file=sys.stderr)
            return payload
        time.sleep(10)
        payload = dw_usage(day, args.dw_bin)


def model_row(payload: dict, model_id: str) -> dict:
    for row in payload.get("by_model", []):
        if row.get("model") == model_id:
            return row
    return {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "request_count": 0}


def cmd_report(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    starts: dict[str, dict] = {}
    intervals: list[tuple[str, dict, dict]] = []
    for row in rows:
        label = row.get("label") or ""
        if label.endswith(":start"):
            starts[label[: -len(":start")]] = row
        elif label.endswith(":end"):
            key = label[: -len(":end")]
            if key in starts:
                intervals.append((key, starts.pop(key), row))
    if not intervals:
        print("no start/end pairs in the log; label snapshots 'tier|model|task:start' and ':end'")
        return 1

    print(f"Doubleword cost by interval, from {args.log}  [MEASURED, meter differenced]\n")
    header = (f"{'tier':<9}{'model':<17}{'task':<32}{'reqs':>6}{'in tok':>12}"
              f"{'out tok':>9}{'cost $':>10}{'$/1M in':>9}")
    print(header)
    print("-" * len(header))
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for key, first, last in intervals:
        parts = key.split("|")
        tier, model, task = (parts + ["?", "?", "?"])[:3]
        model_id = MODEL_IDS.get(model, model)
        a, b = model_row(first["usage"], model_id), model_row(last["usage"], model_id)
        d_cost = float(b.get("cost", 0)) - float(a.get("cost", 0))
        d_in = int(b.get("input_tokens", 0)) - int(a.get("input_tokens", 0))
        d_out = int(b.get("output_tokens", 0)) - int(a.get("output_tokens", 0))
        d_req = int(b.get("request_count", 0)) - int(a.get("request_count", 0))
        per_m = (d_cost / (d_in / 1e6)) if d_in else 0.0
        print(f"{tier:<9}{model:<17}{task[:31]:<32}{d_req:>6}{d_in:>12,}"
              f"{d_out:>9,}{d_cost:>10.6f}{per_m:>9.4f}")
        agg = totals.setdefault((tier, model), {"cost": 0.0, "in": 0, "out": 0, "req": 0})
        agg["cost"] += d_cost
        agg["in"] += d_in
        agg["out"] += d_out
        agg["req"] += d_req

    print(f"\n{'tier':<9}{'model':<17}{'reqs':>7}{'in tok':>14}{'cost $':>11}{'$/1M in':>10}")
    print("-" * 68)
    for (tier, model), agg in sorted(totals.items()):
        per_m = (agg["cost"] / (agg["in"] / 1e6)) if agg["in"] else 0.0
        print(f"{tier:<9}{model:<17}{agg['req']:>7}{agg['in']:>14,}"
              f"{agg['cost']:>11.6f}{per_m:>10.4f}")

    # Window-level integrity check. estimated_realtime_cost is reported per
    # window, not per model, so this is only meaningful across all models in an
    # interval: over realtime-only traffic it must match the billed delta.
    print("\nTier check (window-level, all models in the interval):")
    for key, first, last in intervals:
        tier = key.split("|")[0]
        d_cost = float(last["usage"].get("total_cost", 0)) - float(
            first["usage"].get("total_cost", 0)
        )
        d_est = float(last["usage"].get("estimated_realtime_cost", 0)) - float(
            first["usage"].get("estimated_realtime_cost", 0)
        )
        if d_est <= 0:
            continue
        ratio = d_cost / d_est
        if tier == "realtime":
            verdict = "ok" if abs(ratio - 1.0) < 0.02 else "MIXED? realtime should bill at 1.00"
        elif tier == "flex":
            verdict = f"flex discount {1 - ratio:.0%} off realtime"
        else:
            verdict = ""
        print(f"  {key:<52} billed/all-realtime = {ratio:.3f}  {verdict}")

    if starts:
        print(f"\nUnclosed intervals (no :end snapshot): {sorted(starts)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snap", help="record one reading of the billing meter")
    snap.add_argument("--log", required=True)
    snap.add_argument("--label", required=True, help="'tier|model|task:start' or ':end'")
    snap.add_argument("--day", default=None, help="UTC date, defaults to today")
    snap.add_argument("--dw-bin", default="dw")
    snap.add_argument("--expect", type=int, default=0,
                      help="wait until this many more requests are booked than at --since")
    snap.add_argument("--since", default=None, help="label of the interval's start snapshot")
    snap.add_argument("--model", default="deepseekv4flash", help="grid model label for --expect")
    snap.add_argument("--timeout", type=int, default=600, help="seconds to wait for booking")
    snap.set_defaults(func=cmd_snap)

    report = sub.add_parser("report", help="difference the readings into per-interval cost")
    report.add_argument("--log", required=True)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
