"""Aggregate a terminal-bench provider sweep into the shared report shape.

terminal-bench writes one ``results.json`` per host run (under
``<run>/<host>/<run-id>/results.json``) with a different schema than tau2: a
``results`` list of trials, each carrying ``is_resolved``, token counts, and
agent timing. This module folds those into the **same** summary/per-task shape
that :mod:`compound.bench_report` produces for tau2, so a coding sweep and a
tool-calling sweep render through the identical chart renderer and sit in one
package.

Cost is derived from declared ``--prices`` (terminal-bench does not report a
per-call cost); latency is the agent wall-clock per trial; "context" is the
trial's input-token count, which is what the success-vs-context chart plots.

CLI:
    python -m compound.tb_report artifacts/bench/terminal_bench-sweep \
        --prices deepinfra-fp4=0.14,0.28 --prices doubleword-flex=0.70,2.25
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def _iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _latency(trial: dict) -> float:
    start, end = _iso(trial.get("agent_started_at")), _iso(trial.get("agent_ended_at"))
    if start and end:
        return max(0.0, (end - start).total_seconds())
    return 0.0


def iter_trials(run_dir: Path) -> list[dict]:
    """One record per trial across every host run in the sweep directory."""
    trials: list[dict] = []
    for results in sorted(run_dir.glob("*/*/results.json")):
        host = results.parents[1].name
        payload = json.loads(results.read_text())
        for t in payload.get("results", []):
            trials.append(
                {
                    "host": host,
                    "task": t.get("task_id"),
                    "solved": bool(t.get("is_resolved")),
                    "failure_mode": t.get("failure_mode"),
                    "in_tokens": int(t.get("total_input_tokens") or 0),
                    "out_tokens": int(t.get("total_output_tokens") or 0),
                    "latency_s": _latency(t),
                }
            )
    return trials


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def summarize(trials: list[dict], prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    hosts = sorted({t["host"] for t in trials})
    per_host: dict[str, Any] = {}
    for host in hosts:
        rows = [t for t in trials if t["host"] == host]
        solved = sum(t["solved"] for t in rows)
        lat = [t["latency_s"] for t in rows if t["latency_s"] > 0]
        tps = [
            t["out_tokens"] / t["latency_s"]
            for t in rows
            if t["latency_s"] > 0 and t["out_tokens"]
        ]
        price = prices.get(host)
        costs = [
            (t["in_tokens"] * price[0] + t["out_tokens"] * price[1]) / 1e6
            for t in rows
        ] if price else []
        fails = {}
        for t in rows:
            if not t["solved"] and t["failure_mode"] not in (None, "unset"):
                fails[t["failure_mode"]] = fails.get(t["failure_mode"], 0) + 1
        # unknown_agent_error is the harness's label when the agent aborted on
        # an API-level failure (rate-limit, capability 4xx) — the provider,
        # not the model, killed the episode. Approximate but trace-verified
        # on real sweeps; parse/timeout stay attributed to the model run.
        infra = sum(
            1 for t_ in rows
            if not t_["solved"] and t_["failure_mode"] == "unknown_agent_error"
        )
        per_host[host] = {
            "episodes": len(rows),
            "graded": len(rows),
            "infra_errors": infra,
            "accuracy": round(solved / len(rows), 4) if rows else None,
            "cost_per_task_usd": round(statistics.fmean(costs), 6) if costs else None,
            "median_latency_s": round(statistics.median(lat), 2) if lat else None,
            "p95_latency_s": round(_percentile(lat, 0.95), 2) if lat else None,
            "median_tps": round(statistics.median(tps), 1) if tps else None,
            "served_by": [],
            "service_tier_echo": [],
            "failure_modes": fails,
        }
    return {"hosts": per_host}


def write_csvs(trials: list[dict], prices: dict[str, tuple[float, float]], out: Path) -> None:
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["host", "task", "solved", "failure_mode", "in_tokens", "out_tokens", "latency_s"]
        )
        for t in trials:
            w.writerow([t["host"], t["task"], int(t["solved"]), t["failure_mode"],
                        t["in_tokens"], t["out_tokens"], round(t["latency_s"], 1)])
    # per task per host, in the shape bench_charts expects (ctx_tokens + success_rate)
    tasks = sorted({(t["host"], t["task"]) for t in trials})
    with (out / "per_task.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host", "task", "ctx_tokens", "trials", "solved", "success_rate"])
        for host, task in tasks:
            rows = [t for t in trials if t["host"] == host and t["task"] == task]
            ctx = max((t["in_tokens"] for t in rows), default=0)
            solved = sum(t["solved"] for t in rows)
            w.writerow([host, task, ctx, len(rows), solved, round(solved / len(rows), 3)])


def _parse_prices(items: list[str] | None) -> dict[str, tuple[float, float]]:
    prices: dict[str, tuple[float, float]] = {}
    for item in items or []:
        label, _, pair = item.partition("=")
        pin, _, pout = pair.partition(",")
        prices[label] = (float(pin), float(pout))
    return prices


def build_report(run_dir: Path, prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    trials = iter_trials(run_dir)
    if not trials:
        raise SystemExit(f"error: no terminal-bench results under {run_dir}/*/*/results.json")
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(trials, prices)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csvs(trials, prices, out)
    from compound.bench_charts import render_charts

    render_charts(summary, out)
    summary["_trials"] = len(trials)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prices", action="append",
                        help="host declared price: label=in,out (USD/M tokens)")
    args = parser.parse_args()
    summary = build_report(args.run_dir, _parse_prices(args.prices))
    print(f"report -> {args.run_dir}/report ({summary['_trials']} trials)")
    for host, s in summary["hosts"].items():
        acc = f"{s['accuracy'] * 100:.0f}%" if s["accuracy"] is not None else "n/a"
        lat = f"{s['median_latency_s']}s" if s["median_latency_s"] else "n/a"
        print(f"  {host:22s} solved={acc:>5s} median_latency={lat:>7s} tasks={s['episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
