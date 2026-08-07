"""Turn a provider-sweep output directory into tables, transcripts, and charts.

Reads the per-host episode dumps a tau2 ``--providers`` sweep writes under
``<run>/<host>/episodes/*/results.json`` and emits, next to them under
``<run>/report/``:

    episodes.csv      one row per episode (host, task, trial, reward, tokens,
                      cost, latency, the upstream that actually served it)
    per_task.csv      per host per task: context size, trials, solved, rate
    summary.json      per host: accuracy, cost/task, latency, TPS, served-by
    transcripts.jsonl full ordered messages per episode (model output)
    charts.html       success-vs-context and cost-vs-quality, theme-aware

Cost is taken from OpenRouter's own per-call accounting
(``raw_data.usage.cost``) when present; for hosts that do not report it
(Doubleword), pass ``--prices label=in,out`` (USD per million tokens) to derive
it from token counts. The upstream echo (``raw_data.provider``) is surfaced so a
run can be checked: a pinned host should serve every episode itself.

CLI:
    python -m compound.bench_report artifacts/bench/tau2-sweep
    python -m compound.bench_report <run> --prices doubleword-flex=0.70,2.25
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Episode:
    host: str
    task: str
    trial: int
    reward: float
    solved: bool
    termination: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    gen_time_s: float
    served_by: list[str] = field(default_factory=list)
    service_tier_echo: list[Any] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


def _assistant_signals(messages: list[dict]) -> dict[str, Any]:
    """Pull per-episode token/cost/latency/served-by from assistant raw_data."""
    max_prompt = 0
    completion = 0
    cost = 0.0
    gen_time = 0.0
    served: list[str] = []
    tiers: list[Any] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        rd = m.get("raw_data") or {}
        usage = rd.get("usage") or m.get("usage") or {}
        max_prompt = max(max_prompt, int(usage.get("prompt_tokens") or 0))
        completion += int(usage.get("completion_tokens") or 0)
        cost += float(usage.get("cost") or m.get("cost") or 0.0)
        gen_time += float(m.get("generation_time_seconds") or 0.0)
        if rd.get("provider"):
            served.append(rd["provider"])
        if rd.get("service_tier") is not None:
            tiers.append(rd["service_tier"])
    return {
        "prompt_tokens": max_prompt,
        "completion_tokens": completion,
        "cost_usd": cost,
        "gen_time_s": gen_time,
        "served_by": sorted(set(served)),
        "service_tier_echo": sorted({str(t) for t in tiers}),
    }


def iter_episodes(run_dir: Path) -> list[Episode]:
    """Load every episode across every host subdirectory of a sweep run."""
    episodes: list[Episode] = []
    for results in sorted(run_dir.glob("*/episodes/*/results.json")):
        host = results.parents[2].name
        payload = json.loads(results.read_text())
        for sim in payload.get("simulations", []):
            info = sim.get("reward_info") or {}
            reward = float(info.get("reward") or 0.0)
            sig = _assistant_signals(sim.get("messages", []))
            episodes.append(
                Episode(
                    host=host,
                    task=str(sim.get("task_id")),
                    trial=int(sim.get("trial") or 0),
                    reward=reward,
                    solved=reward >= 1.0,
                    termination=str(sim.get("termination_reason")),
                    messages=sim.get("messages", []),
                    **sig,
                )
            )
    return episodes


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _fill_cost(ep: Episode, prices: dict[str, tuple[float, float]]) -> float:
    """Episode cost: OpenRouter's own figure, or derived from a declared price."""
    if ep.cost_usd > 0:
        return ep.cost_usd
    price = prices.get(ep.host)
    if not price:
        return 0.0
    return (ep.prompt_tokens * price[0] + ep.completion_tokens * price[1]) / 1e6


def summarize(
    episodes: list[Episode], prices: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    hosts = sorted({e.host for e in episodes})
    per_host: dict[str, Any] = {}
    for host in hosts:
        rows = [e for e in episodes if e.host == host]
        clean = [e for e in rows if e.termination != "infrastructure_error"]
        graded = clean or rows
        solved = sum(e.solved for e in graded)
        gen_times = [e.gen_time_s for e in graded if e.gen_time_s > 0]
        tps = [
            e.completion_tokens / e.gen_time_s
            for e in graded
            if e.gen_time_s > 0 and e.completion_tokens
        ]
        costs = [_fill_cost(e, prices) for e in graded]
        served = sorted({p for e in graded for p in e.served_by})
        per_host[host] = {
            "episodes": len(rows),
            "graded": len(graded),
            "infra_errors": len(rows) - len(clean),
            "accuracy": round(solved / len(graded), 4) if graded else None,
            "cost_per_task_usd": round(statistics.fmean(costs), 6) if costs else None,
            "median_latency_s": round(statistics.median(gen_times), 2) if gen_times else None,
            "p95_latency_s": round(_percentile(gen_times, 0.95), 2) if gen_times else None,
            "median_tps": round(statistics.median(tps), 1) if tps else None,
            "served_by": served,
            "service_tier_echo": sorted({t for e in graded for t in e.service_tier_echo}),
        }
    # pooled context-vs-success distribution across all hosts
    buckets = {"<6k": [0, 0], "6-8k": [0, 0], "8k+": [0, 0]}
    for e in episodes:
        if e.termination == "infrastructure_error":
            continue
        key = "<6k" if e.prompt_tokens < 6000 else ("6-8k" if e.prompt_tokens < 8000 else "8k+")
        buckets[key][0] += int(e.solved)
        buckets[key][1] += 1
    return {
        "hosts": per_host,
        "context_vs_success": {
            k: {"solved": s, "total": t, "rate": round(s / t, 3) if t else None}
            for k, (s, t) in buckets.items()
        },
    }


def write_csvs(episodes: list[Episode], prices: dict[str, tuple[float, float]], out: Path) -> None:
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["host", "task", "trial", "reward", "solved", "termination", "prompt_tokens",
             "completion_tokens", "cost_usd", "gen_time_s", "served_by"]
        )
        for e in episodes:
            w.writerow(
                [e.host, e.task, e.trial, e.reward, int(e.solved), e.termination,
                 e.prompt_tokens, e.completion_tokens, round(_fill_cost(e, prices), 6),
                 round(e.gen_time_s, 2), "|".join(e.served_by)]
            )
    # per task per host
    tasks = sorted({(e.host, e.task) for e in episodes})
    with (out / "per_task.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host", "task", "ctx_tokens", "trials", "solved", "success_rate"])
        for host, task in tasks:
            rows = [e for e in episodes if e.host == host and e.task == task
                    and e.termination != "infrastructure_error"]
            if not rows:
                continue
            ctx = max(e.prompt_tokens for e in rows)
            solved = sum(e.solved for e in rows)
            w.writerow([host, task, ctx, len(rows), solved, round(solved / len(rows), 3)])


def write_transcripts(episodes: list[Episode], out: Path) -> int:
    with (out / "transcripts.jsonl").open("w") as f:
        for e in episodes:
            flat = []
            for m in e.messages:
                rd = m.get("raw_data") or {}
                ch = (rd.get("choices") or [{}])[0]
                rec = {"role": m.get("role")}
                if m.get("content"):
                    rec["content"] = m["content"]
                tcs = [
                    {"name": (tc.get("function") or {}).get("name") or tc.get("name"),
                     "arguments": (tc.get("function") or {}).get("arguments")}
                    for tc in (m.get("tool_calls") or [])
                ]
                if tcs:
                    rec["tool_calls"] = tcs
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
                if rd.get("service_tier") is not None:
                    rec["service_tier_echo"] = rd["service_tier"]
                if rd.get("provider"):
                    rec["served_by"] = rd["provider"]
                flat.append(rec)
            f.write(json.dumps({
                "host": e.host, "task": e.task, "trial": e.trial, "reward": e.reward,
                "termination": e.termination, "context_tokens": e.prompt_tokens,
                "messages": flat,
            }, ensure_ascii=False) + "\n")
    return len(episodes)


def _parse_prices(items: list[str] | None) -> dict[str, tuple[float, float]]:
    prices: dict[str, tuple[float, float]] = {}
    for item in items or []:
        label, _, pair = item.partition("=")
        pin, _, pout = pair.partition(",")
        prices[label] = (float(pin), float(pout))
    return prices


def build_report(run_dir: Path, prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    episodes = iter_episodes(run_dir)
    if not episodes:
        raise SystemExit(f"error: no episodes found under {run_dir}/*/episodes/*/results.json")
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(episodes, prices)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csvs(episodes, prices, out)
    n = write_transcripts(episodes, out)
    from compound.bench_charts import render_charts

    render_charts(summary, out)
    summary["_episodes"] = len(episodes)
    summary["_transcripts"] = n
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a provider-sweep output directory")
    parser.add_argument(
        "--prices", action="append",
        help="declared price for a host lacking per-call cost: label=in,out (USD/M tokens)",
    )
    args = parser.parse_args()
    summary = build_report(args.run_dir, _parse_prices(args.prices))
    print(f"report -> {args.run_dir}/report ({summary['_episodes']} episodes)")
    for host, s in summary["hosts"].items():
        acc = f"{s['accuracy'] * 100:.0f}%" if s["accuracy"] is not None else "n/a"
        cost = f"${s['cost_per_task_usd']:.4f}" if s["cost_per_task_usd"] else "n/a"
        served = ",".join(s["served_by"]) or "?"
        print(f"  {host:22s} acc={acc:>5s} cost/task={cost:>9s} served_by={served}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
