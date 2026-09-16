"""Compare the harder-task trial with its replication, episode by episode.

Both runs execute the same frozen spec, so episode ids and spec_sha256 match and
the join is exact. Run offline:

    python scripts/compare_flex_runs.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "run1": ROOT / "artifacts/flex-hard-20260910/run/outcomes.jsonl",
    "run2": ROOT / "artifacts/flex-hard-rep2/outcomes.jsonl",
}
PLAN = ROOT / "artifacts/flex-hard-20260910/plan.json"


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {r["episode_id"]: r for r in rows}


def label(row):
    """Outcome class: a graded pass, a graded miss, or the service failure kind."""
    if row is None:
        return "pending"
    if row["status"] != "graded":
        return row["status"]
    return "pass" if row.get("success") is True else "miss"


def main():
    plan = json.loads(PLAN.read_text())
    episodes = {e["episode_id"]: e for e in plan["episodes"]}
    runs = {name: load(path) for name, path in RUNS.items() if path.exists()}
    if len(runs) < 2:
        raise SystemExit("Both runs must be present; found: " + ", ".join(runs))
    one, two = runs["run1"], runs["run2"]

    print("Spec agreement")
    shas = {name: {r["spec_sha256"] for r in rows.values()} for name, rows in runs.items()}
    print(f"  identical spec_sha256: {shas['run1'] == shas['run2']}")
    print(f"  episodes recorded: run1 {len(one)}, run2 {len(two)}, planned {len(episodes)}")

    print("\nOfficial passes per route and tier")
    print(f"  {'route':<15}{'tier':<10}{'run1':>6}{'run2':>6}   recorded")
    cells = defaultdict(lambda: {"run1": [0, 0], "run2": [0, 0]})
    for eid, meta in episodes.items():
        key = (meta["route"], meta["tier"])
        for name, rows in (("run1", one), ("run2", two)):
            row = rows.get(eid)
            if row is None:
                continue
            cells[key][name][1] += 1
            if row.get("success") is True:
                cells[key][name][0] += 1
    for key in sorted(cells):
        c = cells[key]
        print(
            f"  {key[0]:<15}{key[1]:<10}{c['run1'][0]:>6}{c['run2'][0]:>6}"
            f"   {c['run1'][1]}/{c['run2'][1]}"
        )

    print("\nEpisode-level agreement (same episode, both runs recorded)")
    pairs = [(eid, label(one.get(eid)), label(two.get(eid))) for eid in episodes if eid in one]
    pairs = [p for p in pairs if p[2] != "pending"]
    same_pairs = sum(1 for _, a, b in pairs if a == b)
    print(f"  identical outcome class: {same_pairs}/{len(pairs)}")
    graded = [(a, b) for _, a, b in pairs if a in ("pass", "miss") and b in ("pass", "miss")]
    agree = sum(1 for a, b in graded if a == b)
    print(f"  both graded cleanly: {agree}/{len(graded)} agree", end="")
    if graded:
        flips = Counter((a, b) for a, b in graded if a != b)
        print("  flips: " + (", ".join(f"{a}->{b} x{n}" for (a, b), n in flips.items()) or "none"))
    else:
        print()

    print("\nService reliability by route (non-graded outcomes)")
    print(f"  {'route':<15}{'run1':>18}{'run2':>18}")
    for route in sorted({m["route"] for m in episodes.values()}):
        counts = {}
        for name, rows in (("run1", one), ("run2", two)):
            bad = Counter(
                label(rows[eid])
                for eid, m in episodes.items()
                if m["route"] == route and eid in rows and label(rows[eid]) not in ("pass", "miss")
            )
            counts[name] = ", ".join(f"{k}:{v}" for k, v in sorted(bad.items())) or "clean"
        print(f"  {route:<15}{counts['run1']:>18}{counts['run2']:>18}")

    print("\nWithin-run trial consistency (two trials of the same task)")
    within = {}
    for name, rows in (("run1", one), ("run2", two)):
        by_cell = defaultdict(dict)
        for eid, m in episodes.items():
            if eid in rows:
                by_cell[(m["route"], m["tier"], m["task_id"])][m["trial"]] = label(rows[eid])
        full = [v for v in by_cell.values() if len(v) == 2]
        same = sum(1 for v in full if len(set(v.values())) == 1)
        clean = [v for v in full if all(x in ("pass", "miss") for x in v.values())]
        cagree = sum(1 for v in clean if len(set(v.values())) == 1)
        within[name] = {
            "pairs": len(full),
            "identical": same,
            "clean_pairs": len(clean),
            "clean_identical": cagree,
        }
        print(
            f"  {name}: {same}/{len(full)} trial pairs identical; "
            f"{cagree}/{len(clean)} identical among cleanly graded pairs"
        )

    out = ROOT / "artifacts/flex-hard-rep2"
    spend_path = out / "spend.json"
    spend = json.loads(spend_path.read_text()) if spend_path.exists() else []
    accounting = {
        "settled_inference_usd": sum(e["charged_or_reserved_usd"] for e in spend if e["settled"]),
        "unresolved_reserved_usd": sum(
            e["charged_or_reserved_usd"] for e in spend if not e["settled"]
        ),
        "storage_network_not_reconciled": True,
    }
    summary = {
        "spec_sha256_identical": shas["run1"] == shas["run2"],
        "recorded": {"run1": len(one), "run2": len(two), "planned": len(episodes)},
        "cells": {f"{r}/{t}": cells[(r, t)] for r, t in cells},
        "episode_agreement": {"identical": same_pairs, "compared": len(pairs)},
        "graded_agreement": {"agree": agree, "compared": len(graded)},
        "within_run_trial_consistency": within,
        "accounting_run2": accounting,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# Harder-task trial: replication comparison",
        "",
        f"Two independent executions of the same frozen spec "
        f"(`spec_sha256` identical: {shas['run1'] == shas['run2']}), so the {len(episodes)} "
        "episode ids match and the runs join exactly. Tasks 23, 30 and 41; two trials; "
        "five routes; both tiers.",
        "",
        "| Route | Tier | Passes run 1 | Passes run 2 | Recorded r1/r2 |",
        "|---|---|---:|---:|---|",
    ]
    for key in sorted(cells):
        c = cells[key]
        md.append(
            f"| {key[0]} | {key[1]} | {c['run1'][0]} | {c['run2'][0]} | "
            f"{c['run1'][1]}/{c['run2'][1]} |"
        )
    md += [
        "",
        f"Episode-level: {same_pairs}/{len(pairs)} episodes produced the same outcome class "
        f"across runs. Restricted to episodes graded cleanly in both runs, "
        f"{agree}/{len(graded)} agree.",
        "",
        f"Within a single run, repeated trials of the same task agreed "
        f"{within['run1']['clean_identical']}/{within['run1']['clean_pairs']} (run 1) and "
        f"{within['run2']['clean_identical']}/{within['run2']['clean_pairs']} (run 2) "
        "among cleanly graded pairs.",
        "",
        "Six attempts per cell cannot rank routes or establish tier parity. The per-cell "
        "differences between these two runs are the direct measure of that instability. "
        "Task 41's scoring ambiguity documented in the run 1 report applies to both runs.",
        "",
        "Run 2 accounting (includes simulator; unresolved charges retain reservations): "
        + json.dumps(accounting),
        "",
    ]
    (out / "comparison.md").write_text("\n".join(md))
    print(f"\nwrote {out}/comparison.md and comparison.json")


if __name__ == "__main__":
    main()
