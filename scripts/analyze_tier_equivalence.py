"""Frozen analysis for the tier equivalence study. Written before any data exists.

The decision rule, the estimand, the weighting and the interval method are all
fixed here so that no analysis choice can be made after seeing outcomes. Run:

    python scripts/analyze_tier_equivalence.py --out artifacts/tier-equivalence

Estimand: the task-weighted mean of the per-task difference in success rate,
flex minus standard. Each of the qualified tasks carries equal weight, because
the claim is about the task population rather than about the episodes we happened
to run, and trials of one task are not independent.

Primary outcome is intention to treat: every recorded attempt counts, and an
attempt that timed out, errored or hit a budget stop counts as not successful,
because a tier that fails to return is a tier that did not do the task. The
secondary outcome conditions on attempts that reached the grader, which isolates
answer quality from service delivery. Both are reported always, and which one is
primary is fixed here, not chosen later.

Equivalence is assessed by two one-sided tests at the 5% level, which is the
90% two-sided interval lying entirely inside the margin.
"""

import argparse
import json
import random
from pathlib import Path

MARGIN = 0.075  # Pre-declared, 2026-09-15. Not revisable after seeing outcomes.
CONFIDENCE = 0.90  # Two-sided interval matching a 5% TOST.
DRAWS = 10000
SEED = 20260915


def load(out_dir, plan_path):
    plan = json.loads(Path(plan_path).read_text())
    episodes = {e["episode_id"]: e for e in plan["episodes"]}
    rows = [
        json.loads(line)
        for line in (Path(out_dir) / "outcomes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return episodes, rows


def per_task(episodes, rows, graded_only):
    """Success counts per task and tier, as {task: {tier: [successes, attempts]}}."""
    tally = {}
    for row in rows:
        meta = episodes.get(row["episode_id"])
        if meta is None:
            continue
        graded = row["status"] == "graded"
        if graded_only and not graded:
            continue
        cell = tally.setdefault(meta["task_id"], {})
        counts = cell.setdefault(meta["tier"], [0, 0])
        counts[1] += 1
        if graded and row.get("success") is True:
            counts[0] += 1
    return tally


def differences(tally):
    """Per-task flex minus standard, for tasks with attempts recorded on both tiers."""
    paired, unpaired = {}, []
    for task, cell in tally.items():
        standard, flex = cell.get("standard"), cell.get("flex")
        if not standard or not flex or standard[1] == 0 or flex[1] == 0:
            unpaired.append(task)
            continue
        paired[task] = flex[0] / flex[1] - standard[0] / standard[1]
    return paired, sorted(unpaired)


def interval(values, draws=DRAWS, confidence=CONFIDENCE, seed=SEED):
    """Percentile bootstrap over tasks, the unit of independence."""
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (1 - confidence) / 2
    return means[int(tail * draws)], means[min(int((1 - tail) * draws), draws - 1)]


def verdict(low, high, margin=MARGIN):
    if low is None:
        return "no data"
    if low > -margin and high < margin:
        return "equivalent within the margin"
    if low >= margin or high <= -margin:
        return "difference at least as large as the margin"
    return "inconclusive at this sample size"


def analyse(episodes, rows, graded_only):
    tally = per_task(episodes, rows, graded_only)
    paired, unpaired = differences(tally)
    values = list(paired.values())
    point = sum(values) / len(values) if values else None
    low, high = interval(values)
    attempts = sum(c[1] for cell in tally.values() for c in cell.values())
    return {
        "tasks_paired": len(paired),
        "tasks_unpaired": unpaired,
        "attempts": attempts,
        "task_weighted_difference": point,
        "ci_low": low,
        "ci_high": high,
        "margin": MARGIN,
        "verdict": verdict(low, high),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/tier-equivalence")
    parser.add_argument("--plan", default=None)
    args = parser.parse_args()
    plan_path = args.plan or Path(args.out) / "plan.json"
    episodes, rows = load(args.out, plan_path)
    report = {
        "recorded": len(rows),
        "primary_intention_to_treat": analyse(episodes, rows, graded_only=False),
        "secondary_graded_only": analyse(episodes, rows, graded_only=True),
    }
    Path(args.out, "equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
