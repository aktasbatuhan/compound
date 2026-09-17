"""Frozen analysis for the tier equivalence study.

The numerical estimate is always available for diagnosis. A confirmatory
verdict is available only when the entire pre-declared study is present and
internally valid. A checkpoint or mixed collection must never turn a narrow
bootstrap interval into an equivalence claim.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

MARGIN = 0.075
CONFIDENCE = 0.90
DRAWS = 10000
SEED = 20260915
CONFIRMATORY_TASKS = 88
CONFIRMATORY_TRIALS = 7
TIERS = {"standard", "flex"}
KNOWN_STATUSES = {
    "graded", "provider_error", "timeout", "infrastructure_error", "budget_exhausted"
}
INVALIDATING_STATUSES = {"infrastructure_error", "budget_exhausted"}


def load(out_dir, plan_path):
    plan = json.loads(Path(plan_path).read_text())
    rows = [
        json.loads(line)
        for line in (Path(out_dir) / "outcomes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Retain the plan list for validation: converting to a mapping first used
    # to silently discard duplicate IDs.
    episodes = {e["episode_id"]: e for e in plan.get("episodes", []) if "episode_id" in e}
    return plan, episodes, rows


def confirmatory_issues(plan, rows):
    """Return reasons this collection cannot support a confirmatory verdict."""
    issues = []
    planned = plan.get("episodes")
    if not isinstance(planned, list):
        return ["plan has no episode list"]

    ids = [e.get("episode_id") for e in planned]
    duplicate_plan_ids = sorted(k for k, n in Counter(ids).items() if k is not None and n > 1)
    if None in ids or "" in ids:
        issues.append("plan contains an episode without an ID")
    if duplicate_plan_ids:
        issues.append(f"plan contains duplicate episode IDs: {duplicate_plan_ids}")
    if plan.get("episode_count") != len(planned):
        issues.append(
            f"plan episode_count is {plan.get('episode_count')!r}, "
            f"but the episode list has {len(planned)}"
        )

    required = {"episode_id", "suite", "route", "task_id", "trial", "tier"}
    malformed = [e.get("episode_id") for e in planned if not required <= set(e)]
    if malformed:
        issues.append(f"plan episodes are missing identity fields: {malformed}")
    valid = [e for e in planned if required <= set(e)]

    suites = {e["suite"] for e in valid}
    routes = {e["route"] for e in valid}
    tiers = {e["tier"] for e in valid}
    budgets = {e.get("budget_usd") for e in valid}
    if len(suites) != 1:
        issues.append(f"confirmatory analysis requires one suite, found {sorted(suites)}")
    if len(routes) != 1:
        issues.append(f"confirmatory analysis requires one route, found {sorted(routes)}")
    if tiers != TIERS:
        issues.append(f"plan tiers must be {sorted(TIERS)}, found {sorted(tiers)}")
    if len(budgets) != 1:
        issues.append(
            "tier-equivalence analysis requires one budget level; "
            f"found {sorted(budgets, key=lambda value: (value is None, repr(value)))}"
        )

    cells = defaultdict(list)
    task_trials = defaultdict(set)
    for episode in valid:
        if type(episode["trial"]) is not int or episode["trial"] < 0:
            issues.append(f"episode {episode['episode_id']} has an invalid trial")
            continue
        pair = (
            episode["suite"],
            episode["route"],
            episode.get("budget_usd"),
            episode["task_id"],
            episode["trial"],
        )
        cells[pair].append(episode["tier"])
        task_trials[
            (episode["suite"], episode["route"], episode.get("budget_usd"), episode["task_id"])
        ].add(episode["trial"])
    broken_pairs = sorted(
        str(key) for key, value in cells.items() if Counter(value) != Counter(TIERS)
    )
    if broken_pairs:
        issues.append(f"plan has incomplete or duplicate tier pairs: {broken_pairs}")
    if len(task_trials) != CONFIRMATORY_TASKS:
        issues.append(
            f"confirmatory design requires {CONFIRMATORY_TASKS} tasks, found {len(task_trials)}"
        )
    expected_trials = set(range(CONFIRMATORY_TRIALS))
    wrong_trials = sorted(
        str(key) for key, trials in task_trials.items() if trials != expected_trials
    )
    if wrong_trials:
        issues.append(
            f"each task requires trials 0..{CONFIRMATORY_TRIALS - 1}; mismatches: {wrong_trials}"
        )

    expected_ids = set(ids) - {None, ""}
    row_ids = [row.get("episode_id") for row in rows]
    duplicate_rows = sorted(k for k, n in Counter(row_ids).items() if k is not None and n > 1)
    unknown = sorted({eid for eid in row_ids if eid not in expected_ids}, key=str)
    missing = sorted(expected_ids - set(row_ids), key=str)
    if duplicate_rows:
        issues.append(f"duplicate outcomes: {duplicate_rows}")
    if unknown:
        issues.append(f"unknown outcomes: {unknown}")
    if missing:
        issues.append(f"missing {len(missing)} planned outcomes")
    if len(rows) != len(planned):
        issues.append(f"recorded {len(rows)} outcomes for {len(planned)} planned episodes")

    spec_hash = plan.get("spec_sha256")
    if not isinstance(spec_hash, str) or not spec_hash:
        issues.append("plan has no study specification fingerprint")
    mismatched_specs = sum(row.get("spec_sha256") != spec_hash for row in rows)
    if mismatched_specs:
        issues.append(f"{mismatched_specs} outcomes do not match the plan specification")
    unknown_statuses = sorted({row.get("status") for row in rows} - KNOWN_STATUSES, key=str)
    if unknown_statuses:
        issues.append(f"unknown outcome statuses: {unknown_statuses}")
    malformed_outcomes = []
    for row in rows:
        status = row.get("status")
        if status == "graded" and type(row.get("success")) is not bool:
            malformed_outcomes.append(row.get("episode_id"))
        elif status in KNOWN_STATUSES - {"graded"} and row.get("success") is not None:
            malformed_outcomes.append(row.get("episode_id"))
    if malformed_outcomes:
        issues.append(f"outcomes have status/success contradictions: {malformed_outcomes}")
    invalidating = Counter(
        row.get("status") for row in rows if row.get("status") in INVALIDATING_STATUSES
    )
    if invalidating:
        issues.append(f"study-invalidating outcomes: {dict(invalidating)}")
    return issues


def per_task(episodes, rows, graded_only):
    """Success counts by suite, route, budget, task and tier."""
    tally = {}
    for row in rows:
        meta = episodes.get(row.get("episode_id"))
        if meta is None or meta.get("tier") not in TIERS:
            continue
        graded = row.get("status") == "graded"
        if graded_only and not graded:
            continue
        task_key = (
            meta.get("suite"),
            meta.get("route"),
            meta.get("budget_usd"),
            meta.get("task_id"),
        )
        counts = tally.setdefault(task_key, {}).setdefault(meta["tier"], [0, 0])
        counts[1] += 1
        if graded and row.get("success") is True:
            counts[0] += 1
    return tally


def differences(tally):
    """Per-task flex minus standard for tasks observed on both tiers."""
    paired, unpaired = {}, []
    for task, cell in tally.items():
        standard, flex = cell.get("standard"), cell.get("flex")
        if not standard or not flex or standard[1] == 0 or flex[1] == 0:
            unpaired.append(task)
            continue
        paired[task] = flex[0] / flex[1] - standard[0] / standard[1]
    return paired, sorted(unpaired, key=str)


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


def statistical_verdict(low, high, margin=MARGIN):
    if low is None or high is None:
        return "no data"
    if low > -margin and high < margin:
        return "equivalent within the margin"
    if low >= margin or high <= -margin:
        return "difference at least as large as the margin"
    return "inconclusive at this sample size"


def verdict(low, high, margin=MARGIN):
    """Backward-compatible name for the ungated statistical assessment."""
    return statistical_verdict(low, high, margin)


def analyse(episodes, rows, graded_only, integrity_issues=None, confirmatory=True):
    """Compute descriptive estimates and gate any confirmatory conclusion."""
    tally = per_task(episodes, rows, graded_only)
    paired, unpaired = differences(tally)
    values = list(paired.values())
    point = sum(values) / len(values) if values else None
    low, high = interval(values)
    attempts = sum(c[1] for cell in tally.values() for c in cell.values())
    raw_verdict = statistical_verdict(low, high)

    blockers = list(integrity_issues or [])
    if len(paired) < CONFIRMATORY_TASKS:
        blockers.append(
            f"confirmatory analysis requires {CONFIRMATORY_TASKS} paired tasks, found {len(paired)}"
        )
    if unpaired:
        blockers.append(f"tasks lack an observed tier arm: {unpaired}")
    if values and (len(set(values)) < 2 or not all(math.isfinite(v) for v in values)):
        blockers.append("task bootstrap is degenerate and cannot quantify uncertainty")
    if low is not None and high is not None and low >= high:
        blockers.append("confidence interval has zero or negative width")
    blockers = list(dict.fromkeys(blockers))

    eligible = confirmatory and not blockers
    verdict = (
        "descriptive only"
        if not confirmatory
        else "invalid for confirmatory inference"
        if blockers
        else raw_verdict
    )
    return {
        "analysis_role": "confirmatory primary" if confirmatory else "descriptive secondary",
        "confirmatory_eligible": eligible,
        "confirmatory_blockers": blockers,
        "tasks_paired": len(paired),
        "tasks_unpaired": [list(task) for task in unpaired],
        "attempts": attempts,
        "task_weighted_difference": point,
        "ci_low": low,
        "ci_high": high,
        "margin": MARGIN,
        "descriptive_statistical_assessment": raw_verdict,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/tier-equivalence")
    parser.add_argument("--plan", default=None)
    args = parser.parse_args()
    plan_path = args.plan or Path(args.out) / "plan.json"
    plan, episodes, rows = load(args.out, plan_path)
    issues = confirmatory_issues(plan, rows)
    report = {
        "planned": len(plan.get("episodes", [])),
        "recorded": len(rows),
        "study_integrity": {"valid": not issues, "issues": issues},
        "primary_intention_to_treat": analyse(episodes, rows, False, issues, True),
        # The conditional quality view is useful but is not the registered claim.
        "secondary_graded_only": analyse(episodes, rows, True, issues, False),
    }
    Path(args.out, "equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
