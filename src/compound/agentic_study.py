"""Offline planning and outcome accounting for paired agentic serving studies.

An HTTP success is never a task success. Only a completed benchmark grade
can set ``success``. Missing episodes stay in the planned denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan(spec: dict) -> dict:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported study schema")
    trials = spec["trials"]
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    ids = [m["id"] for m in spec["models"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("model route IDs must be nonempty and unique")
    budgets = spec.get("budget_levels_usd")
    if budgets is not None:
        if not isinstance(budgets, list) or not budgets or len(set(budgets)) != len(budgets):
            raise ValueError("budget levels must be a nonempty unique list")
        for budget in budgets:
            _number(budget, "budget level")
            if budget <= 0:
                raise ValueError("budget level must be positive")
        if spec.get("controls", {}).get("budget_output_policy", "fixed") != "fixed":
            raise ValueError("budget curves require the fixed output policy")
    episodes = []
    for suite, source in spec["sources"].items():
        task_ids = [t["id"] for t in source["tasks"]]
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("task IDs must be nonempty and unique within a suite")
        for task in source["tasks"]:
            for trial in range(trials):
                for model in spec["models"]:
                    levels = budgets if budgets is not None else [model.get("episode_budget_usd")]
                    for budget in levels:
                        for tier in ("standard", "flex"):
                            identity = [suite, task["id"], trial, model["id"], tier]
                            if budgets is not None:
                                identity.append(budget)
                            episodes.append(
                                {
                                    "episode_id": fingerprint(identity)[:24],
                                    "suite": suite,
                                    "task_id": task["id"],
                                    "trial": trial,
                                    "route": model["id"],
                                    "tier": tier,
                                    **({"budget_usd": budget} if budget is not None else {}),
                                }
                            )
    rng = random.Random(spec["selection_seed"])
    if spec.get("controls", {}).get("pair_order"):
        pairs = [episodes[i : i + 2] for i in range(0, len(episodes), 2)]
        rng.shuffle(pairs)
        for pair in pairs:
            rng.shuffle(pair)
        episodes = [episode for pair in pairs for episode in pair]
    else:
        rng.shuffle(episodes)
    return {
        "schema_version": 1,
        "spec_sha256": fingerprint(spec),
        "episode_count": len(episodes),
        "episodes": episodes,
    }


def verify_sources(spec: dict) -> list[str]:
    """No imports, subprocesses, credentials, or network access."""
    errors = []
    for suite, source in spec["sources"].items():
        path = Path(source["data_path"])
        if not path.is_file():
            errors.append(f"{suite}: missing {path}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["data_sha256"]:
            errors.append(f"{suite}: dataset hash mismatch")
    return errors


def finance_success(metrics: dict) -> bool:
    """Predeclared strict composite; retain upstream component scores as well."""
    if metrics.get("parse_success") is not True:
        return False
    if metrics.get("expected_inconsistency") is True:
        names = (
            "inconsistency_flag_matches",
            "inconsistency_code_matches",
            "inconsistency_empty_answer",
        )
    else:
        names = (
            "final_balance_sheet_and_journal_entries_match",
            "inconsistency_flag_matches",
            "inconsistency_code_matches",
        )
    return all(metrics.get(name) is True for name in names)


def wilson(successes: int, n: int) -> list[float] | None:
    if not n:
        return None
    p, z = successes / n, 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0, center - radius), min(1, center + radius)]


def _number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def summarize(spec: dict, outcomes: list[dict]) -> dict:
    study = plan(spec)
    expected = {r["episode_id"]: r for r in study["episodes"]}
    actual = {}
    for row in outcomes:
        eid = row["episode_id"]
        if eid not in expected or eid in actual:
            raise ValueError(f"unknown or duplicate episode: {eid}")
        if row.get("spec_sha256") != study["spec_sha256"]:
            raise ValueError("outcome belongs to a different study specification")
        status = row["status"]
        if status not in {
            "graded",
            "provider_error",
            "timeout",
            "infrastructure_error",
            "budget_exhausted",
        }:
            raise ValueError("unknown episode status")
        if status == "graded":
            if type(row.get("success")) is not bool:
                raise ValueError("graded episodes require an explicit boolean success")
            if not row.get("grader"):
                raise ValueError("graded episodes require grader provenance")
        elif row.get("success") is not None:
            raise ValueError("ungraded episodes cannot claim task success or failure")
        _number(row["duration_s"], "duration_s")
        for key in ("agent_cost_usd", "auxiliary_cost_usd", "sandbox_cost_usd"):
            if row.get(key) is not None:
                _number(row[key], key)
        actual[eid] = row

    groups = {}
    for episode in expected.values():
        key = (episode["suite"], episode["route"], episode["tier"], episode.get("budget_usd"))
        groups.setdefault(key, []).append(actual.get(episode["episode_id"]))
    summaries = []
    for (suite, route, tier, budget), rows in sorted(groups.items()):
        present = [r for r in rows if r is not None]
        graded = [r for r in present if r["status"] == "graded"]
        successes = [r for r in graded if r["success"]]
        n, good = len(rows), len(successes)
        costs = {}
        for field in ("agent_cost_usd", "auxiliary_cost_usd", "sandbox_cost_usd"):
            known = [r[field] for r in present if r.get(field) is not None]
            costs[field] = {
                "known_subtotal": sum(known),
                "coverage": len(known),
                "complete": len(known) == n,
                "per_success": sum(known) / good if len(known) == n and good else None,
            }
        complete = len(present) == n and all(r["status"] != "infrastructure_error" for r in present)
        summaries.append(
            {
                "suite": suite,
                "route": route,
                "tier": tier,
                "budget_usd": budget,
                "planned": n,
                "recorded": len(present),
                "graded": len(graded),
                "successes": good,
                "status_counts": dict(Counter(r["status"] for r in present)),
                "pending": n - len(present),
                "complete": complete,
                "success_rate": good / n if complete else None,
                "observed_success_fraction_of_planned": good / n,
                "success_rate_wilson95": (
                    wilson(good, n) if complete and spec["trials"] == 1 else None
                ),
                "uncertainty_note": (
                    "Repeated trials require task-clustered uncertainty; no episode-level interval."
                    if spec["trials"] > 1
                    else None
                ),
                "graded_success_rate": good / len(graded) if graded else None,
                "success_by_deadline": {
                    str(d): sum(r["duration_s"] <= d for r in successes) / n if complete else None
                    for d in spec["deadlines_s"]
                },
                "costs": costs,
            }
        )
    return {
        "schema_version": 1,
        "tier_evidence_policy": spec.get("controls", {}).get("tier_evidence_policy", "verified"),
        "spec_sha256": study["spec_sha256"],
        "episode_count": len(expected),
        "recorded": len(actual),
        "groups": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "report"))
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/flex-agentic/pilot.json"))
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.command == "plan":
        payload = plan(spec)
        payload["source_errors"] = verify_sources(spec)
    else:
        if args.outcomes is None:
            parser.error("report requires --outcomes")
        rows = [json.loads(s) for s in args.outcomes.read_text().splitlines() if s.strip()]
        payload = summarize(spec, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"{payload['episode_count']} planned episodes; wrote {args.out}")


if __name__ == "__main__":
    main()
