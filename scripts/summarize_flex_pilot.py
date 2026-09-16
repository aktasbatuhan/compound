"""Offline paired results; only compare task IDs attempted in both tiers."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

from compound.agentic_gateway import ProviderResponseError, check_response_errors
from compound.agentic_study import plan, summarize


def analyze(spec, directory):
    episodes = {x["episode_id"]: x for x in plan(spec)["episodes"]}
    rows = [json.loads(x) for x in (directory / "outcomes.jsonl").read_text().splitlines()]
    calls = [json.loads(x) for x in (directory / "calls.jsonl").read_text().splitlines()]
    spend = json.loads((directory / "spend.json").read_text())
    normalized = []
    corrections = []
    for row in rows:
        row = dict(row)
        eid = row["episode_id"]
        episode = episodes[eid]
        for role, field in (("agent", "agent_cost_usd"), ("auxiliary", "auxiliary_cost_usd")):
            selected = [c for c in calls if c["episode_id"] == eid and c["role"] == role]
            pending = any(
                s["episode_id"] == eid and s["role"] == role and not s["settled"] for s in spend
            )
            if (
                (
                    row.get(field) is None
                    or any(c.get("cost_kind") == "derived_responses_no_cache" for c in selected)
                )
                and selected
                and not pending
                and all(c.get("cost_usd") is not None for c in selected)
            ):
                row[field] = sum(c["cost_usd"] for c in selected)
                row["cost_source"] = "reconciled call ledger"
        if any(
            c["episode_id"] == eid and c.get("failure_reason") == "context_limit" for c in calls
        ):
            row["failure_reason"] = "context_limit"
        if row["status"] == "provider_error" and any(
            c["episode_id"] == eid
            and c.get("error_type") == "ValueError"
            and c.get("usage", {}).get("completion_tokens", 0)
            >= spec["controls"]["max_output_tokens"]
            for c in calls
        ):
            row["output_token_cap_reached"] = True
        official = directory / eid / "official.json"
        raw = json.loads(official.read_text()) if official.exists() else {}
        previous_status = row["status"]
        if episode["suite"] == "finance" and raw.get("response_payload"):
            try:
                check_response_errors(raw["response_payload"])
            except ProviderResponseError as exc:
                row.update(status="provider_error", success=None, provider_error_code=exc.code)
        if (
            episode["suite"] == "coding"
            and row["status"] == "infrastructure_error"
            and episode["task_id"] in raw.get("empty_patch_ids", [])
        ):
            row.update(
                status="graded",
                success=False,
                failure_reason="empty_patch",
                grader="swebench@" + spec["sources"]["coding"]["grader_revision"],
                timing_scope="worker duration including setup and grading",
            )
        if previous_status != row["status"]:
            corrections.append(
                {
                    "episode_id": eid,
                    "original_status": previous_status,
                    "status": row["status"],
                    "evidence": str(official),
                }
            )
        if "worker_duration_s" not in row and row["status"] == "graded":
            if official.exists():
                if episode["suite"] == "finance":
                    row["duration_s"] = raw["duration_s"]
                elif episode["suite"] == "retail":
                    row["duration_s"] = raw["simulations"][0]["duration"]
                row["timing_source"] = "upstream official artifact"
        row.update({key: episode[key] for key in ("suite", "task_id", "trial", "route", "tier")})
        normalized.append(row)
    summary = summarize(spec, normalized)
    for group in summary["groups"]:
        valid = [
            r
            for r in normalized
            if all(r[k] == group[k] for k in ("suite", "route", "tier"))
            and r["status"] != "infrastructure_error"
        ]
        times = sorted(r["duration_s"] for r in valid)
        group["median_duration_s"] = median(times) if times else None
        group["p90_duration_s"] = times[math.ceil(0.9 * len(times)) - 1] if times else None
        group["p90_right_censored"] = sum(r["status"] == "timeout" for r in valid) > len(
            times
        ) - math.ceil(0.9 * len(times))
    summary["normalized_outcomes"] = normalized
    summary["accounting_corrections"] = corrections
    by_id = {r["episode_id"]: r for r in normalized}
    matched = []
    for model in spec["models"]:
        pairs = {}
        for eid, e in episodes.items():
            if e["route"] != model["id"] or eid not in by_id:
                continue
            row = by_id[eid]
            if row["status"] == "infrastructure_error":
                continue
            pairs.setdefault((e["suite"], e["task_id"], e["trial"]), {})[e["tier"]] = row
        complete = {k: p for k, p in pairs.items() if len(p) == 2}
        group = {
            "route": model["id"],
            "matched_tasks": len(complete),
            "task_ids": [list(k) for k in complete],
        }
        for tier in ("standard", "flex"):
            selected = [p[tier] for p in complete.values()]
            costs = [r.get("agent_cost_usd") for r in selected]
            group[tier] = {
                "successes": sum(r["success"] is True for r in selected),
                "success_within_300s": sum(
                    r["success"] is True and r["duration_s"] <= 300 for r in selected
                ),
                "success_within_900s": sum(
                    r["success"] is True and r["duration_s"] <= 900 for r in selected
                ),
                "agent_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
                "execution_modes": sorted(
                    {r.get("execution_mode", "sequential") for r in selected}
                ),
            }
        matched.append(group)
    summary["paired_tier_comparisons"] = matched
    summary["inference_spend"] = {
        "settled_usd": sum(r["charged_or_reserved_usd"] for r in spend if r["settled"]),
        "unsettled_reserved_usd": sum(
            r["charged_or_reserved_usd"] for r in spend if not r["settled"]
        ),
        "charged_or_reserved_usd": sum(r["charged_or_reserved_usd"] for r in spend),
        "requests_reserved": len(spend),
        "note": "Includes probes and simulators. Doubleword costs use rates; GCP is separate.",
    }
    (directory / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/flex-agentic/pilot.json"))
    parser.add_argument("--directory", type=Path, default=Path("artifacts/flex-agentic-gcp"))
    args = parser.parse_args()
    result = analyze(json.loads(args.spec.read_text()), args.directory)
    print(
        json.dumps(
            {
                "recorded": result["recorded"],
                "spend": result["inference_spend"],
                "paired": result["paired_tier_comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
