"""Rebuild the harder-task trial report from its local checkpoint."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1] / "artifacts/flex-hard-20260910"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    spec = json.loads((ROOT / "study-spec.json").read_text())
    rows = read_rows(ROOT / "run/outcomes.jsonl")
    outcomes = {r["episode_id"]: r for r in rows}
    assert len(rows) == len(outcomes)
    assert set(outcomes) <= {e["episode_id"] for e in plan["episodes"]}
    assert all(r["spec_sha256"] == plan["spec_sha256"] for r in rows)
    calls = read_rows(ROOT / "run/calls.jsonl")
    interpretations = []
    for eid, original in list(outcomes.items()):
        row = dict(original)
        log = ROOT / "run" / eid / "worker.log"
        if (
            row["status"] == "infrastructure_error"
            and log.exists()
            and "ValueError: AssistantMessage must have either content or tool calls."
            in log.read_text()
            and not any(
                c["role"] == "auxiliary" and c["status"] != 200
                for c in calls
                if c["episode_id"] == eid
            )
        ):
            row.update(status="provider_error", failure_reason="empty_agent_message")
            interpretations.append(
                {
                    "episode_id": eid,
                    "original_status": original["status"],
                    "interpreted_status": row["status"],
                    "evidence": str(log.relative_to(ROOT)),
                }
            )
        proof = ROOT / "run" / eid / "official.json"
        if proof.exists():
            simulations = json.loads(proof.read_text()).get("simulations", [])
            if simulations:
                row["termination_reason"] = simulations[0]["termination_reason"]
        outcomes[eid] = row
    (ROOT / "outcome-interpretations.json").write_text(json.dumps(interpretations, indent=2) + "\n")
    metrics = []
    for model in spec["models"]:
        for tier in ("standard", "flex"):
            expected = [
                e for e in plan["episodes"] if e["route"] == model["id"] and e["tier"] == tier
            ]
            ids = {e["episode_id"] for e in expected}
            selected = [outcomes[eid] for eid in ids if eid in outcomes]
            passed = [r for r in selected if r.get("success") is True]
            costs = {}
            for role in ("agent", "auxiliary"):
                rc = [c for c in calls if c["episode_id"] in ids and c["role"] == role]
                costs[role] = (
                    sum(c["cost_usd"] for c in rc)
                    if all(c.get("cost_usd") is not None for c in rc)
                    else None
                )
            metrics.append(
                {
                    "route": model["id"],
                    "tier": tier,
                    "budget_usd": model["episode_budget_usd"],
                    "planned": len(expected),
                    "recorded": len(selected),
                    "successes": len(passed),
                    "success_by_deadline": {
                        str(d): sum(r["duration_s"] <= d for r in passed) / len(expected)
                        for d in spec["deadlines_s"]
                    },
                    "tasks_passed_both_trials": sum(
                        all(
                            outcomes.get(e["episode_id"], {}).get("success") is True
                            for e in expected
                            if e["task_id"] == task
                        )
                        for task in {e["task_id"] for e in expected}
                    ),
                    "status_counts": dict(Counter(r["status"] for r in selected)),
                    "termination_counts": dict(
                        Counter(r.get("termination_reason", "no_trajectory") for r in selected)
                    ),
                    "median_successful_attempt_s": median(r["duration_s"] for r in passed)
                    if passed
                    else None,
                    "agent_cost_usd": costs["agent"],
                    "simulator_cost_usd": costs["auxiliary"],
                    "inference_cost_per_success_usd": sum(costs.values()) / len(passed)
                    if passed
                    and len(selected) == len(expected)
                    and all(c is not None for c in costs.values())
                    else None,
                }
            )
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    spend_path = ROOT / "run/spend.json"
    spend = json.loads(spend_path.read_text()) if spend_path.exists() else []
    settled = sum(e["charged_or_reserved_usd"] for e in spend if e["settled"])
    reserved = sum(e["charged_or_reserved_usd"] for e in spend if not e["settled"])
    status = json.loads((ROOT / "supervisor-status.json").read_text())
    vm = json.loads((ROOT / "vm.json").read_text())
    compute = None
    if status.get("vm_deleted"):
        compute = (
            (
                datetime.fromisoformat(status["finished_at"])
                - datetime.fromisoformat(vm["creationTimestamp"])
            ).total_seconds()
            / 3600
            * 0.13402284
        )
    accounting = {
        "settled_inference_usd": settled,
        "unresolved_reserved_usd": reserved,
        "estimated_compute_usd": compute,
        "storage_network_not_reconciled": True,
        "combined_conservative_total_usd": 3.6187636556 + settled + reserved + compute
        if compute is not None
        else None,
    }
    (ROOT / "accounting.json").write_text(json.dumps(accounting, indent=2) + "\n")
    audit = {
        "input_bound_violations": [
            c["request_sha256"]
            for c in calls
            if (c.get("usage") or {}).get("prompt_tokens", 0)
            > c.get("input_token_bound", float("inf"))
        ],
        "cost_above_reservation": [
            c["request_sha256"]
            for c in calls
            if c.get("cost_usd") is not None and c["cost_usd"] > c["reservation_usd"] + 1e-9
        ],
    }
    (ROOT / "reservation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# Harder-task Flex trial",
        "",
        f"{len(rows)}/{plan['episode_count']} attempts recorded. Tasks 23, 30 and 41, "
        "two repetitions, both tiers, five routes. These tasks were selected for structural "
        "complexity before observing paid outcomes, not for measured difficulty.",
        "",
        "| Route | Tier | Recorded /6 | Official passes /6 | Correct ≤1m | Correct ≤5m | "
        "Correct ≤15m | Both trials /3 | Status counts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in metrics:
        d = m["success_by_deadline"]
        lines.append(
            f"| {m['route']} | {m['tier']} | {m['recorded']} | {m['successes']} | "
            f"{d['60']:.0%} | {d['300']:.0%} | {d['900']:.0%} | "
            f"{m['tasks_passed_both_trials']} | {json.dumps(m['status_counts'])} |"
        )
    lines += [
        "",
        "Budgets remain $0.40 per Astra attempt and $0.10 for other routes. "
        "Simulator calls have separate limits. The entire new run has a $5.20 inference "
        "ceiling. Pending attempts are unmeasured; errors, timeouts and budget stops "
        "remain in the planned deadline denominator. Budget stops can occur before exact "
        "token spend reaches the allowance because admission reserves conservatively.",
        "",
        "All three reference database trajectories passed offline; empty trajectories failed. "
        "Task 30 additionally requires communicating a tracking number and uses the native "
        "metered communication judge. Simulator behavior remains a source of variance.",
        "",
        "These are single-attempt allowance comparisons, not retries until a shared task "
        "budget is consumed. The small sample cannot establish tier parity or a ranking.",
        "",
        "Task 41 has a confirmed scoring ambiguity. Both Astra Flex trajectories obeyed the "
        "simulated user's explicit request to refund Visa, which the modify-items policy permits. "
        "The reference expects PayPal. Replaying the full trajectories shows "
        "the refund destination "
        "is the only database difference. Official grades remain unchanged, but these failures "
        "must not be claimed as quality degradation. See task41-payment-audit.json and the raw "
        "user confirmations. Passing reference replay alone did not establish semantic validity.",
        "",
        "The first empty agent response was initially labeled an infrastructure error. "
        "outcome-interpretations.json records its evidence-based relabeling without changing "
        "the raw outcome or retrying it. The worker classification was fixed before resuming "
        "unstarted attempts; execution source amendments are retained. max_steps terminations "
        "are step-limit failures, not completed incorrect answers; "
        "see termination_counts in metrics.json.",
        "",
        "Accounting (includes simulator; unresolved charges retain reservations): "
        + json.dumps(accounting),
        "",
        "[Deadline curves](deadline-curves.png). Full costs and repeated-success counts "
        "are in metrics.json; raw calls and grades are in run/.",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines))
    print(json.dumps({"recorded": len(rows), "accounting": accounting, "audit": audit}))


if __name__ == "__main__":
    main()
