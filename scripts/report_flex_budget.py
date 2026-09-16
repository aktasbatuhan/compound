"""Report the small fixed-budget retail experiment from saved calls and grades."""

# ruff: noqa: E501 -- Generated Markdown paragraphs and table rows.

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1] / "artifacts/flex-budget-20260910"


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    stage = "token-informed" if (ROOT / "token-informed/study-spec.json").exists() else "main"
    metadata = ROOT / stage if (ROOT / stage / "study-spec.json").exists() else ROOT
    spec = json.loads((metadata / "study-spec.json").read_text())
    plan = json.loads((metadata / "plan.json").read_text())
    outcomes = {x["episode_id"]: x for x in read_rows(ROOT / stage / "outcomes.jsonl")}
    calls = read_rows(ROOT / stage / "calls.jsonl")
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
    metrics = []
    for model in spec["models"]:
        for tier in ("standard", "flex"):
            expected = [
                e for e in plan["episodes"] if e["route"] == model["id"] and e["tier"] == tier
            ]
            rows = [outcomes[e["episode_id"]] for e in expected if e["episode_id"] in outcomes]
            ids = {e["episode_id"] for e in expected}
            selected = [c for c in calls if c["episode_id"] in ids]
            passed = [r for r in rows if r.get("success") is True]
            metrics.append(
                {
                    "route": model["id"],
                    "tier": tier,
                    "budget_usd": model["episode_budget_usd"],
                    "planned": len(expected),
                    "recorded": len(rows),
                    "successes": len(passed),
                    "tasks_passed_both_trials": sum(
                        all(
                            outcomes.get(e["episode_id"], {}).get("success") is True
                            for e in expected
                            if e["task_id"] == task_id
                        )
                        for task_id in {e["task_id"] for e in expected}
                    ),
                    "success_by_deadline": {
                        str(t): sum(r["duration_s"] <= t for r in passed) / len(expected)
                        for t in spec["deadlines_s"]
                    },
                    "status_counts": dict(Counter(r["status"] for r in rows)),
                    "median_attempt_s": median(r["duration_s"] for r in rows) if rows else None,
                    "median_successful_attempt_s": median(r["duration_s"] for r in passed)
                    if passed
                    else None,
                    "agent_cost_usd": sum(c["cost_usd"] for c in selected if c["role"] == "agent")
                    if all(c.get("cost_usd") is not None for c in selected if c["role"] == "agent")
                    else None,
                    "simulator_cost_usd": sum(
                        c["cost_usd"] for c in selected if c["role"] == "auxiliary"
                    )
                    if all(
                        c.get("cost_usd") is not None for c in selected if c["role"] == "auxiliary"
                    )
                    else None,
                }
            )
    for m in metrics:
        complete = m["recorded"] == m["planned"]
        known_cost = m["agent_cost_usd"] is not None and m["simulator_cost_usd"] is not None
        m["inference_cost_per_success_usd"] = (
            (m["agent_cost_usd"] + m["simulator_cost_usd"]) / m["successes"]
            if complete and known_cost and m["successes"]
            else None
        )
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    spends = {}
    for accounting_stage in ("smoke", "main", "token-informed"):
        p = ROOT / accounting_stage / "spend.json"
        entries = json.loads(p.read_text()) if p.exists() else []
        spends[accounting_stage] = {
            "settled_usd": sum(e["charged_or_reserved_usd"] for e in entries if e["settled"]),
            "unsettled_reserved_usd": sum(
                e["charged_or_reserved_usd"] for e in entries if not e["settled"]
            ),
        }
    status = json.loads((ROOT / "supervisor-status.json").read_text())
    vm = json.loads((ROOT / "vm.json").read_text())
    compute = None
    if status.get("vm_deleted"):
        hours = (
            datetime.fromisoformat(status["finished_at"])
            - datetime.fromisoformat(vm["lastStartTimestamp"])
        ).total_seconds() / 3600
        compute = hours * 0.13402284
    (ROOT / "accounting.json").write_text(
        json.dumps(
            {
                "inference": spends,
                "estimated_compute_usd": compute,
                "storage_network_not_reconciled": True,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Fixed-budget Flex engineering trial",
        "",
        f"Reported phase: {stage}. The initial byte-reservation diagnostic is retained separately in main/ and is not pooled with the corrected run. Its spend remains included below.",
        "",
        f"Recorded {len(outcomes)}/{plan['episode_count']} attempts. Two previously seen retail tasks, two trials, five routes. This is a mechanism test, not a model ranking or a precise estimate of reliability.",
        "",
        "| Route | Tier | Agent cap | Recorded /4 | Passed /4 | Correct ≤1m | Correct ≤5m | Correct ≤15m | Budget stops |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        d = m["success_by_deadline"]
        lines.append(
            f"| {m['route']} | {m['tier']} | ${m['budget_usd']:.2f} | {m['recorded']} | {m['successes']} | {d['60']:.0%} | {d['300']:.0%} | {d['900']:.0%} | {m['status_counts'].get('budget_exhausted', 0)} |"
        )
    lines += [
        "",
        "| Route | Tier | Tasks passing both trials /2 | Agent spend | Simulator spend | Status counts |",
        "|---|---|---:|---:|---:|---|",
    ]
    for m in metrics:
        agent = f"${m['agent_cost_usd']:.4f}" if m["agent_cost_usd"] is not None else "unresolved"
        simulator = (
            f"${m['simulator_cost_usd']:.4f}"
            if m["simulator_cost_usd"] is not None
            else "unresolved"
        )
        lines.append(
            f"| {m['route']} | {m['tier']} | {m['tasks_passed_both_trials']} | {agent} | {simulator} | {json.dumps(m['status_counts'])} |"
        )
    lines += [
        "",
        "Equal dollar allowances apply within each route pair. Astra has a larger allowance than the other models. The corrected gateway uses prior provider-reported input tokens plus changed UTF-8 bytes and a 4096-token framing allowance when available; the first call uses a full byte bound. It shrinks the output limit to fit and can stop before the exact token allowance is spent. These are conservative admission budgets, not exact provider-token budget frontiers.",
        "",
        "Provider errors, budget stops, and timeouts remain unsuccessful attempts in the planned denominator. Missing results are pending, not measured failures. Simulator failures are harness failures. Times are native agent-loop durations when grading completes; timeout ceilings include worker overhead. No hidden-grader retries are used.",
        "",
        "In the completed corrected run, Vertex Flex had two upstream 429 rate-limit failures and Studio realtime had one HTTP 503. GLM Flex's failed task was graded unsuccessful after the user simulator emitted OUT-OF-SCOPE when the agent asked for an order ID. It remains an unsuccessful task attempt, but this single simulator-ended conversation does not establish a Flex quality effect.",
        "",
        "The curves are saved in [deadline-curves.svg](deadline-curves.svg) and [deadline-curves.png](deadline-curves.png). Rebuild them with `uv run --with matplotlib python scripts/plot_flex_budget.py` after rebuilding this report.",
        "",
        "Doubleword uses the uncached Responses API. OpenRouter uses pinned endpoints and ambient caching. Reported costs and Doubleword token-rate estimates are preserved in the call ledger; unresolved calls retain their reservations. Simulator and GCP costs are separate from agent allowances.",
        "",
        "Inference accounting: " + json.dumps(spends),
        "",
        f"GCP compute estimate: {compute}. Storage and network billing are not reconciled.",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"recorded": len(outcomes), "metrics": metrics, "spend": spends, "compute": compute},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
