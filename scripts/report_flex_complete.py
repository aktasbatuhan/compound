"""Merge preserved pilot phases and report task success, timing, and metered cost."""

# ruff: noqa: E501 -- Markdown table rows and explanatory report paragraphs.

import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

from summarize_flex_pilot import analyze

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/flex-agentic-gcp"
FINISH = ROOT / "artifacts/flex-agentic-finish"
OUT = ROOT / "artifacts/flex-agentic-complete"
INTERRUPTED = "7b9d44f02292fb6cff0b2461"
RATE_SNAPSHOT = json.loads(
    (ROOT / "benchmarks/flex-agentic/doubleword-rates-2026-09-09.json").read_text()
)
RATES = {(r["route"], r["tier"]): (r["input"], r["output"]) for r in RATE_SNAPSHOT["rates"]}
ERROR_WAIVER_SOURCE = "https://openrouter.zendesk.com/hc/en-us/articles/51693138951451-Was-I-charged-for-a-failed-errored-or-empty-response-Zero-Completion-Insurance"


def reconcile_error_waivers(calls, spend):
    """Apply OpenRouter's documented error waiver, without claiming invoice verification."""
    events = []
    for call in calls:
        if not (
            call["route"] in {"gpt-sol", "gemini-vertex", "gemini-studio"}
            and call.get("status") == 429
            and call.get("error_type") == "ProviderResponseError"
            and call.get("usage") == {}
            and call.get("cost_usd") is None
        ):
            continue
        matches = [
            entry
            for entry in spend
            if not entry["settled"]
            and all(entry[k] == call[k] for k in ("phase", "episode_id", "role", "reservation_usd"))
        ]
        if len(matches) != 1:
            raise ValueError("Error waiver requires one matching unsettled reservation")
        entry = matches[0]
        events.append(
            {
                "episode_id": call["episode_id"],
                "phase": call["phase"],
                "request_sha256": call["request_sha256"],
                "released_reservation_usd": entry["charged_or_reserved_usd"],
                "cost_usd": 0,
                "source": ERROR_WAIVER_SOURCE,
                "basis": "Documented error waiver; not an account billing reconciliation",
            }
        )
        entry.update(settled=True, charged_or_reserved_usd=0, settlement_source=ERROR_WAIVER_SOURCE)
        call.update(
            cost_usd=0, cost_kind="documented_error_waiver", cost_source=ERROR_WAIVER_SOURCE
        )
    return events


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def combine_accounting(baseline, continuation):
    result = []
    for phase, entries in (("pilot", baseline), ("continuation", continuation)):
        for original in entries:
            entry = {**original, "phase": phase}
            if phase == "pilot" and entry["episode_id"] == INTERRUPTED:
                entry.update(
                    original_episode_id=INTERRUPTED, episode_id=INTERRUPTED + "-interrupted"
                )
            result.append(entry)
    return result


def correct_doubleword_costs(calls, spend):
    """Remove a cache-write estimate on an endpoint that does not support caching."""
    corrections, consumed = [], set()
    for call in calls:
        if call.get("cost_kind") != "derived_with_5m_cache_write_rate":
            continue
        usage = call["usage"]
        if usage.get("prompt_tokens_details", {}).get("cached_tokens", 0):
            raise ValueError("Unexpected cache hit on the Responses endpoint; review evidence")
        rate_in, rate_out = RATES[call["route"], call["requested_tier"]]
        cost = (usage["prompt_tokens"] * rate_in + usage["completion_tokens"] * rate_out) / 1e6
        matches = [
            i
            for i, s in enumerate(spend)
            if i not in consumed
            and all(s[k] == call[k] for k in ("phase", "episode_id", "role", "reservation_usd"))
            and s["settled"]
            and math.isclose(s["charged_or_reserved_usd"], call["cost_usd"], abs_tol=1e-12)
        ]
        if not matches:
            raise ValueError("Derived call has no matching settled ledger entry")
        i = matches[0]
        consumed.add(i)
        corrections.append(
            {
                "episode_id": call["episode_id"],
                "phase": call["phase"],
                "request_sha256": call["request_sha256"],
                "original_estimate_usd": call["cost_usd"],
                "corrected_estimate_usd": cost,
                "source": "https://docs.doubleword.ai/inference-api/prompt-caching",
                "rate_source": RATE_SNAPSHOT["source"],
            }
        )
        spend[i].update(
            original_estimate_usd=spend[i]["charged_or_reserved_usd"], charged_or_reserved_usd=cost
        )
        call.update(cost_usd=cost, cost_kind="derived_responses_no_cache", cache_supported=False)
        usage["cost"] = cost
    return corrections


def merge():
    OUT.mkdir(exist_ok=True)
    shutil.copytree(BASE, OUT, dirs_exist_ok=True)
    shutil.copytree(BASE / INTERRUPTED, OUT / "attempts/budget-interrupted", dirs_exist_ok=True)
    shutil.copytree(FINISH, OUT, dirs_exist_ok=True)
    # Every remaining outcome comes from the continuation's carried-forward list.
    rows = read_rows(FINISH / "outcomes.jsonl")
    if len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate episode IDs")
    calls = combine_accounting(read_rows(BASE / "calls.jsonl"), read_rows(FINISH / "calls.jsonl"))
    spend = combine_accounting(
        json.loads((BASE / "spend.json").read_text()),
        json.loads((FINISH / "spend.json").read_text()),
    )
    corrections = correct_doubleword_costs(calls, spend)
    waivers = reconcile_error_waivers(calls, spend)
    (OUT / "error-waivers.json").write_text(json.dumps(waivers, indent=2) + "\n")
    (OUT / "cost-corrections.json").write_text(json.dumps(corrections, indent=2) + "\n")
    (OUT / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (OUT / "spend.json").write_text(json.dumps(spend, indent=2) + "\n")
    provenance = {
        "baseline": str(BASE),
        "continuation": str(FINISH),
        "superseded_budget_interruption": INTERRUPTED,
        "interruption_costs": "Retained in study spend under an -interrupted attempt ID",
        "recorded": len(rows),
    }
    (OUT / "merge-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def metrics(spec, analysis):
    episodes = {
        e["episode_id"]: e for e in json.loads((BASE / "plan.json").read_text())["episodes"]
    }
    rows = analysis["normalized_outcomes"]
    spend = json.loads((OUT / "spend.json").read_text())
    calls = read_rows(OUT / "calls.jsonl")
    groups = []
    for model in spec["models"]:
        for tier in ("standard", "flex"):
            selected = [
                r
                for r in rows
                if episodes[r["episode_id"]]["route"] == model["id"]
                and episodes[r["episode_id"]]["tier"] == tier
            ]
            valid = [r for r in selected if r["status"] != "infrastructure_error"]
            times = sorted(r["duration_s"] for r in valid)
            good = [r for r in valid if r["success"] is True]
            result = {
                "route": model["id"],
                "tier": tier,
                "recorded": len(selected),
                "valid": len(valid),
                "successes": len(good),
                "success_rate": len(good) / len(valid) if valid else None,
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "context_limit_failures": sum(
                    r.get("failure_reason") == "context_limit" for r in selected
                ),
                "failed_attempts_reaching_output_cap": sum(
                    r.get("output_token_cap_reached") is True for r in selected
                ),
                "finance_correct_accounting_wrong_refs": sum(
                    episodes[r["episode_id"]]["suite"] == "finance"
                    and r.get("metrics", {}).get("entries_accounting_correct_but_doc_refs_wrong")
                    is True
                    and r.get("metrics", {}).get("final_balance_sheet_matches") is True
                    for r in selected
                ),
                "median_duration_s": median(times) if times else None,
                "p90_duration_s": times[math.ceil(0.9 * len(times)) - 1] if times else None,
                "p90_right_censored": sum(r["status"] == "timeout" for r in valid)
                > len(times) - math.ceil(0.9 * len(times)),
                "median_right_censored": sum(r["status"] == "timeout" for r in valid)
                > len(times) // 2,
                "success_within_300s": sum(r["duration_s"] <= 300 for r in good),
                "success_within_900s": sum(r["duration_s"] <= 900 for r in good),
                "suite_successes": {
                    suite: sum(episodes[r["episode_id"]]["suite"] == suite for r in good)
                    for suite in ("coding", "retail", "finance")
                },
            }
            for role in ("agent", "auxiliary"):
                costs = [r.get(role + "_cost_usd") for r in selected]
                result[role + "_cost_usd"] = (
                    sum(costs) if all(c is not None for c in costs) else None
                )
                result[role + "_known_cost_usd"] = sum(c for c in costs if c is not None)
                ids = {r["episode_id"] for r in selected}
                ledger = [s for s in spend if s["episode_id"] in ids and s["role"] == role]
                result[role + "_settled_usd"] = sum(
                    s["charged_or_reserved_usd"] for s in ledger if s["settled"]
                )
                result[role + "_unsettled_reserved_usd"] = sum(
                    s["charged_or_reserved_usd"] for s in ledger if not s["settled"]
                )
            usage_calls = [
                c
                for c in calls
                if c["episode_id"] in ids and c["role"] == "agent" and c.get("usage")
            ]
            agent_calls = [c for c in calls if c["episode_id"] in ids and c["role"] == "agent"]
            result["agent_api_calls"] = len(agent_calls)
            result["median_agent_calls_per_episode"] = (
                median(sum(c["episode_id"] == eid for c in agent_calls) for eid in ids)
                if ids
                else None
            )
            result["completion_tokens"] = sum(
                c["usage"].get("completion_tokens", 0) for c in usage_calls
            )
            result["reasoning_tokens"] = sum(
                (c["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                for c in usage_calls
            )
            prompts = sum(c["usage"].get("prompt_tokens", 0) for c in usage_calls)
            cache_calls = [
                c
                for c in usage_calls
                if "cached_tokens" in c["usage"].get("prompt_tokens_details", {})
            ]
            cache_measured_prompts = sum(c["usage"].get("prompt_tokens", 0) for c in cache_calls)
            cached = sum(
                c["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                for c in cache_calls
            )
            result["prompt_tokens"] = prompts
            result["cached_tokens"] = cached
            result["cache_measurement_prompt_coverage"] = (
                cache_measured_prompts / prompts if prompts else None
            )
            result["cached_input_fraction"] = (
                cached / cache_measured_prompts if cache_measured_prompts else None
            )
            if model["provider"] == "doubleword":
                result.update(
                    cached_input_fraction=None,
                    cache_supported=False,
                    cache_note="Prompt caching is unsupported on Doubleword Responses API",
                )
            complete_cost = all(
                result[role + "_cost_usd"] is not None for role in ("agent", "auxiliary")
            )
            total_cost = (
                sum(result[role + "_cost_usd"] for role in ("agent", "auxiliary"))
                if complete_cost
                else None
            )
            result["task_inference_cost_usd"] = total_cost
            result["inference_cost_per_success_usd"] = (
                total_cost / len(good) if total_cost is not None and good else None
            )
            groups.append(result)
    return groups


def main():
    merge()
    spec = json.loads((ROOT / "benchmarks/flex-agentic/pilot.json").read_text())
    analysis = analyze(spec, OUT)
    groups = metrics(spec, analysis)
    ledger = json.loads((OUT / "spend.json").read_text())
    recorded_ids = {r["episode_id"] for r in analysis["normalized_outcomes"]}
    grouped = sum(g[role + "_settled_usd"] for g in groups for role in ("agent", "auxiliary"))
    outside = sum(
        s["charged_or_reserved_usd"]
        for s in ledger
        if s["settled"] and s["episode_id"] not in recorded_ids
    )
    if not math.isclose(
        grouped + outside, analysis["inference_spend"]["settled_usd"], abs_tol=1e-9
    ):
        raise ValueError("Task-group costs do not reconcile with study spend")
    (OUT / "accounting-audit.json").write_text(
        json.dumps(
            {
                "recorded_task_settled_usd": grouped,
                "outside_recorded_tasks_settled_usd": outside,
                "study_settled_usd": analysis["inference_spend"]["settled_usd"],
                "reconciled": True,
                "note": "Outside-task spend includes probes and the budget-interrupted attempt; during execution it also includes unrecorded active tasks.",
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "metrics.json").write_text(json.dumps(groups, indent=2) + "\n")
    lines = [
        "# Flex API agentic pilot",
        "",
        f"Recorded {analysis['recorded']} of 150 planned episodes: five tasks in each of coding, retail, and finance, across five routes and two tiers. One trial per task. This is a small pilot, not a provider ranking.",
        "",
        "Routes: `gpt-sol` is GPT-5.6 Sol through OpenRouter/OpenAI; `gemini-vertex` and `gemini-studio` are Gemini 3.8 Flash through OpenRouter's Vertex and AI Studio endpoints. `deepseek-dw` is DeepSeek V4 Flash 0731 and `glm-dw` is GLM 5.3, both directly through Doubleword. Standard means the normal OpenRouter endpoint or Doubleword realtime; Flex means the pinned Flex endpoint or explicit Flex service tier. No Batch API is used.",
        "",
        "Controls: 8,192 output tokens per agent call, medium reasoning requested, provider-default temperature, no automatic retries, and a 30-minute worker deadline. Native step limits are 40 for coding, 30 for retail, and eight for finance. Both tiers of each model use the same limits. Medium effort is not equal reasoning compute across different model families.",
        "",
        "## Task outcomes",
        "",
        "| Route | Tier | Success / valid | Coding /5 | Retail /5 | Finance /5 | API errors | Timeouts |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def money(x):
        return f"{x:.4f}" if x is not None else "unknown"

    for g in groups:
        suite = g["suite_successes"]
        lines.append(
            f"| {g['route']} | {g['tier']} | {g['successes']}/{g['valid']} | {suite['coding']} | {suite['retail']} | {suite['finance']} | {g['status_counts'].get('provider_error', 0)} | {g['status_counts'].get('timeout', 0)} |"
        )
    lines += [
        "",
        "## Completion time",
        "",
        "| Route | Tier | Median seconds | P90 seconds | Success ≤5m /15 | Success ≤15m /15 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for g in groups:
        median_label = ("≥" if g["median_right_censored"] else "") + f"{g['median_duration_s']:.1f}"
        p90_label = ("≥" if g["p90_right_censored"] else "") + f"{g['p90_duration_s']:.1f}"
        lines.append(
            f"| {g['route']} | {g['tier']} | {median_label} | {p90_label} | {g['success_within_300s']} | {g['success_within_900s']} |"
        )
    lines += [
        "",
        "## Inference cost and cache",
        "",
        "| Route | Tier | Agent $ settled | Agent $ reserved | Simulator $ | Inference $ / success | Cached input |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        cache = (
            f"{g['cached_input_fraction']:.1%}"
            if g["cached_input_fraction"] is not None
            else "unsupported"
            if g.get("cache_supported") is False
            else "unknown"
        )
        lines.append(
            f"| {g['route']} | {g['tier']} | {money(g['agent_settled_usd'])} | {money(g['agent_unsettled_reserved_usd'])} | {money(g['auxiliary_cost_usd'])} | {money(g['inference_cost_per_success_usd'])} | {cache} |"
        )
    spend = analysis["inference_spend"]
    cloud = None
    first_status = json.loads((BASE / "supervisor-status.json").read_text())
    last_status = json.loads((FINISH / "supervisor-status.json").read_text())
    provision = json.loads((FINISH / "vm-provisioning.json").read_text())
    if first_status.get("vm_deleted") and last_status.get("vm_deleted"):
        lifetimes = [
            {
                "vm": "compound-flex-20260909",
                "start": first_status["first_started_at"],
                "end": first_status["finished_at"],
            },
            {
                "vm": provision["name"],
                "start": provision["first_started_at"],
                "end": last_status["finished_at"],
            },
        ]
        for lifetime in lifetimes:
            lifetime["hours"] = (
                datetime.fromisoformat(lifetime["end"]) - datetime.fromisoformat(lifetime["start"])
            ).total_seconds() / 3600
        cloud = {
            "vm_lifetimes": lifetimes,
            "hourly_usd": provision["on_demand_hourly_usd"],
            "compute_estimate_usd": sum(v["hours"] for v in lifetimes)
            * provision["on_demand_hourly_usd"],
            "source": provision["rate_source"],
            "vm_and_disks_deleted": True,
            "note": "Elapsed lifetime includes the first VM's brief stopped interval. Conservative compute estimate; storage, network, and taxes excluded. Not a billing export.",
        }
        (OUT / "gcp-cost-estimate.json").write_text(json.dumps(cloud, indent=2) + "\n")
    lines += [
        "",
        f"Study inference total: ${spend['settled_usd']:.4f} settled, plus ${spend['unsettled_reserved_usd']:.4f} reserved for uncertain charges. Includes probes, simulators, and the budget-interrupted attempt. GCP compute, storage, and networking are separate.",
        "",
        "Success means the upstream task grader passed. Error and timeout columns count task attempts, not individual API calls. They count as unsuccessful attempts; infrastructure errors are excluded and reported separately. Coding uses SWE-bench Verified, retail uses tau2-verified, and finance uses FinBalance's strict composite scorer, including exact supporting document references. The metrics JSON separately counts finance answers with correct accounting and balances but document-reference mismatches. The primary pass criterion has not been relaxed.",
        "",
        "API errors include explicit request rejections and adapter errors, not just capacity failures. metrics.json separately counts context_limit_failures and failed_attempts_reaching_output_cap. The latter is based on recorded usage, not a recovered upstream error message. These counts must not be presented as provider outage rates.",
        "",
        "Duration measures the agent loop; retail includes user-simulator time. Coding setup and post-submission grading are excluded where separately recorded. Hard timeouts use the 1,800-second worker ceiling, which includes setup and grading; they are censored observations, not measured completion times. The median and nearest-rank P90 include valid successes and failures across the same 15-task mix, so early errors can lower latency. Success within five minutes requires both correctness and completion by the deadline. These blocking calls do not measure TTFT.",
        "",
        "OpenRouter costs come from response usage. Doubleword costs are derived from observed token counts and declared input/output rates. Its Responses endpoint does not support prompt caching; the original harness's cache-write surcharge was removed in offline accounting, with each adjustment in cost-corrections.json and original checkpoints unchanged. See [Doubleword's caching documentation](https://docs.doubleword.ai/inference-api/prompt-caching). Unknown charges remain unknown rather than zero. Task-level costs exclude infrastructure overhead and the earlier budget-interrupted attempt; study totals retain both attempts' inference costs.",
        "",
        f"Four OpenRouter 429 error responses returned no usage or cost. These are assigned $0 under OpenRouter's [documented error waiver]({ERROR_WAIVER_SOURCE}); error-waivers.json records the released reservations and policy source. This is policy-based accounting, not verification against account billing. Task failures remain unchanged.",
        "",
        "Cost per success divides all task inference costs, including unsuccessful attempts and user simulators, by passed tasks. It is unknown where any charge is unresolved. Cache share divides reported cached input tokens by input tokens on calls with cache evidence; it describes these evolving agent conversations, not identical-prompt cache tests. Coverage is included in metrics.json.",
        "",
        "There is one locally active agent episode per route and one active coding worker globally. Stopping a worker at the deadline does not cancel work already submitted to a provider. An in-flight call may finish after the timeout and briefly overlap a subsequent episode; its recorded cost remains attributed to the timed-out attempt.",
        "",
        "The original and continuation checkpoints remain unchanged. merge-provenance.json identifies the budget interruption replaced by its clean rerun. analysis.json records historical status/timing corrections with evidence, and includes matched standard/Flex comparisons. Uncertainty is substantial at five tasks per suite; no statistical superiority is claimed.",
        "",
    ]
    if cloud:
        lines += [
            f"GCP compute estimate: **${cloud['compute_estimate_usd']:.2f}**, based on recorded VM lifetimes and the [on-demand E2 rate]({cloud['source']}). This includes the brief stopped interval conservatively; storage, network, and taxes are not reconciled. Both dedicated VMs and their boot disks were deleted.",
            "",
        ]
    (OUT / "report.md").write_text("\n".join(lines))
    print(
        json.dumps({"recorded": analysis["recorded"], "spend": spend, "metrics": groups}, indent=2)
    )


if __name__ == "__main__":
    main()
