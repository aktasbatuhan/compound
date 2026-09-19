"""Budgeted GCP pilot controller. All inference, including simulators, is metered."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from compound.agentic_gateway import Gateway, missing_rates
from compound.agentic_safety import (
    cache_failures,
    preparation_errors,
    qualification_errors,
    seal_run,
    tier_evidence_errors,
)
from compound.agentic_study import plan, summarize, verify_sources


def qualification_body():
    """Repeated prefix large enough for caching, within the simulator probe cap."""
    return {
        "messages": [
            {
                "role": "user",
                "content": "Reference context for a cache qualification test. "
                * 300
                + "\nCall the echo tool with value READY.",
            }
        ],
        "max_tokens": 512,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
    }


def attempt_calls(calls, episode_id, since):
    """Calls belonging to this attempt only.

    A re-run episode keeps its id, so the ledger still holds the calls of the
    attempt that was interrupted. Classifying or costing an episode from those
    stale rows misreads a clean re-run as a repeat of the old failure.
    """
    return [c for c in calls if c["episode_id"] == episode_id and c["started_at"] >= since]


#: Enough marked calls that a zero hit rate is the endpoint, not a cold start.
CACHE_GATE_MIN_CALLS = 8


def account_failure(calls):
    """True when any call failed for account balance.

    A 402 is an account-level condition, not this episode's fault: every later
    episode will fail the same way. It is terminal for the run, unlike a provider
    error, which is a property of the attempt and stays in the denominator.
    """
    return any(c.get("status") == 402 for c in calls)


def work_units(episodes):
    """Group consecutive episodes of one task and trial so a tier pair shares a lane.

    The plan orders the two tiers of a task adjacently, so grouping on the run
    of equal keys keeps a pair back to back in one lane instead of splitting it
    across two that finish at different times.
    """
    units = []
    for episode in episodes:
        key = (
            episode["suite"],
            episode["task_id"],
            episode["trial"],
            episode["route"],
            episode.get("budget_usd"),
        )
        if units and units[-1][0] == key:
            units[-1][1].append(episode)
        else:
            units.append((key, [episode]))
    return [unit for _, unit in units]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["probe", "oracle", "run"])
    parser.add_argument("--spec", default="benchmarks/flex-agentic/pilot.json")
    parser.add_argument("--out", default="artifacts/flex-agentic-gcp")
    parser.add_argument("--suite", choices=["finance", "retail", "coding"])
    parser.add_argument("--task")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--inference-limit", type=float, default=12)
    parser.add_argument("--parallel-routes", action="store_true")
    parser.add_argument(
        "--lanes",
        type=int,
        default=1,
        help="Concurrent episodes within one route. Latency measured above 1 describes "
        "that load, and is not comparable with a single-lane run.",
    )
    parser.add_argument("--stop-at", help="Stop starting episodes at this ISO UTC timestamp")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.lanes < 1:
        parser.error("--lanes must be at least 1")
    if args.parallel_routes and args.lanes > 1:
        execution_mode = f"parallel_routes_lanes_{args.lanes}"
    elif args.parallel_routes:
        execution_mode = "parallel_routes"
    elif args.lanes > 1:
        execution_mode = f"lanes_{args.lanes}"
    else:
        execution_mode = "sequential"
    stop_at = datetime.fromisoformat(args.stop_at) if args.stop_at else None
    if stop_at is not None and stop_at.utcoffset() is None:
        parser.error("--stop-at requires an explicit timezone")
    spec = json.loads(Path(args.spec).read_text())
    study = plan(spec)
    errors = verify_sources(spec)
    if errors:
        raise ValueError(errors)
    unpriced = missing_rates(spec)
    if unpriced:
        raise ValueError(f"no rate table entry for: {', '.join(unpriced)}")
    if args.stage != "oracle" and not args.go:
        print(
            f"Dry run: add --go to authorize this stage. Inference guard ${args.inference_limit}."
        )
        return
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    runner_lock = (directory / "runner.lock").open("a")
    fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage != "oracle":
        preparation = preparation_errors(spec, directory)
        if preparation:
            raise ValueError("paid execution blocked: " + "; ".join(preparation))
    seal_run(directory, spec, study, execution_mode)
    if args.stage == "run":
        qualification = qualification_errors(spec, directory)
        if qualification:
            raise ValueError("paid run blocked pending qualification: " + "; ".join(qualification))
    if args.stage != "oracle":
        os.environ.update(json.loads(Path("keys.json").read_text()))
    for model in spec["models"]:
        for tier in ["standard", "flex"]:
            study["episodes"].append(
                {
                    "episode_id": "probe-" + model["id"] + "-" + tier,
                    "suite": "finance",
                    "route": model["id"],
                    "tier": tier,
                    "task_id": "probe",
                    "trial": 0,
                }
            )
    if "retail" in spec["sources"]:
        study["episodes"].append(
            {
                "episode_id": "probe-simulator-standard",
                "suite": "finance",
                "route": spec["controls"].get("simulator_route", "gpt-sol"),
                "tier": "standard",
                "task_id": "probe",
                "trial": 0,
            }
        )
    stopped = threading.Event()
    gateway = Gateway(spec, study, directory, limit=args.inference_limit, stop_event=stopped)
    server = gateway.serve()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        if args.stage == "probe":
            probe_path = directory / "probes.json"
            probes = json.loads(probe_path.read_text()) if probe_path.exists() else []
            for e in study["episodes"]:
                role = "auxiliary" if e["episode_id"] == "probe-simulator-standard" else "agent"
                if not e["episode_id"].startswith("probe-"):
                    continue
                if stopped.is_set():
                    break
                try:
                    body = qualification_body()
                    gateway.active_attempts[e["episode_id"]] = uuid.uuid4().hex
                    # Both requests carry the same cache-enabled prefix, including
                    # the first cache write. No uncached control is sent.
                    gateway.call(e["episode_id"], role, body)
                    raw = gateway.call(e["episode_id"], role, body)
                    tool_calls = raw["choices"][0]["message"].get("tool_calls") or []
                    ok = any(
                        c["function"]["name"] == "echo"
                        and json.loads(c["function"]["arguments"]).get("value") == "READY"
                        for c in tool_calls
                    )
                    row = {"episode_id": e["episode_id"], "role": role, "ok": ok, "reply": raw}
                except Exception as exc:
                    row = {
                        "episode_id": e["episode_id"],
                        "role": role,
                        "ok": False,
                        "error": str(exc),
                    }
                row["attempt_id"] = gateway.active_attempts.get(e["episode_id"])
                probes.append(row)
                probe_path.write_text(json.dumps(probes, indent=2) + "\n")
                print(e["episode_id"], row["ok"], flush=True)
            return
        episodes = [
            e
            for e in study["episodes"]
            if not e["episode_id"].startswith("probe-")
            and (not args.suite or e["suite"] == args.suite)
            and (not args.task or e["task_id"] == args.task)
        ]
        if args.stage == "oracle":
            episodes = []
            for task in spec["sources"]["coding"]["tasks"]:
                episodes.append(
                    {"episode_id": "oracle-" + task["id"], "suite": "coding", "task_id": task["id"]}
                )
        rows_path = directory / "outcomes.jsonl"
        rows = (
            [json.loads(line) for line in rows_path.read_text().splitlines()]
            if rows_path.exists()
            else []
        )
        completed = {r["episode_id"] for r in rows}
        if args.stage != "oracle":
            summarize(spec, rows)  # Reject foreign, duplicated, or malformed outcomes before calls.
        remaining = [e for e in episodes if e["episode_id"] not in completed]
        pending_ids = {e["episode_id"] for e in remaining}
        interrupted = {x["episode_id"] for x in gateway.guard.entries} & pending_ids
        if interrupted:
            raise ValueError(
                "interrupted episodes have paid reservations but no outcome; reconcile their "
                "evidence before resuming: " + ", ".join(sorted(interrupted))
            )
        episodes = []
        for unit in work_units(remaining):
            if len(episodes) + len(unit) > args.count:
                break
            episodes.extend(unit)
        rows_lock = threading.Lock()
        coding_lock = threading.Lock()

        def execute(e):
            if stop_at is not None and datetime.now(UTC) >= stop_at:
                print(
                    "Checkpoint deadline reached; leaving unstarted episodes pending.", flush=True
                )
                stopped.set()
                return
            if (
                args.stage != "oracle"
                and sum(x["charged_or_reserved_usd"] for x in gateway.guard.entries)
                >= args.inference_limit
            ):
                print(
                    "Stopping with inference reserve; remaining episodes stay pending.", flush=True
                )
                stopped.set()
                return
            attempt_id = uuid.uuid4().hex
            gateway.active_attempts[e["episode_id"]] = attempt_id
            output = directory / "attempts" / e["episode_id"] / attempt_id
            output.mkdir(parents=True, exist_ok=False)
            log_path = directory / "worker-logs" / (attempt_id + ".log")
            log_path.parent.mkdir(exist_ok=True)
            env = os.environ.copy()
            for key in ["OPENROUTER_API_KEY", "DOUBLEWORD_API_KEY"]:
                env.pop(key, None)
            env.update(
                PYTHONPATH="src:.compound/sources/finbalance:.compound/sources/tau2-bench-verified/src:.compound/sources/mini-swe-agent/src",
                TAU2_DATA_DIR=str(Path(".compound/sources/tau2-bench-verified/data").resolve()),
                MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="1",
                OPENAI_API_KEY="local-meter",
                LITELLM_LOCAL_MODEL_COST_MAP="True",
            )
            cmd = [
                sys.executable,
                "-m",
                "compound.agentic_worker",
                "--spec",
                args.spec,
                "--episode",
                json.dumps(e),
                "--base",
                base + "/" + e["episode_id"] + "/agent/v1",
                "--out",
                str(output),
            ]
            if args.stage == "oracle":
                cmd.append("--oracle")
            attempt_started = datetime.now(UTC).isoformat()
            print("START", e["episode_id"], e["suite"], e.get("route"), e.get("tier"), flush=True)
            with log_path.open("x") as log:
                child = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    child.wait(timeout=spec["controls"]["max_episode_s"])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    (output / "outcome.json").write_text(
                        json.dumps(
                            {
                                "status": "timeout",
                                "success": None,
                                "duration_s": spec["controls"]["max_episode_s"],
                            }
                        )
                    )
            if not (output / "outcome.json").exists():
                result = {"status": "infrastructure_error", "success": None, "duration_s": 0}
            else:
                result = json.loads((output / "outcome.json").read_text())
            if args.stage == "oracle":
                print("ORACLE", e["task_id"], result, flush=True)
                if not result.get("success"):
                    raise ValueError("coding reference patch failed qualification")
                return
            with gateway.calls_lock:
                calls_path = directory / "calls.jsonl"
                calls = (
                    [json.loads(line) for line in calls_path.read_text().splitlines()]
                    if calls_path.exists()
                    else []
                )
            calls_all = calls
            calls = [
                c
                for c in attempt_calls(calls, e["episode_id"], attempt_started)
                if c.get("attempt_id") == attempt_id
            ]
            auxiliary_failed = any(c["status"] != 200 and c["role"] == "auxiliary" for c in calls)
            if (
                result["status"] == "infrastructure_error"
                and not auxiliary_failed
                and any(c["status"] != 200 and c["role"] == "agent" for c in calls)
            ):
                result["status"] = "provider_error"
            if auxiliary_failed:
                result.update(status="infrastructure_error", success=None, failure_role="auxiliary")

            stops_path = directory / "budget-stops.jsonl"
            stops = (
                [json.loads(line) for line in stops_path.read_text().splitlines()]
                if stops_path.exists()
                else []
            )
            if (
                any(
                    s["episode_id"] == e["episode_id"]
                    and s["role"] == "agent"
                    and s.get("attempt_id") == attempt_id
                    for s in stops
                )
                and not auxiliary_failed
            ):
                result.update(status="budget_exhausted", success=None)
            # Applied last so no later label can mask it, and sticky so the other
            # lanes stop dispatching immediately rather than after their episode.
            if account_failure(calls):
                result.update(
                    status="infrastructure_error", success=None, failure_reason="account_balance"
                )
                stopped.set()
            if any(
                c.get("failure_reason") == "invalid_provider_evidence"
                or tier_evidence_errors(spec, [c])
                for c in calls
            ):
                result.update(
                    status="infrastructure_error",
                    success=None,
                    failure_reason="unverified_or_invalid_provider_evidence",
                )
                stopped.set()
            for role, key in [("agent", "agent_cost_usd"), ("auxiliary", "auxiliary_cost_usd")]:
                selected = [c for c in calls if c["role"] == role]
                unsettled = any(
                    x["episode_id"] == e["episode_id"]
                    and x["role"] == role
                    and x.get("attempt_id") == attempt_id
                    and not x["settled"]
                    for x in gateway.guard.entries
                )
                result[key] = (
                    sum(c["cost_usd"] for c in selected)
                    if not unsettled and all(c.get("cost_usd") is not None for c in selected)
                    else None
                )
            result.update(
                episode_id=e["episode_id"],
                spec_sha256=study["spec_sha256"],
                sandbox_cost_usd=None,
                attempt_id=attempt_id,
                artifact_dir=str(output.relative_to(directory)),
                worker_log=str(log_path.relative_to(directory)),
            )
            result["execution_mode"] = execution_mode
            with rows_lock:
                rows.append(result)
                with rows_path.open("a") as f:
                    f.write(json.dumps(result) + "\n")
                (directory / "summary.json").write_text(
                    json.dumps(summarize(spec, rows), indent=2) + "\n"
                )
            print(
                "END",
                e["episode_id"],
                result["status"],
                result["success"],
                result["agent_cost_usd"],
                flush=True,
            )
            # A host can accept a cache marker on an endpoint that does not cache and
            # return zero cached tokens without an error, which reads as an expensive
            # tier rather than an uncacheable API. Stop the run instead of paying for
            # a whole study at uncached rates, once enough calls have asked to be sure.
            failures = cache_failures(calls_all, CACHE_GATE_MIN_CALLS)
            if failures:
                stopped.set()
                raise RuntimeError(
                    "prompt caching requested and never observed, or usage missing: "
                    + json.dumps(failures)
                )
            if result["status"] == "infrastructure_error":
                stopped.set()
                print("Stopping to repair infrastructure before further paid episodes.", flush=True)
                raise RuntimeError("episode infrastructure failure")

        if (args.parallel_routes or args.lanes > 1) and args.stage == "run":
            amendment = {
                "recorded_at": datetime.now(UTC).isoformat(),
                "spec_sha256": study["spec_sha256"],
                "execution_mode": execution_mode,
                "max_active_agent_episodes_per_route": args.lanes,
                "max_active_coding_episodes": 1,
                "max_routes": len(spec["models"]),
                "auxiliary_calls": "simulators may overlap across routes",
                "already_recorded_episodes": sorted(completed),
                "stop_starting_episodes_at": args.stop_at,
                "reason": (
                    f"Independent routes overlap with {args.lanes} episode(s) per route; "
                    "both tiers of a task stay in one lane; "
                    "coding containers remain serialized."
                ),
            }
            with (directory / "execution-amendments.jsonl").open("a") as f:
                f.write(json.dumps(amendment) + "\n")

            def run_unit(unit):
                for e in unit:
                    with coding_lock if e["suite"] == "coding" else nullcontext():
                        if stopped.is_set():
                            return
                        try:
                            execute(e)
                        except Exception:
                            stopped.set()
                            raise

            def run_route(route):
                units = work_units([x for x in episodes if x["route"] == route])
                if args.lanes == 1:
                    for unit in units:
                        run_unit(unit)
                    return
                with ThreadPoolExecutor(max_workers=args.lanes) as lanes:
                    for future in [lanes.submit(run_unit, unit) for unit in units]:
                        future.result()

            if args.parallel_routes:
                with ThreadPoolExecutor(max_workers=len(spec["models"])) as pool:
                    futures = [pool.submit(run_route, m["id"]) for m in spec["models"]]
                    for future in futures:
                        future.result()
            else:
                # Lanes alone: one route at a time, concurrent episodes inside it.
                for model in spec["models"]:
                    run_route(model["id"])
        else:
            for e in episodes:
                if stopped.is_set():
                    break
                execute(e)
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        runner_lock.close()


if __name__ == "__main__":
    main()
