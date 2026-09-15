"""Budgeted GCP pilot controller. All inference, including simulators, is metered."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from compound.agentic_gateway import Gateway, missing_rates
from compound.agentic_study import plan, summarize, verify_sources


def attempt_calls(calls, episode_id, since):
    """Calls belonging to this attempt only.

    A re-run episode keeps its id, so the ledger still holds the calls of the
    attempt that was interrupted. Classifying or costing an episode from those
    stale rows misreads a clean re-run as a repeat of the old failure.
    """
    return [c for c in calls if c["episode_id"] == episode_id and c["started_at"] >= since]


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
        key = (episode["suite"], episode["task_id"], episode["trial"], episode["route"])
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
    gateway = Gateway(spec, study, directory, limit=args.inference_limit)
    server = gateway.serve()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        if args.stage == "probe":
            probe_path = directory / "probes.json"
            probes = json.loads(probe_path.read_text()) if probe_path.exists() else []
            for e in study["episodes"]:
                if not e["episode_id"].startswith("probe-") or any(
                    r["episode_id"] == e["episode_id"] for r in probes
                ):
                    continue
                try:
                    raw = gateway.call(
                        e["episode_id"],
                        "agent",
                        {
                            "messages": [
                                {"role": "user", "content": "Call the echo tool with value READY."}
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
                        },
                    )
                    tool_calls = raw["choices"][0]["message"].get("tool_calls") or []
                    ok = any(
                        c["function"]["name"] == "echo"
                        and json.loads(c["function"]["arguments"]).get("value") == "READY"
                        for c in tool_calls
                    )
                    row = {"episode_id": e["episode_id"], "ok": ok, "reply": raw}
                except Exception as exc:
                    row = {"episode_id": e["episode_id"], "ok": False, "error": str(exc)}
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
        episodes = [e for e in episodes if e["episode_id"] not in completed][: args.count]
        rows_lock = threading.Lock()
        coding_lock = threading.Lock()
        stopped = threading.Event()

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
            output = directory / e["episode_id"]
            # An episode reaching here is not recorded, so anything left in its
            # directory belongs to an attempt that was interrupted. Upstream
            # harnesses prompt on stdin when they find their own artefacts, which
            # blocks forever under a non-interactive runner, so clear it first and
            # keep the remains for audit.
            if output.exists() and any(output.iterdir()):
                stale = directory / "interrupted-episodes" / e["episode_id"]
                stale.parent.mkdir(parents=True, exist_ok=True)
                if stale.exists():
                    shutil.rmtree(stale)
                shutil.move(str(output), str(stale))
            output.mkdir(parents=True, exist_ok=True)
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
            with (output / "worker.log").open("w") as log:
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
            calls = attempt_calls(calls, e["episode_id"], attempt_started)
            auxiliary_failed = any(c["status"] != 200 and c["role"] == "auxiliary" for c in calls)
            if (
                result["status"] == "infrastructure_error"
                and not auxiliary_failed
                and any(c["status"] != 200 and c["role"] == "agent" for c in calls)
            ):
                result["status"] = "provider_error"
            if auxiliary_failed:
                result.update(status="infrastructure_error", success=None, failure_role="auxiliary")
            if account_failure(calls):
                result.update(
                    status="infrastructure_error", success=None, failure_reason="account_balance"
                )
            stops_path = directory / "budget-stops.jsonl"
            stops = (
                [json.loads(line) for line in stops_path.read_text().splitlines()]
                if stops_path.exists()
                else []
            )
            if any(s["episode_id"] == e["episode_id"] and s["role"] == "agent" for s in stops):
                result.update(status="budget_exhausted", success=None)
            for role, key in [("agent", "agent_cost_usd"), ("auxiliary", "auxiliary_cost_usd")]:
                selected = [c for c in calls if c["role"] == role]
                unsettled = any(
                    x["episode_id"] == e["episode_id"] and x["role"] == role and not x["settled"]
                    for x in gateway.guard.entries
                )
                result[key] = (
                    sum(c["cost_usd"] for c in selected)
                    if not unsettled and all(c.get("cost_usd") is not None for c in selected)
                    else None
                )
            result.update(
                episode_id=e["episode_id"], spec_sha256=study["spec_sha256"], sandbox_cost_usd=None
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
            if result["status"] == "infrastructure_error":
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
        server.shutdown()
        server.server_close()
        runner_lock.close()


if __name__ == "__main__":
    main()
