"""Offline evidence gates for the paid agentic runner."""

import hashlib
import json
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

from compound.agentic_study import fingerprint
from compound.serving_metrics import cache_effectiveness


def runtime_identity():
    return {
        "python": sys.version,
        "packages": sorted(
            [d.metadata["Name"], d.version] for d in distributions() if d.metadata["Name"]
        ),
    }


def preparation_errors(spec, directory):
    """Require exact reference coverage and pinned clean harnesses before paid probes."""
    errors = []
    if "retail" in spec["sources"]:
        path = directory / "reference-qualification.json"
        evidence = json.loads(path.read_text()) if path.exists() else {}
        rows = evidence.get("results", [])
        expected = {t["id"] for t in spec["sources"]["retail"]["tasks"]}
        if (
            evidence.get("spec_sha256") != fingerprint(spec)
            or evidence.get("suite") != "retail"
            or evidence.get("ok") is not True
            or len(rows) != len(expected)
            or {r.get("task_id") for r in rows} != expected
            or any(
                r.get("state_changed") is not True or r.get("gold_replay_errors") != []
                for r in rows
            )
        ):
            errors.append("retail reference qualification missing, failed, or from another spec")
    if spec.get("controls", {}).get("require_runtime_pins"):
        if "retail" in spec["sources"] and evidence.get("runtime") != runtime_identity():
            errors.append("reference qualification belongs to a different Python runtime")
        for suite, source in spec["sources"].items():
            root = next(
                (p for p in Path(source["data_path"]).resolve().parents if (p / ".git").exists()),
                None,
            )
            if root is None:
                errors.append(f"{suite}: harness checkout unavailable")
                continue
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if head != source.get("revision") or dirty:
                errors.append(f"{suite}: harness revision differs or tracked files are dirty")
    return errors


def cache_scope(call):
    return tuple(
        call.get(k)
        for k in (
            "route",
            "requested_model",
            "requested_endpoint",
            "request_url",
            "requested_tier",
            "role",
        )
    )


def cache_failures(calls, minimum=8):
    """Check recent calls separately, including missing usage as missing evidence."""
    groups = {}
    for call in calls:
        if call.get("cache_requested"):
            groups.setdefault(cache_scope(call), []).append(call)
    failures = []
    for scope, group in groups.items():
        if len(group) < minimum:
            continue
        asked = group[-minimum:]
        verdict = cache_effectiveness(asked)
        if verdict["cells"] != minimum or not verdict["cache_observed_cells"]:
            failures.append({"scope": scope, "reason": "missing usage or no recent cache hits"})
    return failures


def seal_run(directory, spec, study, execution_mode):
    """Bind resumes before credentials/network; never adopt an unsealed old ledger."""
    source = Path(__file__).parent
    paths = [
        source / name
        for name in (
            "agentic_safety.py",
            "agentic_run.py",
            "agentic_study.py",
            "agentic_gateway.py",
            "agentic_worker.py",
            "cache_policy.py",
        )
    ]
    lock = source.parents[1] / "uv.lock"
    if lock.exists():
        paths.append(lock)
    manifest = {
        "spec_sha256": study["spec_sha256"],
        "execution_mode": execution_mode,
        "runtime_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "runtime": runtime_identity(),
    }
    target = directory / "run-manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError("run identity changed; use a new output directory")
        if json.loads((directory / "plan.json").read_text()) != study:
            raise ValueError("saved plan differs from requested study")
        if json.loads((directory / "spec.json").read_text()) != spec:
            raise ValueError("saved specification differs")
        return
    if any(
        (directory / name).exists()
        for name in (
            "calls.jsonl",
            "outcomes.jsonl",
            "spend.json",
            "plan.json",
            "spec.json",
        )
    ):
        raise ValueError("existing unsealed experiment; preserve it and use a new directory")
    for name, data in (("spec.json", spec), ("plan.json", study), ("run-manifest.json", manifest)):
        with (directory / name).open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")


def tier_evidence_errors(spec, calls):
    """Separate requested-policy studies from verified-serving studies."""
    policy = spec.get("controls", {}).get("tier_evidence_policy", "verified")
    if policy not in {"verified", "requested_policy"}:
        raise ValueError("unknown tier evidence policy")
    errors = []
    for call in calls:
        if call.get("status") != 200:
            continue
        expected = {"flex"} if call.get("requested_tier") == "flex" else {"priority", "default"}
        served = call.get("served_tier")
        if served is not None and served not in expected:
            errors.append("contradictory tier echo")
        elif policy == "verified" and call.get("tier_confirmed") is not True:
            errors.append("served tier unverified")
    return errors


def qualification_errors(spec, directory):
    """A probe must demonstrate tools, cache usage, and the declared tier policy.

    Missing echoes never become confirmations. Requested-policy studies may
    proceed without them; verified-serving studies still require evidence.
    """
    path = directory / "calls.jsonl"
    calls = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    probe_path = directory / "probes.json"
    probes = json.loads(probe_path.read_text()) if probe_path.exists() else []
    errors = []
    required = [(m, tier, "agent") for m in spec["models"] for tier in ("standard", "flex")]
    if "retail" in spec["sources"]:
        route = spec["controls"].get("simulator_route", "gpt-sol")
        models = spec["models"] + spec.get("auxiliary_models", [])
        required.append((next(m for m in models if m["id"] == route), "standard", "auxiliary"))
    for model, tier, role in required:
        eid = (
            "probe-simulator-standard"
            if role == "auxiliary"
            else "probe-" + model["id"] + "-" + tier
        )
        rows = [
            c
            for c in calls
            if c.get("episode_id") == eid
            and c.get("role") == role
            and c.get("requested_model") == model["model"]
            and c.get("requested_tier") == tier
            and c.get("status") == 200
        ]
        matching = [
            p for p in probes if p.get("episode_id") == eid and p.get("role", "agent") == role
        ]
        latest = matching[-1] if matching else {}
        if latest.get("attempt_id"):
            rows = [c for c in rows if c.get("attempt_id") == latest["attempt_id"]]
        tool_ok = latest.get("ok") is True
        if len(rows) < 2 or not tool_ok or tier_evidence_errors(spec, rows):
            errors.append(f"{model['id']}/{tier}/{role}: tools or tier evidence unqualified")
        if not cache_effectiveness(rows)["cache_observed_cells"]:
            errors.append(f"{model['id']}/{tier}/{role}: no verified repeated-prefix cache hit")
    return errors
