"""Provider sweep over tau-bench: one model, many pinned upstream hosts.

Runs the REAL interactive tau2 protocol (agent + user simulator + live domain
tools) once per (model x upstream) config from a declared sweep spec, so every
row of the resulting cost/latency/quality table is the official benchmark
reward, not a derived slice.

Money safety mirrors the TS engine's stance: nothing is spent without an
explicit ``--run``; the default ``--estimate`` prints the declared-price cost
sheet and exits. Prices live in the sweep spec — declared decision context,
never scraped at run time.

Usage:
    uv run python -m compound.tau_sweep --estimate
    uv run python -m compound.tau_sweep --run [--only kimi] [--trials 1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from compound.adapters.tau import TauModel, run_tau_partition, task_ids_by_domain

DEFAULT_SPEC = Path("benchmarks/sweeps/tau-provider-sweep.json")
DEFAULT_OUTPUT = Path("artifacts/tau-sweep")

# Per-episode token averages measured on the recorded smoke episode
# (artifacts/smoke/tau, max_steps=18), scaled to the sweep's max_steps=30.
# These are ESTIMATOR inputs only; actual spend is whatever the run bills.
SMOKE_STEPS = 18
SMOKE_AGENT_IN, SMOKE_AGENT_OUT = 14_009, 296
SMOKE_USER_IN, SMOKE_USER_OUT = 3_182, 178


@dataclass(frozen=True, slots=True)
class SweepConfig:
    model: str
    upstream: str
    quant: str
    usd_in_per_m: float
    usd_out_per_m: float
    provider: str = "openrouter"
    service_tier: str | None = None
    #: Custom OpenAI-compatible host: endpoint + env var holding its key.
    api_base: str | None = None
    api_key_env: str | None = None

    @property
    def label(self) -> str:
        return f"{self.model}@{self.upstream}"

    def required_key_env(self) -> str:
        if self.provider == "openrouter":
            return "OPENROUTER_API_KEY"
        if self.api_key_env:
            return self.api_key_env
        if self.provider == "doubleword":
            return "DOUBLEWORD_API_KEY"
        raise ValueError(
            f"config {self.label!r}: provider {self.provider!r} needs api_key_env"
        )


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    spec["configs"] = [
        SweepConfig(
            model=c["model"],
            upstream=c["upstream"],
            quant=c.get("quant") or "unknown",
            usd_in_per_m=float(c["usd_in_per_m"]),
            usd_out_per_m=float(c["usd_out_per_m"]),
            provider=c.get("provider", "openrouter"),
            service_tier=c.get("service_tier"),
            api_base=c.get("api_base"),
            api_key_env=c.get("api_key_env"),
        )
        for c in spec["configs"]
    ]
    return spec


def episode_count(spec: dict, trials: int) -> int:
    grouped = task_ids_by_domain(spec["manifest"], None)
    return sum(len(ids) for ids in grouped.values()) * trials


def estimate(spec: dict, trials: int) -> float:
    """Print the per-config cost sheet from declared prices; return the total."""
    episodes = episode_count(spec, trials)
    scale = float(spec.get("max_steps", 30)) / SMOKE_STEPS
    agent_in, agent_out = SMOKE_AGENT_IN * scale, SMOKE_AGENT_OUT * scale
    user_in, user_out = SMOKE_USER_IN * scale, SMOKE_USER_OUT * scale
    user_price = spec["user_model"]
    user_usd = (
        user_in * float(user_price["usd_in_per_m"]) + user_out * float(user_price["usd_out_per_m"])
    ) / 1e6

    print(f"episodes per config: {episodes} (tasks x {trials} trial(s)), max_steps scale x{scale:.2f}")
    print(f"user simulator: {user_price['model']} ~ ${user_usd * episodes:.2f} per config")
    print()
    print(f"{'config':52s} {'quant':8s} {'$/episode':>10s} {'$/config':>9s}")
    total = 0.0
    for c in spec["configs"]:
        agent_usd = (agent_in * c.usd_in_per_m + agent_out * c.usd_out_per_m) / 1e6
        per_config = (agent_usd + user_usd) * episodes
        total += per_config
        print(f"{c.label:52s} {c.quant:8s} {agent_usd + user_usd:>10.4f} {per_config:>9.2f}")
    print(f"\nTOTAL (declared prices, smoke-derived volumes): ${total:.2f}")
    print("Actual spend varies with episode length; treat this as a ceiling-setting estimate.")
    return total


def run(
    spec: dict,
    trials: int,
    only: str | None,
    output_dir: Path,
    domains: set[str] | None = None,
    shard: tuple[int, int] | None = None,
) -> int:
    user_spec = spec["user_model"]
    user_model = TauModel(
        provider=user_spec["provider"],
        model=user_spec["model"],
        max_tokens=user_spec.get("max_tokens"),
    )
    configs = [
        c for c in spec["configs"] if only is None or only.lower() in c.label.lower()
    ]
    if shard is not None:
        index, count = shard
        configs = [c for i, c in enumerate(configs) if i % count == index]
    if not configs:
        print(f"error: --only {only!r} matches no config", file=sys.stderr)
        return 2
    missing = {
        c.required_key_env() for c in configs if not os.getenv(c.required_key_env())
    }
    if missing:
        print(f"error: missing env for --run: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    telemetry = output_dir / "telemetry.jsonl"
    ledger_path = output_dir / "sweep-ledger.json"
    ledger: list[dict] = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    done = {entry["label"] for entry in ledger if entry["status"] == "completed"}

    failures = 0
    for c in configs:
        if c.label in done:
            print(f"skip (completed): {c.label}")
            continue
        if c.provider == "openrouter":
            agent = TauModel(
                provider="openrouter",
                model=c.model,
                max_tokens=spec.get("agent_max_tokens"),
                openrouter_provider=c.upstream,
            )
        else:
            # doubleword or any custom OpenAI-compatible host from the spec.
            default_base = (
                spec.get("doubleword_api_base", "https://api.doubleword.ai/v1")
                if c.provider == "doubleword"
                else None
            )
            agent = TauModel(
                provider=c.provider,
                model=c.model,
                api_base=c.api_base or default_base,
                api_key_env=c.api_key_env,
                max_tokens=spec.get("agent_max_tokens"),
                service_tier=c.service_tier,
            )
        print(f"run: {c.label}")
        started = time.time()
        try:
            run_tau_partition(
                manifest_path=spec["manifest"],
                partition=None,
                agent_model=agent,
                user_model=user_model,
                candidate_instruction="",
                trials=trials,
                max_steps=int(spec.get("max_steps", 30)),
                output_dir=output_dir / "episodes",
                telemetry_path=telemetry.resolve(),
                task_split_overrides=spec.get("task_splits"),
                domains=domains,
            )
            status = "completed"
        except Exception as error:  # noqa: BLE001 — one bad host must not kill the sweep
            print(f"FAILED {c.label}: {error}", file=sys.stderr)
            status = f"failed: {error}"
            failures += 1
        ledger.append(
            {
                "label": c.label,
                "status": status,
                "trials": trials,
                "seconds": round(time.time() - started, 1),
                "declared_usd_in_per_m": c.usd_in_per_m,
                "declared_usd_out_per_m": c.usd_out_per_m,
                "quant": c.quant,
            }
        )
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=1))
    print(f"\nsweep done: {len(configs) - failures}/{len(configs)} configs completed")
    return 1 if failures else 0


def repair(output_dir: Path) -> int:
    """Scrub infrastructure-errored episodes so a rerun replays only them.

    tau's resume skips any (trial, task_id, seed) already present in a
    results.json — including episodes that died on a network outage. Dropping
    those simulations from the file makes the next --run re-execute exactly the
    scrubbed episodes; healthy ones are never re-billed. The ledger is reset so
    every config gets revisited (clean ones fast-forward through resume).
    """
    episodes = output_dir / "episodes"
    scrubbed_files = 0
    scrubbed_sims = 0
    for results in sorted(episodes.glob("*.json/results.json")):
        payload = json.loads(results.read_text())
        sims = payload.get("simulations", [])
        keep = [s for s in sims if s.get("termination_reason") != "infrastructure_error"]
        if len(keep) == len(sims):
            continue
        payload["simulations"] = keep
        results.write_text(json.dumps(payload))
        scrubbed_files += 1
        scrubbed_sims += len(sims) - len(keep)
    ledger_path = output_dir / "sweep-ledger.json"
    if ledger_path.exists():
        ledger_path.unlink()
    print(f"repair: dropped {scrubbed_sims} infra-errored episodes across {scrubbed_files} files")
    print("repair: ledger reset — next --run revisits every config, resuming what survived")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--trials", type=int, default=None, help="override spec trials")
    parser.add_argument("--only", help="substring filter on model@upstream labels")
    parser.add_argument("--domains", help="comma-separated domain filter (e.g. airline,retail)")
    parser.add_argument("--shard", help="i/n stripe over the filtered configs (e.g. 0/2)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="print cost sheet (default)")
    mode.add_argument("--run", action="store_true", help="execute the sweep (spends money)")
    mode.add_argument(
        "--repair",
        action="store_true",
        help="scrub infra-errored episodes + reset ledger so --run replays only them",
    )
    args = parser.parse_args()

    if args.repair:
        return repair(args.output)
    spec = load_spec(args.spec)
    trials = args.trials if args.trials is not None else int(spec.get("trials", 1))
    if args.run:
        domains = set(args.domains.split(",")) if args.domains else None
        shard = None
        if args.shard:
            index, count = args.shard.split("/")
            shard = (int(index), int(count))
        return run(spec, trials, args.only, args.output, domains=domains, shard=shard)
    estimate(spec, trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
