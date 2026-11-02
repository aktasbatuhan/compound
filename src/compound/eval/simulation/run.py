"""Load scenarios, check the simulator on the production model, then grade candidates.

The order matters. The production model goes first, on the same scenarios and
the same simulator, and if it cannot reproduce its own real conversations the
run stops there with ``simulator_unfaithful``: no candidate is paid for when
the yardstick itself is bent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compound.eval.simulation.chat import ChatFn
from compound.eval.simulation.environment import Environment
from compound.eval.simulation.episode import SpendCap, dumps, run_episode
from compound.eval.simulation.scenario import Scenario, scenario_from_trace
from compound.eval.simulation.users import User
from compound.eval.simulation.verdict import decide, fidelity, grade


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    chat: ChatFn


def load_scenarios(
    paths: list[str | Path], *, task_key: str | None = None, default_policy: str = "recorded"
) -> list[Scenario]:
    """Interactive scenarios from ``compound.trace`` files (JSON, JSON array or JSONL)."""
    docs: list[dict[str, Any]] = []
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix in (".json", ".jsonl")) \
            if p.is_dir() else [p]
    for path in files:
        text = path.read_text()
        try:
            data = json.loads(text)
            docs += data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            docs += [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for doc in docs:
        if not isinstance(doc, dict) or doc.get("schema") != "compound.trace":
            continue
        if task_key and doc.get("task_key") != task_key:
            continue
        scenario = scenario_from_trace(doc, default_policy=default_policy)
        if scenario is not None and scenario.user_turns and scenario.interactive:
            out.append(scenario)
    return out


def run_arm(
    arm: Arm,
    scenarios: list[Scenario],
    make_user: Callable[[], User],
    *,
    cap: SpendCap,
    out: Path,
    max_model_calls: int,
    env_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        for scenario in scenarios:
            env = Environment.for_scenario(scenario, **(env_kwargs or {}))
            episode = run_episode(scenario, arm.chat, make_user(), env, model=arm.model,
                                  max_model_calls=max_model_calls, cap=cap)
            g = grade(episode, scenario)
            record = json.loads(dumps(episode))
            record.update({"arm": arm.name, "passed": g.passed, "recall": g.recall,
                           "precision": g.precision, "grade_reason": g.reason})
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)
    return rows


def simulate(
    scenarios: list[Scenario],
    incumbent: Arm,
    candidates: list[Arm],
    make_user: Callable[[], User],
    *,
    out_dir: Path,
    margin: float = 0.05,
    fidelity_threshold: float = 0.8,
    min_n: int = 10,
    max_usd: float = 10.0,
    max_model_calls: int = 24,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cap = SpendCap(max_usd)
    episodes = out_dir / "episodes.jsonl"
    base = run_arm(incumbent, scenarios, make_user, cap=cap, out=episodes,
                   max_model_calls=max_model_calls, env_kwargs=env_kwargs)
    report = fidelity([r["passed"] for r in base], fidelity_threshold, min_n)
    result: dict[str, Any] = {"incumbent": incumbent.name, "fidelity": report, "candidates": {},
                              "episodes": str(episodes)}
    for arm in candidates:
        if not report["faithful"]:
            result["candidates"][arm.name] = {"verdict": "simulator_unfaithful",
                                              "fidelity": report, "skipped": True}
            continue
        if cap.reached:
            result["candidates"][arm.name] = {"verdict": "not_run", "reason": "spend cap"}
            continue
        rows = run_arm(arm, scenarios, make_user, cap=cap, out=episodes,
                       max_model_calls=max_model_calls, env_kwargs=env_kwargs)
        verdict = decide([r["passed"] for r in rows], [r["passed"] for r in base],
                         margin=margin, fidelity_report=report, min_n=min_n)
        verdict["cost_usd"] = sum(r["cost_usd"] for r in rows)
        verdict["stops"] = _count(r["stop"] for r in rows)
        result["candidates"][arm.name] = verdict
    result["spent_usd"] = cap.spent
    (out_dir / "verdict.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def format_result(result: dict[str, Any]) -> str:
    f = result["fidelity"]
    rate = "-" if f["rate"] is None else f"{f['rate']:.0%}"
    lines = [
        f"fidelity: {result['incumbent']} reproduced {f['reproduced']} of {f['scenarios']} "
        f"real conversations ({rate}, needs {f['threshold']:.0%})"
        + ("" if f["faithful"] else ", so the simulator is not trusted for this task"),
    ]
    for name, v in result["candidates"].items():
        if v.get("verdict") in ("simulator_unfaithful", "not_run"):
            lines.append(f"{name}: {v['verdict']}")
            continue
        diff = v.get("difference")
        detail = "" if not diff else (
            f", difference {diff['point']:+.0%} (95% {diff['ci95'][0]:+.0%} to "
            f"{diff['ci95'][1]:+.0%}, margin {v['margin']:.0%})"
        )
        lines.append(
            f"{name}: {v['verdict']}, passed {v['candidate_pass_rate']:.0%} against "
            f"{v['incumbent_pass_rate']:.0%}{detail}, ${v['cost_usd']:.4f}"
        )
    lines.append(f"spent ${result['spent_usd']:.4f}; episodes in {result['episodes']}")
    return "\n".join(lines)
