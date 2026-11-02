"""Grade episodes, check the simulator is faithful, and decide a verdict.

**Grading.** An episode passes when it ends normally and performs the same set
of actions (tool calls by name and arguments) as the real conversation did.
That is the tau-bench "actions" criterion. It is strict on purpose: an extra
refund or a missing lookup is a different outcome for the business, whatever
the final text says. Recall and precision are kept alongside, so a near miss
is visible. A text grader (a calibrated judge) can be added per task later;
it is not assumed here.

**Fidelity.** Before any comparison, the production model is run through the
same simulator. If it cannot reproduce its own real conversations, the
simulator (or the replay policy) is not good enough for this task, and the
verdict is ``simulator_unfaithful`` instead of a number that would mislead.

**Verdict.** The candidate and the production model are compared scenario by
scenario on pass or fail. A paired bootstrap gives an interval for the
difference in pass rate, and a non-inferiority margin, declared before the run,
decides: ``meets_gate`` when even the low end is above minus the margin,
``fails_gate`` when even the high end is below it, ``insufficient_data``
otherwise. The names match the TypeScript gate.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from compound.eval.simulation.episode import Episode
from compound.eval.simulation.scenario import Scenario

BOOTSTRAP = 2000


@dataclass(frozen=True)
class Grade:
    passed: bool
    recall: float | None
    precision: float | None
    reason: str


def grade(episode: Episode, scenario: Scenario) -> Grade:
    want, got = Counter(scenario.reference_actions), Counter(episode.actions)
    overlap = sum((want & got).values())
    recall = overlap / sum(want.values()) if want else None
    precision = overlap / sum(got.values()) if got else None
    if episode.stop != "user_done":
        return Grade(False, recall, precision, f"stopped: {episode.stop}")
    if want != got:
        return Grade(False, recall, precision, "different actions")
    return Grade(True, recall, precision, "same actions")


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fidelity(passes: list[bool], threshold: float, min_n: int) -> dict[str, Any]:
    n, k = len(passes), sum(passes)
    low, high = wilson(k, n)
    rate = k / n if n else None
    return {
        "scenarios": n,
        "reproduced": k,
        "rate": rate,
        "ci95": [low, high],
        "threshold": threshold,
        "faithful": n >= min_n and rate is not None and rate >= threshold,
    }


def paired_interval(diffs: list[int], seed: int = 11) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    point = sum(diffs) / n
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return point, stats[int(0.025 * BOOTSTRAP)], stats[int(0.975 * BOOTSTRAP) - 1]


def decide(
    candidate: list[bool],
    incumbent: list[bool],
    *,
    margin: float,
    fidelity_report: dict[str, Any],
    min_n: int,
) -> dict[str, Any]:
    if len(candidate) != len(incumbent):
        raise ValueError("candidate and incumbent must cover the same scenarios")
    out: dict[str, Any] = {
        "scenarios": len(candidate),
        "margin": margin,
        "candidate_pass_rate": sum(candidate) / len(candidate) if candidate else None,
        "incumbent_pass_rate": sum(incumbent) / len(incumbent) if incumbent else None,
        "fidelity": fidelity_report,
    }
    if not fidelity_report["faithful"]:
        out["verdict"] = "simulator_unfaithful"
        return out
    if len(candidate) < min_n:
        out["verdict"] = "insufficient_data"
        return out
    point, low, high = paired_interval([int(c) - int(i) for c, i in zip(candidate, incumbent,
                                                                        strict=True)])
    out["difference"] = {"point": point, "ci95": [low, high]}
    if low >= -margin:
        out["verdict"] = "meets_gate"
    elif high < -margin:
        out["verdict"] = "fails_gate"
    else:
        out["verdict"] = "insufficient_data"
    return out
