import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from compound import agentic_run


def test_parallel_execution_serializes_each_route_and_coding(tmp_path, monkeypatch):
    spec_path = Path("benchmarks/flex-agentic/pilot.json").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "parent-only-test-key")
    (tmp_path / "keys.json").write_text(json.dumps({"OPENROUTER_API_KEY": "parent-only-test-key"}))
    monkeypatch.setattr(agentic_run, "verify_sources", lambda spec: [])
    monkeypatch.setattr(
        agentic_run.Gateway,
        "serve",
        lambda self: SimpleNamespace(
            server_port=1, shutdown=lambda: None, server_close=lambda: None
        ),
    )
    lock = threading.Lock()
    active = Counter()
    maximum = Counter()

    class Process:
        def __init__(self, cmd, *, env, **kwargs):
            assert "OPENROUTER_API_KEY" not in env
            assert "DOUBLEWORD_API_KEY" not in env
            self.episode = json.loads(cmd[cmd.index("--episode") + 1])
            self.output = Path(cmd[cmd.index("--out") + 1])
            self.labels = ["all", "route:" + self.episode["route"]]
            if self.episode["suite"] == "coding":
                self.labels.append("coding")
            with lock:
                for label in self.labels:
                    active[label] += 1
                    maximum[label] = max(maximum[label], active[label])

        def wait(self, timeout):
            time.sleep(0.005)
            (self.output / "outcome.json").write_text(
                json.dumps(
                    {
                        "status": "graded",
                        "success": True,
                        "duration_s": 0.005,
                        "grader": "fake@1",
                    }
                )
            )
            with lock:
                for label in self.labels:
                    active[label] -= 1
            return 0

    monkeypatch.setattr(agentic_run.subprocess, "Popen", Process)
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "run", "--spec", str(spec_path), "--count", "30", "--parallel-routes", "--go"],
    )
    agentic_run.main()
    rows = [
        json.loads(x)
        for x in Path("artifacts/flex-agentic-gcp/outcomes.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len({r["episode_id"] for r in rows}) == 30
    assert maximum["all"] > 1
    assert maximum["coding"] == 1
    assert all(n == 1 for label, n in maximum.items() if label.startswith("route:"))
    assert all(r["execution_mode"] == "parallel_routes" for r in rows)
    assert json.loads(Path("artifacts/flex-agentic-gcp/summary.json").read_text())["recorded"] == 30


def test_work_units_keep_both_tiers_of_a_task_in_one_unit():
    plan = [
        {"suite": "retail", "task_id": "23", "trial": 0, "route": "a", "tier": "standard"},
        {"suite": "retail", "task_id": "23", "trial": 0, "route": "a", "tier": "flex"},
        {"suite": "retail", "task_id": "30", "trial": 1, "route": "a", "tier": "flex"},
        {"suite": "retail", "task_id": "30", "trial": 1, "route": "a", "tier": "standard"},
        {"suite": "retail", "task_id": "23", "trial": 1, "route": "a", "tier": "standard"},
    ]
    units = agentic_run.work_units(plan)
    assert [len(u) for u in units] == [2, 2, 1]
    for unit in units:
        assert len({(e["task_id"], e["trial"]) for e in unit}) == 1
        assert len({e["tier"] for e in unit}) == len(unit)
    # Every planned episode survives grouping exactly once, in order.
    assert [e for unit in units for e in unit] == plan


def test_a_task_that_repeats_later_is_not_merged_into_one_unit():
    plan = [
        {"suite": "retail", "task_id": "23", "trial": 0, "route": "a", "tier": "standard"},
        {"suite": "retail", "task_id": "99", "trial": 0, "route": "a", "tier": "standard"},
        {"suite": "retail", "task_id": "23", "trial": 0, "route": "a", "tier": "flex"},
    ]
    assert [len(u) for u in agentic_run.work_units(plan)] == [1, 1, 1]


def test_lanes_overlap_episodes_inside_one_route_without_exceeding_the_lane_count(
    tmp_path, monkeypatch
):
    spec_path = Path("benchmarks/flex-agentic/tier-equivalence.json").resolve()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "keys.json").write_text(json.dumps({"DOUBLEWORD_API_KEY": "parent-only-test-key"}))
    monkeypatch.setattr(agentic_run, "verify_sources", lambda spec: [])
    monkeypatch.setattr(
        agentic_run.Gateway,
        "serve",
        lambda self: SimpleNamespace(
            server_port=1, shutdown=lambda: None, server_close=lambda: None
        ),
    )
    lock = threading.Lock()
    active = Counter()
    maximum = Counter()
    lane_of_pair = {}

    class Process:
        def __init__(self, cmd, *, env, **kwargs):
            assert "DOUBLEWORD_API_KEY" not in env
            self.episode = json.loads(cmd[cmd.index("--episode") + 1])
            self.output = Path(cmd[cmd.index("--out") + 1])
            with lock:
                active["route"] += 1
                maximum["route"] = max(maximum["route"], active["route"])
                key = (self.episode["task_id"], self.episode["trial"])
                lane_of_pair.setdefault(key, []).append(threading.get_ident())

        def wait(self, timeout):
            time.sleep(0.005)
            (self.output / "outcome.json").write_text(
                json.dumps(
                    {"status": "graded", "success": True, "duration_s": 0.005, "grader": "fake@1"}
                )
            )
            with lock:
                active["route"] -= 1
            return 0

    monkeypatch.setattr(agentic_run.subprocess, "Popen", Process)
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "run", "--spec", str(spec_path), "--count", "24", "--lanes", "4", "--go"],
    )
    agentic_run.main()
    rows = [
        json.loads(x)
        for x in Path("artifacts/flex-agentic-gcp/outcomes.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len({r["episode_id"] for r in rows}) == 24
    assert 1 < maximum["route"] <= 4
    assert all(r["execution_mode"] == "lanes_4" for r in rows)
    # Both tiers of a task ran in the same lane, so a pair never straddles two.
    assert all(len(set(threads)) == 1 for threads in lane_of_pair.values() if len(threads) == 2)
    amendments = [
        json.loads(x)
        for x in (
            Path("artifacts/flex-agentic-gcp/execution-amendments.jsonl").read_text().splitlines()
        )
    ]
    assert amendments[-1]["max_active_agent_episodes_per_route"] == 4


def test_an_account_balance_failure_is_terminal_not_a_per_episode_provider_error():
    """A 402 dooms every later episode, so it must halt the run rather than be counted."""
    assert agentic_run.account_failure([{"status": 402, "role": "agent"}])
    assert agentic_run.account_failure([{"status": 200}, {"status": 402}])
    # Ordinary provider faults stay per-episode and must not halt anything.
    assert not agentic_run.account_failure([{"status": 502}, {"status": 429}, {"status": 200}])
    assert not agentic_run.account_failure([])


def test_only_the_current_attempts_calls_are_classified_or_costed():
    """A re-run episode must not inherit the failure of the attempt it replaces."""
    ledger = [
        {"episode_id": "a", "started_at": "2026-09-15T10:00:00+00:00", "status": 402},
        {"episode_id": "a", "started_at": "2026-09-15T12:00:00+00:00", "status": 200},
        {"episode_id": "b", "started_at": "2026-09-15T12:00:00+00:00", "status": 402},
    ]
    current = agentic_run.attempt_calls(ledger, "a", "2026-09-15T11:00:00+00:00")
    assert [c["status"] for c in current] == [200]
    assert not agentic_run.account_failure(current)
    # The stale attempt is still visible when the whole history is asked for.
    assert agentic_run.account_failure(
        agentic_run.attempt_calls(ledger, "a", "2026-09-15T09:00:00+00:00")
    )
