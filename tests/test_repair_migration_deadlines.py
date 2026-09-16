"""Offline integration tests for the migration deadline repair."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts/repair_migration_deadlines.py"
_SPEC = importlib.util.spec_from_file_location("repair_migration_deadlines", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
repair = _MODULE.repair


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _ledger_rows(path: Path) -> list[tuple]:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT * FROM calls ORDER BY call_id").fetchall()


def _create_study(tmp_path: Path, *, judge_latency: float = 120.0) -> tuple[Path, dict]:
    root = tmp_path / "study"
    root.mkdir()
    spec = {"per_call_deadline_s": 300, "frozen": "fixture"}
    (root / "spec.json").write_text(json.dumps(spec))
    (root / "manifest.json").write_text(json.dumps({"spec_sha256": _digest(spec)}))
    (root / "runner.lock").touch()

    ledger_path = root / "ledger.sqlite"
    with sqlite3.connect(ledger_path) as conn:
        conn.execute(
            """CREATE TABLE calls (
                call_id TEXT PRIMARY KEY,
                route_id TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_s REAL NOT NULL,
                cost_usd TEXT,
                result_json TEXT,
                response_artifact TEXT
            )"""
        )
        conn.executemany(
            "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "candidate/baseline/case-1/openrouter-modal/method/0",
                    "openrouter-modal",
                    "completed",
                    317.128,
                    "0.012345",
                    '{"raw":"candidate-response"}',
                    "calls/candidate.response.json",
                ),
                (
                    "candidate/baseline/case-2/openrouter-modal/method/0",
                    "openrouter-modal",
                    "completed",
                    299.999,
                    "0.010000",
                    '{"raw":"candidate-two"}',
                    "calls/candidate-two.response.json",
                ),
                (
                    "judge/judge-call",
                    "judge-glm",
                    "completed",
                    judge_latency,
                    "0.001000",
                    '{"raw":"paid-judge-response"}',
                    "calls/judge.response.json",
                ),
            ],
        )

    call = {
        "call_id": "candidate/baseline/case-1/openrouter-modal/method/0",
        "route_id": "openrouter-modal",
        "latency_s": 317.128,
        "cost_usd": 0.012345,
        "cost_kind": "reported",
        "raw": {"model": "fixture", "usage": {"input_tokens": 100, "output_tokens": 20}},
        "response_artifact": "calls/candidate.response.json",
    }
    affected = {
        "case_id": "case-1",
        "role": "mobile",
        "route": "openrouter-modal",
        "stage": "baseline",
        "methodology_sha256": "method",
        "call_id": call["call_id"],
        "call": call,
        "status": "valid",
        "output": {
            "scores": {"criterion": 3},
            "recommendation": "MAYBE",
            "rationale": "Fixture output.",
        },
    }
    unaffected = {
        **affected,
        "case_id": "case-2",
        "call_id": "candidate/baseline/case-2/openrouter-modal/method/0",
        "call": {**call, "latency_s": 299.999},
    }
    outcomes = root / "outcomes"
    outcomes.mkdir()
    for outcome in (affected, unaffected):
        (outcomes / f"{_digest(outcome['call_id'])}.json").write_text(json.dumps(outcome))

    paid_reference = {
        "status": "graded",
        "score": 0.75,
        "call_id": "judge/judge-call",
        "call": {"cost_usd": 0.001, "raw": {"model": "judge-fixture"}},
    }
    stale_grade = {
        "status": "graded",
        "score": 1.0,
        "acceptable": True,
        "critical": False,
        "feedback": "stale",
    }
    progress = {
        "completed": 2,
        "rows": [
            {
                "case_id": "case-1",
                "role": "mobile",
                "route": "openrouter-modal",
                "outcome": affected,
                "grade": stale_grade,
                "reference": paid_reference,
            },
            {
                "case_id": "case-2",
                "role": "mobile",
                "route": "openrouter-modal",
                "outcome": unaffected,
                "grade": stale_grade,
                "reference": paid_reference,
            },
        ],
    }
    (root / "baseline-progress.json").write_text(json.dumps(progress))
    grades = root / "grades"
    grades.mkdir()
    (grades / "paid-judge.json").write_text(json.dumps(paid_reference))
    return root, affected


def test_repairs_overdue_candidate_and_progress_without_touching_ledger_or_judges(
    tmp_path: Path,
) -> None:
    root, original = _create_study(tmp_path)
    ledger_before = _ledger_rows(root / "ledger.sqlite")
    judge_path = root / "grades/paid-judge.json"
    judge_before = judge_path.read_bytes()
    affected_path = root / "outcomes" / f"{_digest(original['call_id'])}.json"

    audit = repair(root)

    assert audit["status"] == "repaired"
    assert audit["deadline_s"] == 300.0
    assert audit["offline_only"] is True
    assert audit["counts"] == {
        "candidate_outcomes_scanned": 2,
        "candidate_outcomes_reclassified": 1,
        "baseline_rows_rebuilt": 1,
        "overdue_completed_judges": 0,
        "historical_judge_artifacts_modified": 0,
    }
    repaired = json.loads(affected_path.read_text())
    assert repaired["status"] == "delivery_failure"
    assert repaired["reason"] == "deadline_exceeded"
    assert "output" not in repaired
    assert repaired["call"] == original["call"]

    progress = json.loads((root / "baseline-progress.json").read_text())
    changed_row, unchanged_row = progress["rows"]
    assert changed_row["outcome"] == repaired
    assert changed_row["grade"] == {
        "status": "model_failure",
        "score": 0.0,
        "acceptable": False,
        "critical": False,
        "feedback": "delivery_failure",
    }
    assert changed_row["reference"]["call_id"] == "judge/judge-call"
    assert unchanged_row["outcome"]["status"] == "valid"
    assert unchanged_row["grade"]["status"] == "graded"

    assert _ledger_rows(root / "ledger.sqlite") == ledger_before
    assert judge_path.read_bytes() == judge_before
    backup = root / "amendments/pre-deadline-repair"
    assert (backup / "ledger.sqlite").is_file()
    assert _ledger_rows(backup / "ledger.sqlite") == ledger_before
    assert json.loads((backup / affected_path.relative_to(root)).read_text()) == original
    assert (backup / "baseline-progress.json").is_file()
    assert json.loads((backup / "repair.json").read_text()) == audit
    assert audit["before"]["files"] != audit["after"]["files"]

    repeated = repair(root)
    assert repeated["status"] == "already_repaired"
    assert _ledger_rows(root / "ledger.sqlite") == ledger_before


def test_blocks_completed_overdue_judge_before_creating_archive_or_changes(tmp_path: Path) -> None:
    root, original = _create_study(tmp_path, judge_latency=300.0)
    affected_path = root / "outcomes" / f"{_digest(original['call_id'])}.json"
    before = affected_path.read_bytes()

    with pytest.raises(RuntimeError, match="completed judge calls exceed the deadline"):
        repair(root)

    assert affected_path.read_bytes() == before
    assert not (root / "amendments/pre-deadline-repair").exists()


def test_requires_exclusive_runner_lock(tmp_path: Path) -> None:
    root, _ = _create_study(tmp_path)

    with (root / "runner.lock").open("a") as active_lock:
        fcntl.flock(active_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="runner is active"):
            repair(root)


@pytest.mark.parametrize("forbidden", ["frozen-selection.json", "final-progress.json", "final.json"])
def test_refuses_after_selection_or_final_artifacts(tmp_path: Path, forbidden: str) -> None:
    root, _ = _create_study(tmp_path)
    (root / forbidden).write_text("{}")

    with pytest.raises(RuntimeError, match="must run before frozen selection/final"):
        repair(root)


def test_requires_the_exact_frozen_300_second_deadline(tmp_path: Path) -> None:
    root, _ = _create_study(tmp_path)
    spec_path = root / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec["per_call_deadline_s"] = 301
    spec_path.write_text(json.dumps(spec))
    (root / "manifest.json").write_text(json.dumps({"spec_sha256": _digest(spec)}))

    with pytest.raises(ValueError, match="requires frozen per_call_deadline_s=300"):
        repair(root)
