"""Focused corruption fixtures for the independent migration result audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from compound.migration_grading import (
    accepted_grade,
    case_from_trace,
    judge_messages,
    parse_judge_grade,
    quality_score,
)

SCRIPT = Path(__file__).parents[1] / "scripts/audit_migration_results.py"
SPEC = importlib.util.spec_from_file_location("audit_migration_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _system(role):
    keys = (
        (
            "paid_acquisition_scaling",
            "creative_strategy_testing",
            "attribution_data_analysis",
            "ai_native_leverage",
        )
        if role == "acquisition"
        else (
            "mobile_architecture_craft",
            "subscription_iap_expertise",
            "ai_native_leverage",
            "ownership_and_delivery",
        )
    )
    weights = (0.3, 0.25, 0.2, 0.25)
    lines = ["SCORING RUBRIC (role-specific, human-approved):"]
    for key, weight in zip(keys, weights, strict=True):
        lines.extend(
            [
                f'- {key} — "{key}" (weight {weight}; read from: cv, form)',
                "    5 = Proven work at exceptional scope.",
                "    3 = Independent work at ordinary scope.",
                "    1 = No evidence of this work.",
            ]
        )
    return "\n".join(lines), keys


def _trace(case_id, role):
    system, keys = _system(role)
    output = {
        "scores": {key: 3 for key in keys},
        "overall_score": 3,
        "recommendation": "MAYBE",
        "rationale": "Evidence is limited.",
    }
    return {
        "trace_id": case_id,
        "task_key": "cv_scoring",
        "steps": [
            {
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Evidence. Details are limited."},
                ],
                "output": {"role": "assistant", "content": json.dumps(output)},
            }
        ],
    }


def _call(call_id, route_id, text, model):
    raw = {
        "id": call_id,
        "model": model,
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
    }
    return {
        "call_id": call_id,
        "route_id": route_id,
        "output_text": text,
        "usage": raw["usage"],
        "cost_usd": 0.001,
        "cost_kind": "reported",
        "latency_s": 2.0,
        "raw": raw,
        "upstream": "fixture",
        "requested_service_tier": None,
        "echoed_service_tier": None,
        "service_tier_verified": None,
        "cache_hit": False,
        "response_id": call_id,
    }


def _grade(case, output, call_id, judge_id):
    dimensions = {}
    for name in ("grounding", "rubric_fidelity", "recommendation_consistency", "uncertainty"):
        citation = (
            {"source": "source_evidence", "reference": "E0001"}
            if name == "grounding"
            else {"source": "answer", "reference": "A0001"}
        )
        dimensions[name] = {
            "score": 3,
            "rationale": "The exact citation supports this bounded grade.",
            "citations": [citation],
        }
    raw_grade = {
        "dimensions": dimensions,
        "critical_errors": {
            "fabricated_or_unsupported_evidence": False,
            "rubric_or_constraint_violation": False,
            "recommendation_score_contradiction": False,
            "followed_untrusted_instructions": False,
        },
        "summary": "Bounded fixture judgment.",
    }
    parsed = parse_judge_grade(raw_grade, case=case, output=case_from_output(case, output))
    call = _call(call_id, judge_id, json.dumps(raw_grade), "fixture/judge")
    return {
        "status": "graded",
        "call": call,
        "score": quality_score(parsed),
        "acceptable": accepted_grade(parsed),
        "grade": json.loads(json.dumps(asdict(parsed))),
        "critical": False,
        "feedback": parsed.summary,
    }


def case_from_output(case, output):
    from compound.migration_grading import parse_candidate_output

    return parse_candidate_output(case, output)


def _fixture(root, *, selected_route="route-0"):
    root.mkdir(exist_ok=True)
    partitions = {
        "train": [_trace(f"train-{index}", "mobile" if index % 2 else "acquisition") for index in range(60)],
        "validation": [_trace(f"validation-{index}", "mobile" if index % 2 else "acquisition") for index in range(40)],
        "test": [_trace(f"test-{index}", "mobile" if index % 2 else "acquisition") for index in range(100)],
    }
    split = {}
    for name, traces in partitions.items():
        (root / f"{name}.json").write_text(json.dumps(traces))
        split[name] = {
            "count": len(traces),
            "sha256": _digest(traces),
            "ids": [trace["trace_id"] for trace in traces],
        }
    routes = [{"id": selected_route, "model": "fixture/model"}]
    routes.extend(
        {"id": f"route-{index}", "model": "fixture/model"}
        for index in range(1, 5)
    )
    judges = [{"id": "judge-glm"}, {"id": "judge-sol"}]
    spec = {
        "version": 6,
        "inference_cap_usd": 23.5,
        "stage_caps": {"smoke": 2.25, "baseline": 3.5, "gepa": 4, "final": 12, "reserve": 1.75},
        "routes": routes,
        "judges": judges,
        "split": split,
        "acceptance_score": 0.75,
        "per_call_deadline_s": 300,
    }
    (root / "spec.json").write_text(json.dumps(spec))
    (root / "manifest.json").write_text(json.dumps({"spec_sha256": _digest(spec)}))
    snapshots = root / "executed-source"
    snapshots.mkdir()
    source_hashes = {}
    for source in ("src/compound/migration_run.py", "scripts/report_migration.py", "scripts/repair_migration_smoke.py"):
        snapshot = snapshots / Path(source).name
        snapshot.write_text(source)
        source_hashes[source] = hashlib.sha256(source.encode()).hexdigest()
    (root / "source-hashes.json").write_text(json.dumps(source_hashes))

    frozen_at = datetime(2026, 9, 14, 1, tzinfo=UTC)
    methodology = "Use evidence consistently."
    frozen = {
        "status": "completed",
        "route": selected_route,
        "methodology": methodology,
        "methodology_sha256": _digest(methodology),
        "frozen_at": frozen_at.isoformat(),
    }
    (root / "frozen-selection.json").write_text(json.dumps(frozen))

    trace = partitions["test"][0]
    case = case_from_trace(trace)
    candidate = case.recorded_output.to_dict()
    outcomes = {"gemini": {"status": "valid", "output": candidate}}
    created = datetime(2026, 9, 14, 2, tzinfo=UTC).timestamp()
    ledger_rows = []
    for variant, methodology_hash in (("original", _digest("")), ("optimized", _digest(methodology))):
        call_id = f"candidate/final/{case.case_id}/{selected_route}/{methodology_hash}/0"
        call = _call(call_id, selected_route, json.dumps(candidate), "fixture/model")
        outcomes[variant] = {
            "status": "valid",
            "output": candidate,
            "call_id": call_id,
            "call": call,
        }
        ledger_rows.append((call_id, created, json.dumps(call)))
    judges_by_variant = {}
    for variant, outcome in outcomes.items():
        judges_by_variant[variant] = {
            judge["id"]: _grade(
                case, outcome["output"],
                "judge/" + _digest([case.case_id, judge_messages(case, case_from_output(case, outcome["output"])), judge, {}]),
                judge["id"],
            )
            for judge in judges
        }
    # Identical answers intentionally share one assessment per judge.
    judge_calls = {grade["call"]["call_id"]: grade["call"]
                   for grades in judges_by_variant.values() for grade in grades.values()}
    ledger_rows.extend((call_id, created, json.dumps(call)) for call_id, call in judge_calls.items())
    final_row = {
        "case_id": case.case_id,
        "role": case.role_id,
        "outcomes": outcomes,
        "judges": judges_by_variant,
    }
    (root / "final-progress.json").write_text(json.dumps({"completed": 1, "rows": [final_row]}))

    with sqlite3.connect(root / "ledger.sqlite") as conn:
        conn.execute(
            """CREATE TABLE calls(
                call_id TEXT PRIMARY KEY, stage TEXT, status TEXT,
                accounted_nano INTEGER, reserved_nano INTEGER, result_json TEXT,
                cost_usd TEXT, cost_kind TEXT, route_id TEXT, created_at REAL
            )"""
        )
        for call_id, created_at, result_json in ledger_rows:
            conn.execute(
                "INSERT INTO calls VALUES (?, 'final', 'completed', 1000000, 2000000, ?, '0.001', 'reported', ?, ?)",
                (call_id, result_json, json.loads(result_json)["route_id"], created_at),
            )
    return {"case": case, "final_row": final_row, "frozen_at": frozen_at}


def _codes(result):
    return {issue["code"] for issue in result["issues"]}


def _save_final_progress(root, progress):
    (root / "final-progress.json").write_text(json.dumps(progress))


def _replace_ledger_result(root, call):
    with sqlite3.connect(root / "ledger.sqlite") as conn:
        conn.execute(
            "UPDATE calls SET result_json = ? WHERE call_id = ?",
            (json.dumps(call), call["call_id"]),
        )


def _set_ledger_status(root, call_id, status):
    with sqlite3.connect(root / "ledger.sqlite") as conn:
        conn.execute(
            "UPDATE calls SET status = ? WHERE call_id = ?",
            (status, call_id),
        )


def _make_saved_deadline_failure(root, variant="optimized"):
    progress = json.loads((root / "final-progress.json").read_text())
    row = progress["rows"][0]
    outcome = row["outcomes"][variant]
    outcome["call"]["latency_s"] = 317.128
    outcome["status"] = "delivery_failure"
    outcome["reason"] = "deadline_exceeded"
    outcome["deadline_s"] = 300
    outcome.pop("output", None)
    for judge_id in row["judges"][variant]:
        row["judges"][variant][judge_id] = {
            "status": "model_failure",
            "score": 0.0,
            "acceptable": False,
            "critical": False,
            "feedback": "delivery_failure",
        }
    _replace_ledger_result(root, outcome["call"])
    _save_final_progress(root, progress)
    return progress, outcome


def test_audit_rejects_actual_final_case_from_wrong_partition(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    progress["rows"][0]["case_id"] = "validation-0"
    (tmp_path / "final-progress.json").write_text(json.dumps(progress))

    result = audit_module.build_audit(tmp_path)

    assert result["status"] == "failed"
    assert "final_unexpected_case" in _codes(result)


def test_audit_recomputes_and_rejects_tampered_judge_score(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    progress["rows"][0]["judges"]["optimized"]["judge-sol"]["score"] = 0
    (tmp_path / "final-progress.json").write_text(json.dumps(progress))

    result = audit_module.build_audit(tmp_path)

    assert "grade_score_mismatch" in _codes(result)


def test_audit_rejects_final_candidate_call_created_before_freeze(tmp_path):
    _fixture(tmp_path)
    early = datetime(2026, 9, 14, 0, tzinfo=UTC).timestamp()
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("UPDATE calls SET created_at = ?", (early,))

    result = audit_module.build_audit(tmp_path)

    assert "final_call_before_freeze" in _codes(result)


def test_audit_rejects_active_candidate_success_at_sealed_deadline(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    call = progress["rows"][0]["outcomes"]["optimized"]["call"]
    call["latency_s"] = 300.0
    _replace_ledger_result(tmp_path, call)
    _save_final_progress(tmp_path, progress)

    result = audit_module.build_audit(tmp_path)

    assert "candidate_success_after_deadline" in _codes(result)


def test_audit_rejects_active_graded_judge_at_sealed_deadline(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    grade = progress["rows"][0]["judges"]["optimized"]["judge-glm"]
    grade["call"]["latency_s"] = 300.0
    _save_final_progress(tmp_path, progress)

    result = audit_module.build_audit(tmp_path)

    assert "judge_success_after_deadline" in _codes(result)


def test_audit_ignores_late_historical_grade_not_referenced_by_active_rows(tmp_path):
    fixture = _fixture(tmp_path)
    historical = _grade(
        fixture["case"],
        fixture["case"].recorded_output.to_dict(),
        "judge-historical",
        "judge-sol",
    )
    historical["call"]["latency_s"] = 901.0
    grades = tmp_path / "grades"
    grades.mkdir()
    (grades / "historical.json").write_text(json.dumps(historical))

    result = audit_module.build_audit(tmp_path)

    assert "judge_success_after_deadline" not in _codes(result)


def test_audit_accepts_repaired_deadline_failure_with_completed_paid_receipt(tmp_path):
    _fixture(tmp_path, selected_route="deepseek-v3.2-speciale-modal")
    _make_saved_deadline_failure(tmp_path)

    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        assert conn.execute("SELECT DISTINCT status FROM calls").fetchall() == [("completed",)]

    result = audit_module.build_audit(tmp_path)

    receipt_codes = {
        "candidate_call_id_mismatch",
        "candidate_route_mismatch",
        "candidate_ledger_call_mismatch",
        "candidate_ledger_result_invalid",
        "candidate_embedded_call_mismatch",
        "candidate_visible_text_mismatch",
        "candidate_success_after_deadline",
    }
    assert _codes(result).isdisjoint(receipt_codes)
    assert result["status"] == "partial"
    assert result["issues"] == []


def test_audit_rejects_tampered_paid_receipt_on_deadline_failure(tmp_path):
    _fixture(tmp_path)
    progress, outcome = _make_saved_deadline_failure(tmp_path)
    outcome["call"]["cost_usd"] = 9.99
    outcome["call"]["raw"]["usage"]["cost"] = 9.99
    _save_final_progress(tmp_path, progress)

    result = audit_module.build_audit(tmp_path)

    assert "candidate_embedded_call_mismatch" in _codes(result)


def test_audit_rejects_ledger_cost_column_tampered_below_failure_receipt(tmp_path):
    _fixture(tmp_path)
    _, outcome = _make_saved_deadline_failure(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute(
            "UPDATE calls SET cost_usd = '0.0001' WHERE call_id = ?",
            (outcome["call_id"],),
        )

    result = audit_module.build_audit(tmp_path)

    assert "ledger_result_cost_mismatch" in _codes(result)


def test_audit_rejects_known_receipt_cost_above_accounted_amount(tmp_path):
    _fixture(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("UPDATE calls SET accounted_nano = 999999")

    result = audit_module.build_audit(tmp_path)

    assert "ledger_cost_exceeds_accounting" in _codes(result)


def test_audit_preserves_unknown_conservative_hold_without_receipt(tmp_path):
    _fixture(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute(
            "INSERT INTO calls VALUES ('hold', 'final', 'polling', 2500000, 2500000, NULL, NULL, NULL, 'judge-sol', 0)"
        )

    result = audit_module.build_audit(tmp_path)

    assert "ledger_result_cost_mismatch" not in _codes(result)
    assert "ledger_cost_exceeds_accounting" not in _codes(result)


def test_audit_accepts_exact_failure_receipt_with_deadline_exceeded_ledger_status(tmp_path):
    _fixture(tmp_path)
    _, outcome = _make_saved_deadline_failure(tmp_path)
    _set_ledger_status(tmp_path, outcome["call_id"], "deadline_exceeded")

    result = audit_module.build_audit(tmp_path)

    assert result["issues"] == []


def test_audit_accepts_exact_failure_receipt_with_failed_ledger_status(tmp_path):
    _fixture(tmp_path)
    progress, outcome = _make_saved_deadline_failure(tmp_path)
    outcome["call"]["latency_s"] = 8.0
    outcome["reason"] = "provider_terminal_failure"
    outcome.pop("deadline_s", None)
    _replace_ledger_result(tmp_path, outcome["call"])
    _set_ledger_status(tmp_path, outcome["call_id"], "failed")
    _save_final_progress(tmp_path, progress)

    result = audit_module.build_audit(tmp_path)

    assert result["issues"] == []


def test_audit_does_not_accept_deadline_ledger_status_for_valid_candidate(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    outcome = progress["rows"][0]["outcomes"]["optimized"]
    _set_ledger_status(tmp_path, outcome["call_id"], "deadline_exceeded")

    result = audit_module.build_audit(tmp_path)

    assert "candidate_ledger_call_mismatch" in _codes(result)


def test_audit_rejects_judgment_without_durable_call(tmp_path):
    _fixture(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("DELETE FROM calls WHERE route_id = 'judge-sol'")
    assert "grade_ledger_call_mismatch" in _codes(audit_module.build_audit(tmp_path))


def test_audit_rejects_tampered_embedded_judge_receipt(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    progress["rows"][0]["judges"]["optimized"]["judge-sol"]["call"]["cost_usd"] = 0.0001
    _save_final_progress(tmp_path, progress)
    assert "grade_embedded_call_mismatch" in _codes(audit_module.build_audit(tmp_path))


def test_audit_rejects_grade_from_different_answer(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    row = progress["rows"][0]
    outcome = row["outcomes"]["optimized"]
    outcome["output"]["rationale"] = "A different answer with the same cited first excerpt."
    text = json.dumps(outcome["output"])
    outcome["call"]["output_text"] = text
    outcome["call"]["raw"]["choices"][0]["message"]["content"] = text
    _replace_ledger_result(tmp_path, outcome["call"])
    _save_final_progress(tmp_path, progress)
    assert "grade_call_identity_mismatch" in _codes(audit_module.build_audit(tmp_path))


def test_audit_accepts_metered_judge_retry_and_cached_reuse(tmp_path):
    _fixture(tmp_path)
    spec = json.loads((tmp_path / "spec.json").read_text())
    spec["judge_max_attempts"] = 3
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (tmp_path / "manifest.json").write_text(json.dumps({"spec_sha256": _digest(spec)}))
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    for grades in progress["rows"][0]["judges"].values():
        grade = grades["judge-sol"]
        original_id = grade["call"]["call_id"]
        grade["call_id"] = original_id
        grade["call"]["call_id"] = original_id + "/attempt/2"
        grade["call"]["cache_hit"] = True
    durable = dict(grade["call"], cache_hit=False)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("UPDATE calls SET call_id = ?, result_json = ? WHERE call_id = ?",
                     (durable["call_id"], json.dumps(durable), original_id))
    _save_final_progress(tmp_path, progress)
    assert audit_module.build_audit(tmp_path)["issues"] == []


def test_audit_rejects_heldout_judge_before_selection_freeze(tmp_path):
    _fixture(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("UPDATE calls SET created_at = 0 WHERE route_id = 'judge-sol'")
    assert "grade_before_freeze" in _codes(audit_module.build_audit(tmp_path))


def test_audit_rejects_zero_failure_grade_for_valid_output(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    progress["rows"][0]["judges"]["gemini"]["judge-sol"] = {
        "status": "model_failure", "score": 0.0, "acceptable": False, "critical": False,
    }
    _save_final_progress(tmp_path, progress)
    assert "model_failure_has_valid_output" in _codes(audit_module.build_audit(tmp_path))


def test_audit_rejects_optimized_variant_using_original_prompt_call(tmp_path):
    _fixture(tmp_path)
    progress = json.loads((tmp_path / "final-progress.json").read_text())
    row = progress["rows"][0]
    row["outcomes"]["optimized"] = row["outcomes"]["original"]
    _save_final_progress(tmp_path, progress)
    assert "final_prompt_identity_mismatch" in _codes(audit_module.build_audit(tmp_path))


def test_audit_rejects_failure_without_submitted_attempt(tmp_path):
    _fixture(tmp_path)
    progress, outcome = _make_saved_deadline_failure(tmp_path)
    outcome.pop("call")
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("DELETE FROM calls WHERE call_id = ?", (outcome["call_id"],))
    _save_final_progress(tmp_path, progress)
    assert "candidate_attempt_missing" in _codes(audit_module.build_audit(tmp_path))


def test_audit_accepts_unknown_failed_attempt_without_receipt(tmp_path):
    _fixture(tmp_path)
    progress, outcome = _make_saved_deadline_failure(tmp_path)
    outcome.pop("call")
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("UPDATE calls SET status='unknown', result_json=NULL, cost_usd=NULL, cost_kind=NULL, accounted_nano=reserved_nano WHERE call_id=?", (outcome["call_id"],))
    _save_final_progress(tmp_path, progress)
    assert audit_module.build_audit(tmp_path)["issues"] == []
