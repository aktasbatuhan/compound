from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "report_migration.py"
SPEC = importlib.util.spec_from_file_location("report_migration", SCRIPT)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

PLOT_SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_migration_report.py"
PLOT_SPEC = importlib.util.spec_from_file_location("plot_migration_report", PLOT_SCRIPT)
assert PLOT_SPEC and PLOT_SPEC.loader
plot = importlib.util.module_from_spec(PLOT_SPEC)
PLOT_SPEC.loader.exec_module(plot)

GEPA_AUDIT_FIXTURE = {
    "version": 1,
    "recorded_at_utc": "2026-09-14T03:00:00Z",
    "runner_preflight_rejected_revisions": 1,
    "runner_preflight_callback_zeroes": 3,
    "provider_tested_revisions": 3,
    "provider_requests": 9,
    "contract_valid_answers": 5,
    "generated_format_failures": 4,
    "valid_answer_judge_score": 1.0,
    "provenance": {"fixture.json": "abc"},
}


def _output(score: int = 3, recommendation: str = "MAYBE") -> dict:
    return {
        "scores": {"craft": score, "delivery": score},
        "recommendation": recommendation,
        "rationale": "Private rationale is never copied into the report.",
    }


def _grade(score: float, *, status: str = "graded", critical: bool = False) -> dict:
    return {
        "status": status,
        "score": score,
        "acceptable": score >= 0.75 and not critical,
        "critical": critical,
        "feedback": "private feedback",
    }


def _row(
    case_id: str,
    *,
    role: str = "mobile",
    gemini: float = 0.8,
    original: float = 0.8,
    optimized: float = 0.8,
    optimized_status: str = "valid",
) -> dict:
    outcomes = {
        "gemini": {"status": "valid", "output": _output()},
        "original": {
            "status": "valid",
            "call_id": f"candidate/final/{case_id}/original",
            "output": _output(),
            "call": {"latency_s": 20.0, "cost_usd": 0.01},
        },
        "optimized": {
            "status": optimized_status,
            "call_id": f"candidate/final/{case_id}/optimized",
            "output": _output(4, "HIRE") if optimized_status == "valid" else None,
            "call": {"latency_s": 25.0, "cost_usd": 0.01},
        },
    }
    judges = {}
    for variant, score in (("gemini", gemini), ("original", original), ("optimized", optimized)):
        judges[variant] = {
            "judge-sol": _grade(score),
            "judge-glm": _grade(score),
        }
    return {"case_id": case_id, "role": role, "outcomes": outcomes, "judges": judges}


def _write_fixture(
    root: Path,
    rows: list[dict],
    *,
    expected: int | None = None,
    gepa_audit: dict | None = None,
) -> None:
    expected = len(rows) if expected is None else expected
    routes = [{"id": f"route-{index}"} for index in range(5)]
    spec = {
        "primary_judge": "sol",
        "secondary_judge": "glm",
        "noninferiority_margin": 0.05,
        "acceptance_score": 0.75,
        "deadline_metrics_s": [10, 30, 60, 300],
        "inference_cap_usd": 23.5,
        "total_cap_usd": 25.0,
        "source_revision": "fixture-revision",
        "judge_host_policy": "GLM uses a frozen FP8 allowlist; Sol is pinned to OpenAI.",
        "judges": [
            {
                "id": "judge-glm",
                "model": "z-ai/glm-5.3-flash",
                "expected_upstreams": ["BaseTen", "Sail Research", "Phala", "Reka"],
                "provider": {"only": ["baseten/fp8", "sail-research/fp8"], "allow_fallbacks": True},
            },
            {
                "id": "judge-sol",
                "model": "openai/gpt-5.6-sol",
                "expected_upstream": "OpenAI",
                "provider": {"only": ["openai"], "allow_fallbacks": False},
            },
        ],
        "routes": routes,
        "split": {"test": {"count": expected}},
    }
    root.mkdir(exist_ok=True)
    (root / "spec.json").write_text(json.dumps(spec))
    digest = report.hashlib.sha256(
        json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"spec_sha256": digest}))
    (root / "source-hashes.json").write_text(json.dumps({"migration_run.py": "abc"}))
    baseline_routes = [
        {
            "route": route["id"],
            "mean_quality": 0.8,
            "delta": 0.0,
            "valid": 2,
            "n": 2,
            "mean_cost": 0.01,
            "cost_known": True,
            "eligible": True,
        }
        for route in routes
    ]
    (root / "baseline.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "selected_route": "route-0",
                "qualified_on_validation": True,
                "routes": baseline_routes,
                "rows": [],
            }
        )
    )
    (root / "frozen-selection.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "methodology": "Improved general method.",
                "gepa_status": "completed",
                "validation_result": {
                    "before_val_score": 0.6,
                    "after_val_score": 0.7,
                    "valid_proposals": 4,
                    "rejected_proposals": 0,
                    "candidate_count": 1,
                    "metric_calls": 64,
                    "reflection_calls": 4,
                },
            }
        )
    )
    if gepa_audit is not None:
        (root / "gepa-development-audit.json").write_text(json.dumps(gepa_audit))
    (root / "final.json").write_text(json.dumps({"status": "completed", "rows": rows}))
    (root / "vm.json").write_text(
        json.dumps(
            {
                "created_at": "2026-09-14T00:00:00+00:00",
                "estimated_compute_usd_per_hour": 0.06701142,
                "compute_reserve_usd": 1.5,
                "boot_disk_gb": 20,
            }
        )
    )
    (root / "cleanup.json").write_text(
        json.dumps(
            {
                "vm_deleted": True,
                "disk_deleted": True,
                "at": "2026-09-14T01:00:00+00:00",
            }
        )
    )
    connection = sqlite3.connect(root / "ledger.sqlite")
    connection.execute(
        """CREATE TABLE calls (
        call_id TEXT, stage TEXT, status TEXT, accounted_nano INTEGER,
        cost_usd TEXT, cost_kind TEXT, result_json TEXT
        )"""
    )
    connection.execute(
        "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "candidate/final/one",
            "final",
            "completed",
            10_000_000,
            "0.01",
            "reported",
            json.dumps(
                {
                    "route_id": "doubleword-realtime",
                    "upstream": "Doubleword",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ),
        ),
    )
    connection.commit()
    connection.close()


def _add_ledger_call(
    root: Path,
    call_id: str,
    *,
    status: str = "completed",
    accounted_nano: int = 10_000_000,
    cost_usd: str | None = "0.01",
    cost_kind: str | None = "reported",
) -> None:
    connection = sqlite3.connect(root / "ledger.sqlite")
    connection.execute(
        "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
        (call_id, "final", status, accounted_nano, cost_usd, cost_kind, None),
    )
    connection.commit()
    connection.close()


def test_paired_bootstrap_is_case_level_deterministic() -> None:
    first = report.paired_bootstrap([0.1, 0.2, 0.3], draws=10_000, seed=7419)
    second = report.paired_bootstrap([0.1, 0.2, 0.3], draws=10_000, seed=7419)
    assert first == second
    assert first["n"] == 3
    assert first["mean"] == pytest.approx(0.2)
    assert first["ci95"][0] <= first["mean"] <= first["ci95"][1]


def test_noninferiority_crossing_margin_is_inconclusive() -> None:
    comparison = {"ci95": [-0.08, 0.02], "missing_pairs": 0}
    conclusion = report._ni_conclusion(comparison, margin=0.05, complete=True)
    assert conclusion["conclusion"] == "inconclusive"


def test_complete_supported_fixture_passes_conservative_gate(tmp_path) -> None:
    rows = [
        _row(f"case-{index}", role="mobile" if index % 2 else "acquisition", original=0.75)
        for index in range(6)
    ]
    _write_fixture(tmp_path, rows)

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    assert summary["gate"]["status"] == "passed"
    assert summary["gate"]["noninferiority_claim"] is True
    assert len(summary["baseline"]["routes"]) == 5
    assert summary["provenance"]["manifest_matches_spec"] is True
    assert summary["cost"]["inference"]["settled_known_cost_usd"] == pytest.approx(0.01)
    assert summary["cost"]["inference"]["groups"]["candidate_generation"][
        "actual_upstreams"
    ] == {"Doubleword": 1}
    assert summary["judge_host_policy"]["judges"][0]["expected_upstreams"] == [
        "BaseTen",
        "Sail Research",
        "Phala",
        "Reka",
    ]
    markdown = report.render_markdown(summary)
    assert "Frozen judge host policy" in markdown
    assert "Allow fallback inside allowlist" in markdown


def test_missing_judgment_is_counted_and_makes_ni_inconclusive(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    rows[1]["judges"]["optimized"]["judge-sol"] = {"status": "missing_judgment"}
    _write_fixture(tmp_path, rows)

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))
    sol = summary["final"]["judges"]["judge-sol"]

    assert sol["counts"]["optimized"]["missing_judgments"] == 1
    assert sol["comparisons"]["optimized_vs_gemini"]["missing_pairs"] == 1
    assert sol["optimized_vs_gemini_noninferiority"]["conclusion"] == "inconclusive"
    assert summary["gate"]["noninferiority_claim"] is False


def test_model_failure_zero_is_included_in_paired_quality(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    rows[1]["outcomes"]["optimized"] = {"status": "delivery_failure"}
    rows[1]["judges"]["optimized"]["judge-sol"] = _grade(0.0, status="model_failure")
    rows[1]["judges"]["optimized"]["judge-glm"] = _grade(0.0, status="model_failure")
    _write_fixture(tmp_path, rows)

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))
    counts = summary["final"]["judges"]["judge-sol"]["counts"]["optimized"]
    comparison = summary["final"]["judges"]["judge-sol"]["comparisons"][
        "optimized_vs_gemini"
    ]

    assert counts["model_failures_included_as_zero"] == 1
    assert counts["missing_judgments"] == 0
    assert comparison["n"] == 2
    assert comparison["missing_pairs"] == 0


def test_latency_labels_disclose_response_receipt_sample_and_planned_denominator(
    tmp_path,
) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    rows[1]["outcomes"]["optimized"] = {
        "status": "delivery_failure",
        "call_id": "candidate/final/case-b/optimized",
    }
    rows[1]["judges"]["optimized"]["judge-sol"] = _grade(
        0.0, status="model_failure"
    )
    rows[1]["judges"]["optimized"]["judge-glm"] = _grade(
        0.0, status="model_failure"
    )
    _write_fixture(tmp_path, rows)

    summary = report.build_summary(
        tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC)
    )
    markdown = report.render_markdown(summary)

    assert summary["final"]["timing"]["optimized"]["judge-sol"]["latency_sample"] == 1
    assert "Response-receipt p50 / p90 (n)" in markdown
    assert "attempts without a receipt are absent from that sample" in markdown
    assert "use all planned cases as the denominator" in markdown


def test_negative_noninferiority_fails(tmp_path) -> None:
    rows = [
        _row(f"case-{index}", role="mobile" if index % 2 else "acquisition", optimized=0.5)
        for index in range(8)
    ]
    _write_fixture(tmp_path, rows)

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))
    ni = summary["final"]["judges"]["judge-sol"][
        "optimized_vs_gemini_noninferiority"
    ]

    assert ni["conclusion"] == "failed"
    assert summary["gate"]["status"] == "failed"
    assert summary["gate"]["noninferiority_claim"] is False


def test_low_delivery_and_unresolved_cost_fail_gate(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", optimized_status="delivery_failure")]
    rows[1]["judges"]["optimized"]["judge-sol"] = _grade(0.0, status="model_failure")
    rows[1]["judges"]["optimized"]["judge-glm"] = _grade(0.0, status="model_failure")
    _write_fixture(tmp_path, rows)
    connection = sqlite3.connect(tmp_path / "ledger.sqlite")
    connection.execute(
        "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("judge/unresolved", "final", "unknown", 500_000_000, None, None, None),
    )
    connection.commit()
    connection.close()

    summary = report.write_report(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    assert summary["final"]["delivery"]["optimized"]["valid_fraction_of_planned"] == 0.5
    assert summary["cost"]["inference"]["unresolved_calls"] == 1
    assert summary["cost"]["inference"]["settled_known_cost_usd"] == pytest.approx(0.01)
    assert summary["cost"]["inference"]["unresolved_accounted_usd"] == pytest.approx(0.5)
    assert summary["gate"]["status"] == "failed"
    assert "Private rationale" not in (tmp_path / "report.md").read_text()
    assert "Private rationale" not in (tmp_path / "summary.json").read_text()


def test_known_charge_on_terminal_failure_is_settled_not_unresolved(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    _write_fixture(tmp_path, rows)
    connection = sqlite3.connect(tmp_path / "ledger.sqlite")
    connection.execute(
        "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "judge/terminal-failure",
            "final",
            "failed",
            20_000_000,
            "0.02",
            "reported",
            json.dumps(
                {
                    "route_id": "judge-sol",
                    "upstream": "OpenAI",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2},
                }
            ),
        ),
    )
    connection.commit()
    connection.close()

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    inference = summary["cost"]["inference"]
    assert inference["status"] == "resolved"
    assert inference["unresolved_calls"] == 0
    assert inference["settled_known_cost_usd"] == pytest.approx(0.03)
    assert inference["judge_upstreams_by_stage"]["final"]["judge-sol"][
        "actual_upstreams"
    ] == {"OpenAI": 1}
    assert summary["gate"]["status"] == "passed"


def test_unresolved_billing_is_inconclusive_but_preserves_quality_ni(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    _write_fixture(tmp_path, rows)
    connection = sqlite3.connect(tmp_path / "ledger.sqlite")
    connection.execute(
        "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("judge/unresolved", "smoke", "failed", 500_000_000, None, None, None),
    )
    connection.commit()
    connection.close()

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    assert summary["gate"]["quality_noninferiority_status"] == "supported"
    assert summary["gate"]["noninferiority_claim"] is True
    assert summary["gate"]["study_gate_status"] == "inconclusive"
    assert "operational_readiness_status" not in summary["gate"]
    assert summary["gate"]["status"] == "inconclusive"


def test_conservative_inference_cap_excess_still_fails(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    _write_fixture(tmp_path, rows)
    connection = sqlite3.connect(tmp_path / "ledger.sqlite")
    connection.execute(
        "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("judge/over-cap", "final", "failed", 24_000_000_000, None, None, None),
    )
    connection.commit()
    connection.close()

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    assert summary["cost"]["inference"]["within_inference_cap"] is False
    assert summary["gate"]["status"] == "failed"
    assert summary["gate"]["noninferiority_claim"] is True


def test_candidate_cost_is_undefined_if_any_final_charge_is_unknown(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    _write_fixture(tmp_path, rows)
    for row in rows:
        _add_ledger_call(tmp_path, row["outcomes"]["original"]["call_id"])
    _add_ledger_call(tmp_path, rows[0]["outcomes"]["optimized"]["call_id"])
    _add_ledger_call(
        tmp_path,
        rows[1]["outcomes"]["optimized"]["call_id"],
        status="unknown",
        accounted_nano=50_000_000,
        cost_usd=None,
        cost_kind=None,
    )
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    baseline["routes"][0]["cost_known"] = False
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    original = summary["final"]["candidate_cost"]["original"]
    optimized = summary["final"]["candidate_cost"]["optimized"]
    assert original["mean_candidate_cost_usd"] == pytest.approx(0.01)
    assert optimized["known_charge_subtotal_usd"] == pytest.approx(0.01)
    assert optimized["unknown_charge_calls"] == 1
    assert optimized["mean_candidate_cost_usd"] is None
    assert all(
        value["cost_per_acceptable_usd"] is None
        for value in optimized["cost_per_acceptable_by_judge"].values()
    )
    assert summary["baseline"]["routes"][0]["mean_candidate_cost_usd"] is None


def test_failed_candidate_with_known_charge_is_included_in_cost(tmp_path) -> None:
    rows = [_row("case-a", optimized_status="delivery_failure")]
    rows[0]["judges"]["optimized"]["judge-sol"] = _grade(0.0, status="model_failure")
    rows[0]["judges"]["optimized"]["judge-glm"] = _grade(0.0, status="model_failure")
    _write_fixture(tmp_path, rows)
    _add_ledger_call(
        tmp_path,
        rows[0]["outcomes"]["optimized"]["call_id"],
        status="failed",
        accounted_nano=30_000_000,
        cost_usd="0.03",
    )

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    optimized = summary["final"]["candidate_cost"]["optimized"]
    assert optimized["total_candidate_cost_usd"] == pytest.approx(0.03)
    assert optimized["known_charged_calls"] == 1
    assert optimized["cost_per_acceptable_by_judge"]["judge-sol"][
        "cost_per_acceptable_usd"
    ] is None


def test_baseline_total_uses_durable_mean_when_paid_failure_has_no_receipt() -> None:
    spec = {"routes": [{"id": "route-a"}]}
    baseline = {
        "status": "completed",
        "routes": [
            {
                "route": "route-a",
                "mean_quality": 0.7,
                "delta": -0.01,
                "valid": 1,
                "n": 2,
                "mean_cost": 0.03,
                "cost_known": True,
                "eligible": False,
            }
        ],
        "rows": [
            {
                "route": "route-a",
                "outcome": {
                    "status": "valid",
                    "call": {"cost_usd": 0.02, "latency_s": 4.0},
                },
            },
            {
                "route": "route-a",
                "outcome": {"status": "delivery_failure"},
            },
        ],
    }

    summary = report.summarize_baseline(baseline, spec)
    route = summary["routes"][0]

    assert route["mean_candidate_cost_usd"] == pytest.approx(0.03)
    assert route["total_candidate_cost_usd"] == pytest.approx(0.06)
    assert route["latency_sample"] == 1


def test_baseline_reports_conditional_semantic_quality_with_its_denominator() -> None:
    spec = {"routes": [{"id": "route-a"}]}
    baseline = {
        "status": "completed",
        "selected_route": "route-a",
        "routes": [
            {
                "route": "route-a",
                "mean_quality": 0.28,
                "delta": -0.02,
                "valid": 3,
                "n": 5,
                "mean_cost": 0.01,
                "cost_known": True,
                "eligible": False,
            }
        ],
        "rows": [
            {
                "route": "route-a",
                "outcome": {"status": "valid"},
                "grade": {"status": "graded", "score": 0.8},
            },
            {
                "route": "route-a",
                "outcome": {"status": "valid"},
                "grade": {"status": "graded", "score": 0.6},
            },
            {
                "route": "route-a",
                "outcome": {"status": "valid"},
                "grade": {"status": "missing_judgment", "score": 0.9},
            },
            {
                "route": "route-a",
                "outcome": {"status": "invalid_output"},
                "grade": {"status": "model_failure", "score": 0.0},
            },
            {
                "route": "route-a",
                "outcome": {"status": "delivery_failure"},
                "grade": {"status": "model_failure", "score": 0.0},
            },
        ],
    }

    summary = report.summarize_baseline(baseline, spec)
    route = summary["routes"][0]
    table = plot.validation_chart(
        {"baseline": {"routes": [route], "selected_route": "route-a"}}
    )

    assert route["glm_mean_quality"] == pytest.approx(0.28)
    assert route["glm_difference_vs_gemini"] == pytest.approx(-0.02)
    assert route["eligible_on_validation"] is False
    assert summary["selected_route"] == "route-a"
    assert route["glm_semantic_valid_answers"] == 2
    assert route["glm_mean_semantic_quality"] == pytest.approx(0.7)
    assert route["failure_status_counts"] == {
        "delivery_failure": 1,
        "invalid_output": 1,
    }
    assert "Planned-cohort GLM usable quality" in table
    assert "Valid-answer GLM semantic quality" in table
    assert "0.700 (n=2)" in table
    assert "delivery_failure: 1; invalid_output: 1" in table


def test_cost_per_acceptable_requires_complete_judge_assessments(tmp_path) -> None:
    rows = [_row("case-a")]
    rows[0]["judges"]["optimized"]["judge-sol"] = {"status": "missing_judgment"}
    _write_fixture(tmp_path, rows)
    _add_ledger_call(
        tmp_path,
        rows[0]["outcomes"]["optimized"]["call_id"],
        accounted_nano=20_000_000,
        cost_usd="0.02",
    )

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    by_judge = summary["final"]["candidate_cost"]["optimized"][
        "cost_per_acceptable_by_judge"
    ]
    assert by_judge["judge-sol"]["cost_per_acceptable_usd"] is None
    assert by_judge["judge-glm"]["cost_per_acceptable_usd"] == pytest.approx(0.02)


def test_identical_variant_call_ids_share_ledger_cost_without_all_in_double_count(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    for row in rows:
        shared = f"candidate/final/{row['case_id']}/shared"
        row["outcomes"]["original"]["call_id"] = shared
        row["outcomes"]["optimized"]["call_id"] = shared
    _write_fixture(tmp_path, rows)
    connection = sqlite3.connect(tmp_path / "ledger.sqlite")
    connection.execute("DELETE FROM calls")
    connection.commit()
    connection.close()
    for row in rows:
        _add_ledger_call(tmp_path, row["outcomes"]["original"]["call_id"])

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    assert summary["final"]["candidate_cost"]["original"][
        "total_candidate_cost_usd"
    ] == pytest.approx(0.02)
    assert summary["final"]["candidate_cost"]["optimized"][
        "total_candidate_cost_usd"
    ] == pytest.approx(0.02)
    assert summary["cost"]["inference"]["accounted_usd"] == pytest.approx(0.02)


def test_final_progress_is_partial_and_has_no_verdict(tmp_path) -> None:
    rows = [_row("case-a")]
    _write_fixture(tmp_path, rows, expected=2)
    (tmp_path / "final.json").unlink()
    (tmp_path / "final-progress.json").write_text(json.dumps({"completed": 1, "rows": rows}))

    summary = report.build_summary(tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC))

    assert summary["status"] == "partial"
    assert summary["gate"]["status"] == "partial"
    assert summary["gate"]["noninferiority_claim"] is False


def test_report_separates_planned_failure_zeroes_from_valid_answer_semantics(
    tmp_path,
) -> None:
    rows = [_row("case-a", optimized=0.8), _row("case-b", optimized=0.0)]
    rows[1]["outcomes"]["optimized"] = {
        "status": "invalid_output",
        "call_id": "candidate/final/case-b/optimized",
    }
    rows[1]["judges"]["optimized"]["judge-sol"] = _grade(
        0.0, status="model_failure"
    )
    rows[1]["judges"]["optimized"]["judge-glm"] = _grade(
        0.0, status="model_failure"
    )
    _write_fixture(tmp_path, rows)

    summary = report.build_summary(
        tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC)
    )
    counts = summary["final"]["judges"]["judge-sol"]["counts"]["optimized"]
    markdown = report.render_markdown(summary)

    assert counts["mean_usable_quality"] == pytest.approx(0.4)
    assert counts["model_failures_included_as_zero"] == 1
    assert counts["semantic_valid_answers"] == 1
    assert counts["mean_semantic_quality"] == pytest.approx(0.8)
    assert "Planned-cohort usable quality" in markdown
    assert "Valid-answer semantic quality" in markdown
    assert "without erasing those failures" in markdown


def test_ni_plot_discloses_missing_pairs_and_limits_margin_to_ni_contrast(
    tmp_path,
) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    rows[1]["judges"]["optimized"]["judge-sol"] = {
        "status": "missing_judgment"
    }
    _write_fixture(tmp_path, rows)
    summary = report.build_summary(
        tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC)
    )

    chart = plot.quality_delta_chart(summary)

    assert "n=1/2, missing=1; NI inconclusive" in chart
    assert "red row ticks apply only to optimized vs recorded Gemini" in chart
    assert chart.count("NI margin -0.050") == 2
    assert "-0.05 NI margin" not in chart


def test_seed_retention_reports_raw_gepa_counts_and_hides_redundant_figure_arms(
    tmp_path,
) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    for row in rows:
        shared = f"candidate/final/{row['case_id']}/shared"
        row["outcomes"]["original"]["call_id"] = shared
        row["outcomes"]["optimized"]["call_id"] = shared
    _write_fixture(tmp_path, rows, gepa_audit=GEPA_AUDIT_FIXTURE)
    frozen_path = tmp_path / "frozen-selection.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["methodology"] = ""
    frozen["validation_result"]["before_val_score"] = 0.98
    frozen["validation_result"]["after_val_score"] = 0.98
    frozen_path.write_text(json.dumps(frozen))

    summary = report.build_summary(
        tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC)
    )
    gepa = summary["final"]["gepa"]
    markdown = report.render_markdown(summary)
    figures = plot.render(summary)

    assert gepa["valid_proposals"] == 4
    assert gepa["rejected_proposals"] == 0
    assert gepa["candidate_count"] == 1
    assert gepa["revisions_reaching_full_validation"] == 0
    assert gepa["metric_calls"] == 64
    assert gepa["reflection_calls"] == 4
    assert gepa["development_screen_audit"] == GEPA_AUDIT_FIXTURE
    assert "without a provider request" in markdown
    assert "5 valid answers were each judged 1.0" in markdown
    assert "4 common-contract format failures scored zero" in markdown
    assert "neither semantic-quality gain nor loss" in markdown
    assert "does not establish that other methodologies cannot improve" in markdown
    assert "neither semantic-quality gain nor loss" in figures
    assert "redundant separately seeded Monte Carlo contrasts are hidden" in figures
    assert "same shared calls and are not additive" in figures
    assert "<td>original</td>" not in figures


def test_seed_retention_without_development_audit_does_not_invent_screen_counts(
    tmp_path,
) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    _write_fixture(tmp_path, rows)
    frozen_path = tmp_path / "frozen-selection.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["methodology"] = ""
    frozen_path.write_text(json.dumps(frozen))

    summary = report.build_summary(
        tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC)
    )
    markdown = report.render_markdown(summary)
    figures = plot.render(summary)

    assert summary["final"]["gepa"]["development_screen_audit"] is None
    assert "No development-screen audit was supplied" in markdown
    assert "No development-screen audit was supplied" in figures
    assert "without a provider request" not in markdown
    assert "without a provider request" not in figures
    assert "sampled-minibatch requests" not in markdown
    assert "valid answers were each judged" not in figures


def test_passed_gate_is_named_exploratory_study_gate(tmp_path) -> None:
    rows = [_row("case-a"), _row("case-b", role="acquisition")]
    _write_fixture(tmp_path, rows)

    summary = report.build_summary(
        tmp_path, now=datetime(2026, 9, 14, 2, tzinfo=UTC)
    )
    markdown = report.render_markdown(summary)
    figures = plot.render(summary)

    assert summary["gate"]["study_gate_status"] == "passed"
    assert "operational_readiness_status" not in summary["gate"]
    assert "Exploratory study gate: **passed**" in markdown
    assert "Exploratory study gate: passed" in figures
    assert "Operational readiness" not in markdown
