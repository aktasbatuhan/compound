"""End-to-end offline fixture for the visible-output smoke repair."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

from compound.migration_io import CallResult, Ledger

_SCRIPT = Path(__file__).parents[1] / "scripts/repair_migration_smoke.py"
_SPEC = importlib.util.spec_from_file_location("repair_migration_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
repair = _MODULE.repair


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _trace():
    criteria = (
        ("paid_acquisition_scaling", "Paid Acquisition", 0.3),
        ("creative_strategy_testing", "Creative Strategy", 0.25),
        ("attribution_data_analysis", "Attribution", 0.25),
        ("ai_native_leverage", "AI Workflow", 0.2),
    )
    rubric = []
    for key, title, weight in criteria:
        rubric.extend(
            [
                f'- {key} — "{title}" (weight {weight}; read from: cv, form)',
                "    5 = Proven work at exceptional scope.",
                "    3 = Independent work at ordinary scope.",
                "    1 = No evidence of this work.",
            ]
        )
    system = "\n".join(["SCORING RUBRIC (role-specific, human-approved):", *rubric])
    return {
        "trace_id": "case-acquisition",
        "task_key": "cv_scoring",
        "steps": [
            {
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "<candidate_data>Evidence.</candidate_data>"},
                ],
                "output": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "scores": {key: 3 for key, _, _ in criteria},
                            "overall_score": 3,
                            "recommendation": "MAYBE",
                            "rationale": "Recorded rationale.",
                        }
                    ),
                },
            }
        ],
    }


def test_repairs_ledger_outcome_and_smoke_from_raw_visible_output(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    spec = {
        "inference_cap_usd": 23.5,
        "stage_caps": {"smoke": 1.5},
    }
    (root / "spec.json").write_text(json.dumps(spec))
    (root / "manifest.json").write_text(json.dumps({"spec_sha256": _digest(spec)}))
    (root / "train.json").write_text(json.dumps([_trace()]))

    ledger = Ledger(
        root / "ledger.sqlite", 23.5, {"smoke": 1.5}, artifact_dir=root / "calls"
    )
    call_id = "candidate/smoke/case-acquisition/doubleword-async/method/0"
    visible = json.dumps(
        {
            "scores": {
                "paid_acquisition_scaling": 3,
                "creative_strategy_testing": 3,
                "attribution_data_analysis": 3,
                "ai_native_leverage": 3,
            },
            "recommendation": "MAYBE",
            "rationale": "Visible grounded rationale.",
        }
    )
    raw = {
        "id": "resp-live-shape",
        "status": "completed",
        "service_tier": "flex",
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "private reasoning"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": visible}],
            },
        ],
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }
    contaminated = "private reasoning" + visible
    result = CallResult(
        call_id=call_id,
        route_id="doubleword-async",
        output_text=contaminated,
        usage=raw["usage"],
        cost_usd=0.001,
        cost_kind="derived",
        latency_s=42.5,
        raw=raw,
        upstream=None,
        requested_service_tier="flex",
        echoed_service_tier="flex",
        service_tier_verified=True,
        cache_hit=False,
        response_id="resp-live-shape",
    )
    ledger.reserve(
        call_id=call_id,
        fingerprint="fixture",
        route_id="doubleword-async",
        stage="smoke",
        api="responses",
        reserved_usd=0.01,
        input_token_bound=5000,
        output_token_bound=100,
        request_artifact="request.json",
    )
    ledger.update(
        call_id,
        status="completed",
        accounted_nano=1_000_000,
        result_json=json.dumps(result.as_dict()),
        cost_usd="0.001",
        cost_kind="derived",
        latency_s=42.5,
        response_id="resp-live-shape",
        response_artifact="raw-response.json",
    )

    outcome = {
        "case_id": "case-acquisition",
        "role": "acquisition",
        "route": "doubleword-async",
        "stage": "smoke",
        "methodology_sha256": "method",
        "call_id": call_id,
        "call": result.as_dict(),
        "status": "invalid_output",
        "reason": "candidate output is not valid JSON",
    }
    outcome_path = root / "outcomes" / f"{_digest(call_id)}.json"
    outcome_path.parent.mkdir()
    outcome_path.write_text(json.dumps(outcome))
    judges = {"judge-sol": {"status": "model_failure", "score": 0}}
    smoke = {
        "status": "needs_review",
        "rows": [{"case_id": "case-acquisition", "outcome": outcome, "judges": judges}],
        "controls": [],
        "used_usd": 0.001,
    }
    (root / "smoke.json").write_text(json.dumps(smoke))

    summary = repair(root)

    assert summary["status"] == "repaired"
    assert summary["ledger_calls_repaired"] == 1
    assert summary["outcomes_repaired"] == 1
    repaired_outcome = json.loads(outcome_path.read_text())
    assert repaired_outcome["status"] == "valid"
    assert repaired_outcome["call"]["output_text"] == visible
    assert "reason" not in repaired_outcome
    repaired_smoke = json.loads((root / "smoke.json").read_text())
    assert repaired_smoke["rows"][0]["outcome"] == repaired_outcome
    assert repaired_smoke["rows"][0]["judges"] == judges

    live_row = ledger.get(call_id)
    assert live_row["status"] == "completed"
    assert live_row["cost_usd"] == "0.001"
    assert live_row["latency_s"] == 42.5
    assert json.loads(live_row["result_json"])["output_text"] == visible

    backup = root / "amendments/pre-visible-output-repair"
    assert (backup / "smoke.json").exists()
    assert (backup / outcome_path.relative_to(root)).exists()
    with sqlite3.connect(backup / "ledger.sqlite") as conn:
        old_result = json.loads(
            conn.execute("SELECT result_json FROM calls WHERE call_id = ?", (call_id,)).fetchone()[0]
        )
    assert old_result["output_text"] == contaminated
