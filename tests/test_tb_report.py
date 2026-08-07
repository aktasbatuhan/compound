from __future__ import annotations

import json
from pathlib import Path

from compound.tb_report import build_report, iter_trials


def _results(trials):
    return {
        "n_resolved": sum(t["is_resolved"] for t in trials),
        "n_unresolved": sum(not t["is_resolved"] for t in trials),
        "accuracy": sum(t["is_resolved"] for t in trials) / len(trials),
        "results": trials,
    }


def _trial(task, resolved, tin, tout, start, end, fail="unset"):
    return {
        "task_id": task,
        "is_resolved": resolved,
        "failure_mode": fail,
        "total_input_tokens": tin,
        "total_output_tokens": tout,
        "agent_started_at": start,
        "agent_ended_at": end,
    }


def _write(run: Path, host: str, trials: list[dict]) -> None:
    d = run / host / "2026-08-08__00-00-00"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(_results(trials)))


def _build(tmp_path: Path) -> Path:
    run = tmp_path / "tb"
    _write(run, "deepinfra-fp4", [
        _trial("create-bucket", True, 5000, 400, "2026-08-08T00:00:00", "2026-08-08T00:00:20"),
        _trial("crack-7z-hash", False, 9000, 800, "2026-08-08T00:01:00", "2026-08-08T00:02:00",
               fail="test_failed"),
    ])
    _write(run, "doubleword-flex", [
        _trial("create-bucket", True, 5000, 400, "2026-08-08T00:00:00", "2026-08-08T00:00:40"),
        _trial("crack-7z-hash", True, 9000, 800, "2026-08-08T00:01:00", "2026-08-08T00:03:00"),
    ])
    return run


def test_iter_trials_reads_resolution_tokens_latency(tmp_path):
    trials = iter_trials(_build(tmp_path))
    assert len(trials) == 4
    di = [t for t in trials if t["host"] == "deepinfra-fp4" and t["task"] == "create-bucket"][0]
    assert di["solved"] and di["in_tokens"] == 5000 and di["latency_s"] == 20.0


def test_build_report_matches_shared_shape_and_costs(tmp_path):
    run = _build(tmp_path)
    summary = build_report(run, {"deepinfra-fp4": (0.14, 0.28), "doubleword-flex": (0.70, 2.25)})
    for name in ("summary.json", "episodes.csv", "per_task.csv", "charts.html"):
        assert (run / "report" / name).exists(), name
    di = summary["hosts"]["deepinfra-fp4"]
    assert di["accuracy"] == 0.5  # 1 of 2
    assert di["cost_per_task_usd"] > 0
    assert di["failure_modes"] == {"test_failed": 1}
    flex = summary["hosts"]["doubleword-flex"]
    assert flex["accuracy"] == 1.0
    # flex is slower on the same task -> higher latency than deepinfra
    assert flex["median_latency_s"] > di["median_latency_s"]


def test_charts_render_from_tb_summary(tmp_path):
    run = _build(tmp_path)
    build_report(run, {"deepinfra-fp4": (0.14, 0.28), "doubleword-flex": (0.70, 2.25)})
    html = (run / "report" / "charts.html").read_text()
    assert "Success vs context window" in html
    assert "doubleword-flex" in html
