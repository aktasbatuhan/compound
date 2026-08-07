from __future__ import annotations

import json
from pathlib import Path

from compound.bench_report import build_report, iter_episodes


def _sim(task, trial, reward, prompt, completion, cost, provider, tier=None, term="user_stop"):
    return {
        "task_id": task,
        "trial": trial,
        "termination_reason": term,
        "reward_info": {"reward": reward},
        "messages": [
            {
                "role": "assistant",
                "content": "ok",
                "generation_time_seconds": 2.0,
                "raw_data": {
                    "provider": provider,
                    "service_tier": tier,
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "cost": cost,
                    },
                },
            }
        ],
    }


def _write_host(run: Path, host: str, sims: list[dict]) -> None:
    d = run / host / "episodes" / f"{host}.json"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({"simulations": sims}))


def _build(tmp_path: Path) -> Path:
    run = tmp_path / "sweep"
    _write_host(run, "deepinfra-fp4", [
        _sim("6", 0, 1.0, 5000, 300, 0.0026, "DeepInfra"),
        _sim("20", 0, 0.0, 9000, 400, 0.004, "DeepInfra"),
    ])
    _write_host(run, "doubleword-flex", [
        _sim("6", 0, 1.0, 5000, 300, 0.0, "Doubleword", tier="flex"),
        _sim("20", 0, 1.0, 9000, 400, 0.0, "Doubleword", tier="flex"),
    ])
    return run


def test_iter_episodes_extracts_cost_tokens_and_served_by(tmp_path):
    run = _build(tmp_path)
    eps = iter_episodes(run)
    assert len(eps) == 4
    di = [e for e in eps if e.host == "deepinfra-fp4" and e.task == "6"][0]
    assert di.solved and di.cost_usd == 0.0026 and di.prompt_tokens == 5000
    assert di.served_by == ["DeepInfra"]
    flex = [e for e in eps if e.host == "doubleword-flex" and e.task == "6"][0]
    assert flex.service_tier_echo == ["flex"]


def test_build_report_writes_all_artifacts_and_fills_dw_cost(tmp_path):
    run = _build(tmp_path)
    # Doubleword reports no per-call cost -> derive from declared price.
    summary = build_report(run, {"doubleword-flex": (0.70, 2.25)})
    report = run / "report"
    for name in ("episodes.csv", "per_task.csv", "summary.json", "transcripts.jsonl", "charts.html"):
        assert (report / name).exists(), name

    di = summary["hosts"]["deepinfra-fp4"]
    assert di["accuracy"] == 0.5  # 1 of 2 solved
    assert di["cost_per_task_usd"] > 0  # from OpenRouter's own figures

    flex = summary["hosts"]["doubleword-flex"]
    assert flex["accuracy"] == 1.0
    # cost derived from declared price and token counts, not left at zero
    assert flex["cost_per_task_usd"] > 0
    assert flex["service_tier_echo"] == ["flex"]

    dist = summary["context_vs_success"]
    assert dist["8k+"]["total"] == 2  # the two 9k-token episodes


def test_context_chart_and_cost_chart_present(tmp_path):
    run = _build(tmp_path)
    build_report(run, {"doubleword-flex": (0.70, 2.25)})
    html = (run / "report" / "charts.html").read_text()
    assert "Success vs context window" in html
    assert "Cost vs quality" in html
    assert "doubleword-flex" in html
