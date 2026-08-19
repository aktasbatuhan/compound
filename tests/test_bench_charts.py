from __future__ import annotations

from compound.bench_charts import radar_axes, render_charts


def _summary():
    return {
        "hosts": {
            "deepinfra": {
                "episodes": 42, "infra_errors": 2, "accuracy": 0.50,
                "cost_per_task_usd": 0.002, "median_latency_s": 21.2, "median_tps": 79.4,
            },
            "doubleword-realtime": {
                "episodes": 42, "infra_errors": 0, "accuracy": 0.55,
                "cost_per_task_usd": 0.0065, "median_latency_s": 4.1, "median_tps": 56.1,
            },
            "novita": {
                "episodes": 42, "infra_errors": 42, "accuracy": 0.0,
                "cost_per_task_usd": None, "median_latency_s": 26.5, "median_tps": None,
            },
        }
    }


def _rows():
    rows = []
    for host, solved in (("deepinfra", [3, 0, 2]), ("doubleword-realtime", [3, 3, 1]),
                         ("novita", [0, 0, 0])):
        for i, s in enumerate(solved):
            rows.append({"host": host, "task": f"t{i}", "ctx_tokens": "1000",
                         "trials": "3", "solved": str(s), "success_rate": str(s / 3)})
    return rows


def test_radar_axes_values_and_determinism():
    axes = radar_axes(_summary(), _rows())
    di = axes["deepinfra"]
    assert di["quality"] == 0.50
    assert abs(di["reliability"] - 40 / 42) < 1e-9
    assert abs(di["speed"] - 1 / 21.2) < 1e-9
    assert di["TPS"] == 79.4
    assert abs(di["cost"] - 1 / 0.002) < 1e-9
    # one of deepinfra's three tasks is mixed (2/3) -> determinism 1 - 1/3
    assert abs(di["determinism"] - (1 - 1 / 3)) < 1e-9
    # a host that never solves anything gets no determinism credit for
    # failing consistently
    assert axes["novita"]["determinism"] == 0.0
    # ...nor speed credit: failing instantly is not fast serving
    assert "speed" not in axes["novita"] or axes["novita"]["speed"] == 0.0


def test_radar_axes_drops_axes_missing_on_most_hosts():
    summary = _summary()
    for h in summary["hosts"]:
        summary["hosts"][h]["cost_per_task_usd"] = None
    axes = radar_axes(summary, _rows())
    assert "cost" not in axes["deepinfra"]  # <2 hosts have cost data
    assert "quality" in axes["deepinfra"]


def test_render_charts_includes_provider_profiles(tmp_path):
    import csv

    with (tmp_path / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["host", "task", "ctx_tokens", "trials",
                                          "solved", "success_rate"])
        w.writeheader()
        w.writerows(_rows())
    path = render_charts(_summary(), tmp_path)
    doc = path.read_text()
    assert "Provider profiles" in doc
    assert doc.count("<polygon") >= 9  # 3 hosts x (2 rings + 1 shape)
    assert "Success vs context window" in doc
