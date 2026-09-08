import json
import subprocess
import sys
import threading
from copy import deepcopy

import pytest

from compound import cache_study as study
from compound.providers_registry import parse_providers
from compound.serving_report import aggregate, build_report, read_results

SHAPES = {"document": {"messages": [{"role": "user", "content": "A document."}], "max_tokens": 16}}


def test_prefix_policies_and_input_preservation():
    original = deepcopy(SHAPES["document"])
    for policy in study.REUSE:
        first = study.payload(original, "a" * 32, policy, 0)
        next_ = study.payload(original, "a" * 32, policy, 1)
        assert (first == next_) == (policy == "exact")
        assert (first["messages"][0] == next_["messages"][0]) == (policy != "none")
    assert original == SHAPES["document"]


def test_priming_finishes_before_delay_and_burst_and_reports_stay_separate(tmp_path):
    specs = parse_providers("openrouter/auto,openrouter/relace")
    conditions = study.plan(specs, SHAPES, ["exact", "prefix", "none"], [0, 1], [2], 1)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    events = []
    clock = [0.0]

    def call(spec, model, mode, name, shape, round_no, rep, **kwargs):
        with lock:
            events.append(("call", rep))
        if rep:
            barrier.wait(timeout=5)  # proves both probes are submitted together
        return dict(
            route=spec.label,
            model=model,
            shape=name,
            mode=mode,
            status=200,
            temperature=0,
            max_tokens=16,
            prompt_tokens=100,
            cached_tokens=80,
            cost_usd=0.001,
        )

    def sleep(seconds):
        assert events[-1] == ("call", 0)
        events.append(("sleep", seconds))
        clock[0] += seconds

    out = study.run(
        specs,
        "m",
        SHAPES,
        conditions,
        tmp_path / "run",
        call=call,
        sleep=sleep,
        clock=lambda: clock[0],
    )
    raw = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(raw) == 36
    assert len([r for r in raw if r["phase"] == "prime"]) == 12
    assert all(r["actual_idle_s"] == r["idle_s"] for r in raw if r["phase"] == "probe")
    assert len({r["prompt_sha256"] for r in raw if r["phase"] == "prime"}) == 12
    rows, _ = read_results([out])
    cells = aggregate(rows)
    assert len(cells) == 24  # never pools reuse policy, delay, route or prime/probe
    assert all(c["n"] == (1 if c["cache_mode"] == "prime" else 2) for c in cells)
    html = build_report([out], tmp_path / "report.html").read_text()
    assert "Prime ○ → probe ●" in html
    assert "Prefix reuse: exact" in html
    assert "Prefix reuse: none" in html
    with pytest.raises(FileExistsError):
        study.run(specs, "m", SHAPES, conditions, tmp_path / "run", call=call)


def test_call_limit_before_output_or_calls(tmp_path):
    specs = parse_providers("openrouter/auto")
    conditions = study.plan(specs, SHAPES, ["exact"], [0], [4], 1)
    with pytest.raises(ValueError, match="exceeds"):
        study.run(specs, "m", SHAPES, conditions, tmp_path / "run", max_calls=4)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        dict(delays=[float("nan")]),
        dict(delays=[-1]),
        dict(concurrency=[0]),
        dict(trials=0),
        dict(reuse=["bad"]),
        dict(reuse=["exact", "exact"]),
        dict(shapes={"x": {"messages": [{}]}}),
        dict(specs=[]),
    ],
)
def test_invalid_plan(overrides):
    args = dict(
        specs=parse_providers("openrouter/auto"),
        shapes=SHAPES,
        reuse=["exact"],
        delays=[0],
        concurrency=[1],
        trials=1,
    )
    with pytest.raises(ValueError):
        study.plan(**{**args, **overrides})


def test_cli_dry_run_counts_every_call_without_writes(tmp_path):
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(SHAPES))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compound.bench",
            "cache-study",
            "--providers",
            "openrouter/auto,openrouter/relace",
            "--model-or",
            "m",
            "--shapes",
            str(path),
            "--out",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "252 calls" in result.stdout
    assert "dry run" in result.stdout
    assert not (tmp_path / "run").exists()


def test_doubleword_model_and_markers_required(tmp_path, monkeypatch):
    specs = parse_providers("openrouter/auto,doubleword/realtime,doubleword/flex")
    conditions = study.plan(specs, SHAPES, ["exact"], [0], [1], 1)
    monkeypatch.setenv("COMPOUND_DW_CACHE", "0")
    with pytest.raises(ValueError, match="cache markers"):
        study.run(specs, "or-model", SHAPES, conditions, tmp_path / "run", direct_model="dw-model")
    assert not (tmp_path / "run").exists()
    monkeypatch.setenv("COMPOUND_DW_CACHE", "1")
    seen = []

    def call(spec, model, mode, name, shape, *args, **kwargs):
        body = study.sm.build_body(spec, model, mode, shape, nonce="")
        if spec.kind == "doubleword":
            assert "cache_control" in json.dumps(body)
            assert model == "dw-model"
        else:
            assert model == "or-model"
        seen.append(spec.token)
        return {"status": 200}

    study.run(
        specs, "or-model", SHAPES, conditions, tmp_path / "run", direct_model="dw-model", call=call
    )
    assert len(seen) == 6


def test_paid_cli_loads_env_without_real_requests(tmp_path, monkeypatch):
    from compound import bench

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-placeholder\n")
    (tmp_path / "shapes.json").write_text(json.dumps(SHAPES))
    monkeypatch.setattr(bench, "_load_providers_config", lambda: {})
    invoked = []

    def fake_run(*args, **kwargs):
        import os

        assert os.environ["OPENROUTER_API_KEY"] == "test-placeholder"
        invoked.append(True)
        return tmp_path / "results.jsonl"

    monkeypatch.setattr(study, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compound-bench",
            "cache-study",
            "--providers",
            "openrouter/auto",
            "--model-or",
            "m",
            "--shapes",
            "shapes.json",
            "--go",
        ],
    )
    assert bench.main() == 0
    assert invoked == [True]
