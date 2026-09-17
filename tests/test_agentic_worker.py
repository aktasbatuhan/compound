import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from compound import agentic_worker


def test_official_empty_patch_result_is_task_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    constants = ModuleType("swebench.harness.constants")
    constants.RUN_EVALUATION_LOG_DIR = tmp_path / "grader-logs"
    for name in ("swebench", "swebench.harness"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "swebench.harness.constants", constants)
    instance = {
        "instance_id": "example-1",
        "problem_statement": "Fix a bug",
        "patch": "",
        "test_patch": "",
        "base_commit": "abc",
        "FAIL_TO_PASS": ["test_fix"],
        "PASS_TO_PASS": ["test_existing"],
        "image": "fake",
    }
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps([instance]))
    metadata = tmp_path / ".compound/sources/swe-verified-grader.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps([instance]))
    spec = {"sources": {"coding": {"data_path": str(frozen), "grader_revision": "pinned-test"}}}
    episode = {"episode_id": "episode-1", "task_id": "example-1"}
    output = tmp_path / "episode-1"
    output.mkdir()

    def grade(cmd, **kwargs):
        prediction = json.loads((output / "predictions.jsonl").read_text())
        assert prediction["model_patch"] == ""
        report = constants.RUN_EVALUATION_LOG_DIR / "episode-1" / "results.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"empty_patch_ids": ["example-1"], "error_ids": []}))
        # Official grader intentionally creates no per-instance report for an empty patch.

    monkeypatch.setattr(agentic_worker.subprocess, "run", grade)
    result = agentic_worker.coding(spec, episode, "http://unused", output, oracle=True)
    assert result["status"] == "graded"
    assert result["success"] is False
    assert result["failure_reason"] == "empty_patch"
    assert result["grader"] == "swebench@pinned-test"
    assert (output / "official.json").exists()


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "AssistantMessage must have either content or tool calls. Got AssistantMessage",
            "provider_error",
        ),
        ("UserMessage must have either content or tool calls.", "infrastructure_error"),
        ("missing retail grade", "infrastructure_error"),
    ],
)
def test_empty_agent_message_does_not_stop_other_episodes(tmp_path, monkeypatch, message, expected):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    out = tmp_path / "episode"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--spec",
            str(spec),
            "--episode",
            json.dumps({"suite": "retail"}),
            "--base",
            "http://unused",
            "--out",
            str(out),
        ],
    )

    def failure(*args):
        raise ValueError(message)

    monkeypatch.setattr(agentic_worker, "retail", failure)
    agentic_worker.main()
    result = json.loads((out / "outcome.json").read_text())
    assert result["status"] == expected
    assert result["success"] is None
    if expected == "provider_error":
        assert result["failure_reason"] == "empty_agent_message"
    assert result["error_message"] == message
    assert result["failure_artifact"] == "failure.json"
    failure = json.loads((out / "failure.json").read_text())
    assert failure["error_type"] == "ValueError"
    assert failure["error_message"] == message
    assert "raise ValueError(message)" in failure["traceback"]


def test_retail_uses_absolute_tau_checkpoint_and_requires_it(tmp_path, monkeypatch):
    litellm = ModuleType("litellm")
    litellm.completion = lambda *args, **kwargs: None
    tau2 = ModuleType("tau2")
    tau2_utils = ModuleType("tau2.utils")
    tau2_utils_utils = ModuleType("tau2.utils.utils")
    tau2_utils_utils.DATA_DIR = Path("tau-data")
    tau_llm = ModuleType("tau2.utils.llm_utils")
    tau_llm.completion = lambda *args, **kwargs: None
    tau_data_model = ModuleType("tau2.data_model")
    tau_simulation = ModuleType("tau2.data_model.simulation")

    class RunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    tau_simulation.RunConfig = RunConfig
    tau_run = ModuleType("tau2.run")
    captured = {}

    def run_domain(config):
        captured["save_to"] = config.save_to
        Path(config.save_to + ".json").write_text("{}\n")
        reward = SimpleNamespace(reward=1.0, model_dump=lambda **kwargs: {"reward": 1.0})
        simulation = SimpleNamespace(reward_info=reward, duration=0.25)
        return SimpleNamespace(simulations=[simulation])

    tau_run.run_domain = run_domain
    for name, module in {
        "litellm": litellm,
        "tau2": tau2,
        "tau2.utils": tau2_utils,
        "tau2.utils.utils": tau2_utils_utils,
        "tau2.utils.llm_utils": tau_llm,
        "tau2.data_model": tau_data_model,
        "tau2.data_model.simulation": tau_simulation,
        "tau2.run": tau_run,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    output = tmp_path / "nested" / "episode"
    output.mkdir(parents=True)
    spec = {"sources": {"retail": {"max_steps": 4, "revision": "pinned-test"}}}
    episode = {"task_id": "retail-1", "trial": 7}

    result = agentic_worker.retail(spec, episode, "http://127.0.0.1:9/agent/v1", output)

    assert captured["save_to"] == str((output / "official").resolve())
    assert (output / "official.json").is_file()
    assert result == {
        "status": "graded",
        "success": True,
        "duration_s": 0.25,
        "metrics": {"reward": 1.0},
        "grader": "tau2-verified@pinned-test",
    }


def test_retail_rejects_missing_tau_checkpoint(tmp_path, monkeypatch):
    litellm = ModuleType("litellm")
    litellm.completion = lambda *args, **kwargs: None
    tau_llm = ModuleType("tau2.utils.llm_utils")
    tau_llm.completion = lambda *args, **kwargs: None
    tau2_utils_utils = ModuleType("tau2.utils.utils")
    tau2_utils_utils.DATA_DIR = Path("tau-data")
    tau_simulation = ModuleType("tau2.data_model.simulation")
    tau_simulation.RunConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    tau_run = ModuleType("tau2.run")
    tau_run.run_domain = lambda config: SimpleNamespace(simulations=[])
    for name, module in {
        "litellm": litellm,
        "tau2": ModuleType("tau2"),
        "tau2.utils": ModuleType("tau2.utils"),
        "tau2.utils.utils": tau2_utils_utils,
        "tau2.utils.llm_utils": tau_llm,
        "tau2.data_model": ModuleType("tau2.data_model"),
        "tau2.data_model.simulation": tau_simulation,
        "tau2.run": tau_run,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    output = tmp_path / "episode"
    output.mkdir()
    spec = {"sources": {"retail": {"max_steps": 4, "revision": "pinned-test"}}}

    with pytest.raises(FileNotFoundError, match="expected checkpoint"):
        agentic_worker.retail(
            spec,
            {"task_id": "retail-1", "trial": 1},
            "http://127.0.0.1:9/agent/v1",
            output,
        )


def _stub_tau_data_dir(monkeypatch, data_dir):
    utils = ModuleType("tau2.utils.utils")
    utils.DATA_DIR = data_dir
    for name in ("tau2", "tau2.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "tau2.utils.utils", utils)


def test_absolute_stem_survives_tau_data_dir_join(tmp_path, monkeypatch):
    """The absolute-stem trick must keep working on the pinned tau revision."""
    _stub_tau_data_dir(monkeypatch, Path(".compound/sources/tau2-bench-verified/data"))
    stem = (tmp_path / "episode" / "official").resolve()
    assert agentic_worker.tau_checkpoint_path(stem) == stem.parent / "official.json"


def test_relative_stem_is_refused_before_any_paid_episode(tmp_path, monkeypatch):
    """A relative --out silently relocates the checkpoint under tau's data dir.

    That previously surfaced only after a full episode had already been paid
    for, so it must fail up front instead.
    """
    _stub_tau_data_dir(monkeypatch, tmp_path / "tau-data")
    with pytest.raises(FileNotFoundError) as caught:
        agentic_worker.tau_checkpoint_path(Path("artifacts/run/episode/official"))
    assert "relocated" in str(caught.value)
    assert "absolute --out" in str(caught.value)


def test_stem_containing_a_dot_keeps_its_full_name(tmp_path, monkeypatch):
    """``with_suffix`` would truncate at the dot and look for the wrong file."""
    _stub_tau_data_dir(monkeypatch, Path("data"))
    stem = (tmp_path / "run-v1.2" / "official").resolve()
    assert agentic_worker.tau_checkpoint_path(stem).name == "official.json"
    stem = (tmp_path / "official.v2").resolve()
    assert agentic_worker.tau_checkpoint_path(stem).name == "official.v2.json"
