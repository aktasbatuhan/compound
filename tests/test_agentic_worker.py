import json
import sys
from types import ModuleType

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
