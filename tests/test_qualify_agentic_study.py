import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from compound.agentic_study import fingerprint


def _load_script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    sys.modules.pop("qualify_agentic_study", None)
    return importlib.import_module("qualify_agentic_study")


def _install_fake_tau(monkeypatch):
    tasks_module = ModuleType("tau2.data_model.tasks")

    class Task:
        @classmethod
        def model_validate(cls, row):
            actions = [
                SimpleNamespace(
                    name=name,
                    requestor="assistant",
                    arguments={},
                )
                for name in row.get("actions", ["mutate"])
            ]
            criteria = SimpleNamespace(
                actions=actions,
                reward_basis=[SimpleNamespace(value="DB")],
            )
            initial = SimpleNamespace(
                initialization_data=None,
                initialization_actions=None,
                message_history=[],
            )
            return SimpleNamespace(
                id=row["id"],
                evaluation_criteria=criteria,
                initial_state=initial,
            )

    tasks_module.Task = Task
    environment_module = ModuleType("tau2.domains.retail.environment")

    class Environment:
        def __init__(self):
            self.changed = False

        def set_state(self, **kwargs):
            assert set(kwargs) == {
                "initialization_data",
                "initialization_actions",
                "message_history",
            }

        def get_db_hash(self):
            return "after" if self.changed else "before"

        def make_tool_call(self, *, tool_name, requestor, **kwargs):
            assert requestor == "assistant"
            if tool_name == "fail":
                raise RuntimeError("reference action rejected")
            self.changed = True
            return SimpleNamespace(error=None, content="ok")

    environment_module.get_environment = Environment
    modules = {
        "tau2": ModuleType("tau2"),
        "tau2.data_model": ModuleType("tau2.data_model"),
        "tau2.data_model.tasks": tasks_module,
        "tau2.domains": ModuleType("tau2.domains"),
        "tau2.domains.retail": ModuleType("tau2.domains.retail"),
        "tau2.domains.retail.environment": environment_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _write_retail_study(tmp_path, selected, rows, filename="selected-spec.json"):
    data = tmp_path / "retail.json"
    data.write_text(json.dumps(rows) + "\n")
    spec = {
        "schema_version": 1,
        "sources": {
            "retail": {
                "data_path": str(data),
                "tasks": [{"id": task_id} for task_id in selected],
            }
        },
    }
    path = tmp_path / filename
    path.write_text(json.dumps(spec) + "\n")
    return path, spec


def test_retail_uses_selected_spec_and_exact_nested_output(tmp_path, monkeypatch):
    _install_fake_tau(monkeypatch)
    script = _load_script(monkeypatch)
    spec_path, spec = _write_retail_study(
        tmp_path,
        selected=["chosen"],
        rows=[{"id": "not-selected"}, {"id": "chosen"}],
    )
    out = tmp_path / "nested" / "custom-result.json"

    envelope = script.qualify_retail(spec_path, out)

    assert out.is_file()
    assert json.loads(out.read_text()) == envelope
    assert envelope["runtime"] == script.runtime_identity()
    assert {k: v for k, v in envelope.items() if k != "runtime"} == {
        "spec_sha256": fingerprint(spec),
        "suite": "retail",
        "results": [
            {
                "task_id": "chosen",
                "gold_replay_errors": [],
                "state_changed": True,
                "reward_basis": ["DB"],
            }
        ],
        "ok": True,
    }


def test_retail_writes_failed_envelope_before_raising(tmp_path, monkeypatch):
    _install_fake_tau(monkeypatch)
    script = _load_script(monkeypatch)
    spec_path, spec = _write_retail_study(
        tmp_path,
        selected=["broken", "missing"],
        rows=[{"id": "broken", "actions": ["fail"]}],
    )
    out = tmp_path / "failed" / "reference-qualification.json"

    with pytest.raises(ValueError, match="failed reference replay"):
        script.qualify_retail(spec_path, out)

    envelope = json.loads(out.read_text())
    assert envelope["spec_sha256"] == fingerprint(spec)
    assert envelope["suite"] == "retail"
    assert envelope["ok"] is False
    assert [row["task_id"] for row in envelope["results"]] == ["broken"]
    assert envelope["results"][0]["gold_replay_errors"] == [
        "RuntimeError: reference action rejected"
    ]


def test_cli_passes_spec_and_exact_out_without_inference(tmp_path, monkeypatch):
    _install_fake_tau(monkeypatch)
    script = _load_script(monkeypatch)
    spec_path, spec = _write_retail_study(
        tmp_path,
        selected=["chosen"],
        rows=[{"id": "chosen"}],
    )
    out = tmp_path / "cli" / "reference-qualification.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["qualify", "retail", "--spec", str(spec_path), "--out", str(out)],
    )

    script.main()

    envelope = json.loads(out.read_text())
    assert envelope["spec_sha256"] == fingerprint(spec)
    assert envelope["ok"] is True
