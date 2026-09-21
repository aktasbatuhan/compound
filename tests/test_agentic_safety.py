import json
from pathlib import Path

import pytest

from compound import agentic_safety


def _cache_call(*, tier="standard", cached=0, usage=True, role="agent"):
    row = {
        "route": "deepseek-dw",
        "requested_model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "requested_endpoint": "chat_completions",
        "request_url": "https://api.doubleword.ai/v1/chat/completions",
        "requested_tier": tier,
        "role": role,
        "cache_requested": True,
    }
    if usage:
        row["usage"] = {
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
    return row


def test_cache_gate_keeps_standard_and_flex_scopes_separate():
    calls = [
        *[_cache_call(tier="standard", cached=90) for _ in range(8)],
        *[_cache_call(tier="flex", cached=0) for _ in range(8)],
    ]

    failures = agentic_safety.cache_failures(calls)

    assert len(failures) == 1
    assert failures[0]["scope"][4] == "flex"


def test_cache_gate_treats_missing_usage_as_missing_evidence():
    calls = [_cache_call(usage=False) for _ in range(8)]

    failures = agentic_safety.cache_failures(calls)

    assert len(failures) == 1
    assert failures[0]["reason"] == "missing usage or no recent cache hits"


def test_cache_gate_uses_only_the_recent_sliding_window():
    calls = [
        *[_cache_call(cached=90) for _ in range(8)],
        *[_cache_call(cached=0) for _ in range(8)],
    ]

    failures = agentic_safety.cache_failures(calls, minimum=8)

    assert len(failures) == 1
    assert failures[0]["scope"][4] == "standard"


def _seal_inputs():
    spec = {"schema_version": 1, "controls": {"fixed": True}}
    study = {
        "schema_version": 1,
        "spec_sha256": "study-spec-hash",
        "episode_count": 0,
        "episodes": [],
    }
    return spec, study


def test_seal_refuses_changed_spec_and_execution_mode(tmp_path):
    spec, study = _seal_inputs()
    agentic_safety.seal_run(tmp_path, spec, study, "sequential")
    agentic_safety.seal_run(tmp_path, spec, study, "sequential")

    with pytest.raises(ValueError, match="saved specification differs"):
        agentic_safety.seal_run(
            tmp_path,
            spec | {"controls": {"fixed": False}},
            study,
            "sequential",
        )

    with pytest.raises(ValueError, match="run identity changed"):
        agentic_safety.seal_run(tmp_path, spec, study, "parallel_routes")


def test_reference_qualification_is_required_and_bound_to_spec(tmp_path):
    spec = {"sources": {"retail": {"tasks": [{"id": "a"}]}}, "controls": {}}
    assert agentic_safety.preparation_errors(spec, tmp_path)
    evidence = {
        "spec_sha256": agentic_safety.fingerprint(spec),
        "suite": "retail",
        "ok": True,
        "results": [{"task_id": "a", "state_changed": True, "gold_replay_errors": []}],
    }
    path = tmp_path / "reference-qualification.json"
    path.write_text(json.dumps(evidence))
    assert not agentic_safety.preparation_errors(spec, tmp_path)
    evidence["results"][0]["state_changed"] = False
    path.write_text(json.dumps(evidence))
    assert agentic_safety.preparation_errors(spec, tmp_path)
    evidence["results"][0]["state_changed"] = True
    evidence["spec_sha256"] = "different"
    path.write_text(json.dumps(evidence))
    assert agentic_safety.preparation_errors(spec, tmp_path)


def test_seal_refuses_changed_runtime(tmp_path, monkeypatch):
    spec, study = _seal_inputs()
    agentic_safety.seal_run(tmp_path, spec, study, "sequential")
    original = Path.read_bytes

    def changed_runtime(path):
        if path.name == "agentic_run.py":
            return b"changed runtime for test"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed_runtime)

    with pytest.raises(ValueError, match="run identity changed"):
        agentic_safety.seal_run(tmp_path, spec, study, "sequential")


@pytest.mark.parametrize("ledger_name", ["calls.jsonl", "outcomes.jsonl", "spend.json"])
def test_seal_never_adopts_an_existing_unsealed_ledger(tmp_path, ledger_name):
    spec, study = _seal_inputs()
    (tmp_path / ledger_name).write_text("{}\n")

    with pytest.raises(ValueError, match="existing unsealed experiment"):
        agentic_safety.seal_run(tmp_path, spec, study, "sequential")

    assert not (tmp_path / "run-manifest.json").exists()


def _qualification_spec(*, retail=False):
    spec = {
        "models": [
            {
                "id": "deepseek-dw",
                "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            }
        ],
        "controls": {},
        "sources": {},
    }
    if retail:
        spec["sources"]["retail"] = {}
        spec["controls"]["simulator_route"] = "gpt-sol"
        spec["auxiliary_models"] = [{"id": "gpt-sol", "model": "openai/gpt-5.6-sol"}]
    return spec


def _qualified_rows(model_id, model, tier, role="agent", episode_id=None):
    eid = episode_id or f"probe-{model_id}-{tier}"
    common = {
        "episode_id": eid,
        "role": role,
        "requested_model": model,
        "requested_tier": tier,
        "status": 200,
        "tier_confirmed": True,
        "cache_requested": True,
    }
    return [
        common | {"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}},
        common | {"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}}},
    ]


def _write_qualification(directory, calls, probes):
    (directory / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls))
    (directory / "probes.json").write_text(json.dumps(probes) + "\n")


@pytest.mark.parametrize("missing", ["tier", "cache", "tool"])
def test_qualification_requires_actual_tier_cache_and_tool_evidence(tmp_path, missing):
    spec = _qualification_spec()
    calls = []
    probes = []
    for tier in ("standard", "flex"):
        rows = _qualified_rows("deepseek-dw", "deepseek-ai/DeepSeek-V4.1-Flash", tier)
        if tier == "flex" and missing == "tier":
            rows[1]["tier_confirmed"] = False
        if tier == "flex" and missing == "cache":
            for row in rows:
                row["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
        calls.extend(rows)
        if not (tier == "flex" and missing == "tool"):
            probes.append({"episode_id": f"probe-deepseek-dw-{tier}", "role": "agent", "ok": True})
    _write_qualification(tmp_path, calls, probes)

    errors = agentic_safety.qualification_errors(spec, tmp_path)

    assert errors
    assert all("deepseek-dw/flex/agent" in error for error in errors)
    if missing == "cache":
        assert any("no verified repeated-prefix cache hit" in error for error in errors)
    else:
        assert any("tools or tier evidence unqualified" in error for error in errors)


def test_qualification_accepts_dedicated_simulator_probe_identity(tmp_path):
    spec = _qualification_spec(retail=True)
    calls = []
    probes = []
    for tier in ("standard", "flex"):
        calls.extend(_qualified_rows("deepseek-dw", "deepseek-ai/DeepSeek-V4.1-Flash", tier))
        probes.append({"episode_id": f"probe-deepseek-dw-{tier}", "role": "agent", "ok": True})
    calls.extend(
        _qualified_rows(
            "gpt-sol",
            "openai/gpt-5.6-sol",
            "standard",
            role="auxiliary",
            episode_id="probe-simulator-standard",
        )
    )
    probes.append(
        {
            "episode_id": "probe-simulator-standard",
            "role": "auxiliary",
            "ok": True,
        }
    )
    _write_qualification(tmp_path, calls, probes)

    assert agentic_safety.qualification_errors(spec, tmp_path) == []


def test_requested_policy_allows_absence_but_never_contradiction():
    from compound.agentic_safety import tier_evidence_errors

    spec = {"controls": {"tier_evidence_policy": "requested_policy"}}
    call = {"status": 200, "requested_tier": "flex", "tier_confirmed": False}
    assert not tier_evidence_errors(spec, [call])
    assert call["tier_confirmed"] is False
    assert tier_evidence_errors({}, [call])
    assert tier_evidence_errors(spec, [call | {"served_tier": "priority"}])
    with pytest.raises(ValueError, match="unknown tier evidence policy"):
        tier_evidence_errors({"controls": {"tier_evidence_policy": "typo"}}, [])


def test_requested_policy_qualification_keeps_cache_and_tools_required(tmp_path):
    spec = _qualification_spec()
    spec.setdefault("controls", {})["tier_evidence_policy"] = "requested_policy"
    calls, probes = [], []
    for tier in ("standard", "flex"):
        rows = _qualified_rows("deepseek-dw", "deepseek-ai/DeepSeek-V4.1-Flash", tier)
        for row in rows:
            row["tier_confirmed"] = False
        calls.extend(rows)
        probes.append({"episode_id": f"probe-deepseek-dw-{tier}", "role": "agent", "ok": True})
    _write_qualification(tmp_path, calls, probes)
    assert not agentic_safety.qualification_errors(spec, tmp_path)
    calls[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    _write_qualification(tmp_path, calls, probes)
    assert agentic_safety.qualification_errors(spec, tmp_path)


def test_admission_gate_requires_exact_coverage_and_runtime(tmp_path, monkeypatch):
    spec = {
        "sources": {"retail": {"tasks": [{"id": "a"}]}},
        "controls": {"require_admission_qualification": True},
    }
    monkeypatch.setattr(agentic_safety, "runtime_identity", lambda: {"python": "pinned"})

    def blocked():
        return any("admission" in x for x in agentic_safety.preparation_errors(spec, tmp_path))

    assert blocked()
    proof = {
        "spec_sha256": agentic_safety.fingerprint(spec),
        "ok": True,
        "runtime": {"python": "pinned"},
        "checks": [
            {"task_id": "a", "tier": t, "role": r, "admitted": True}
            for t in ("standard", "flex")
            for r in ("agent", "auxiliary")
        ],
    }
    path = tmp_path / "admission-qualification.json"
    path.write_text(json.dumps(proof))
    assert not blocked()
    proof["checks"].pop()
    path.write_text(json.dumps(proof))
    assert blocked()
