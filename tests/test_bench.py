import argparse
import json
import os

import pytest

from compound.bench import BENCHMARKS, _apply_tb_env, _require_keys, select_case_ids
from compound.tau_sweep import SweepConfig


def _tb_args(**over):
    """A run-command namespace with the terminal_bench pinning flags."""
    base = {
        "reasoning": None,
        "cache_optin": False,
        "tb_timeout_mult": None,
        "call_ledger": None,
    }
    base.update(over)
    return argparse.Namespace(**base)

CASES = [
    {"case_id": "retail:10", "partition": "optimizer_validation"},
    {"case_id": "retail:33", "partition": "optimizer_train"},
    {"case_id": "airline:3", "partition": "optimizer_train"},
]


def test_probe_requires_explicit_go(monkeypatch, capsys):
    from compound import openrouter_discovery
    from compound.bench import main

    monkeypatch.setattr("sys.argv", ["compound-bench", "providers", "m", "--probe"])
    monkeypatch.setattr(
        openrouter_discovery, "fetch_endpoints", lambda *a: pytest.fail("unexpected network"),
    )
    assert main() == 0
    assert "Add --go" in capsys.readouterr().out


def test_paid_probe_dispatches_only_after_key_preflight(monkeypatch):
    from compound import openrouter_discovery
    from compound.bench import main

    monkeypatch.setattr(
        "sys.argv", ["compound-bench", "providers", "m", "--probe", "--go"],
    )
    monkeypatch.setattr(openrouter_discovery, "fetch_endpoints", lambda *a: [])
    called = []
    monkeypatch.setattr(
        openrouter_discovery, "probe_endpoints", lambda *a: called.append(a) or {},
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="missing required API key"):
        main()
    assert called == []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert main() == 0
    assert len(called) == 1


def test_select_case_ids_filters_and_validates() -> None:
    assert select_case_ids(CASES) == ["retail:10", "retail:33", "airline:3"]
    assert select_case_ids(CASES, partition="optimizer_train") == ["retail:33", "airline:3"]
    assert select_case_ids(CASES, contains="RETAIL") == ["retail:10", "retail:33"]
    # Explicit ids bypass filters but must exist in the manifest.
    assert select_case_ids(CASES, explicit=["airline:3"]) == ["airline:3"]
    with pytest.raises(SystemExit):
        select_case_ids(CASES, explicit=["bogus:1"])


def test_registry_manifests_are_partitioned() -> None:
    for bench in BENCHMARKS.values():
        if not bench.manifest.exists():
            # mmlu's manifest embeds the answer key, so the repo ships without it;
            # it is rebuilt deterministically with `compound-bench prepare mmlu`.
            assert bench.name == "mmlu", f"{bench.name} manifest missing from the repo"
            continue
        cases = json.loads(bench.manifest.read_text())["cases"]
        assert cases, bench.name
        assert all("case_id" in c and "partition" in c for c in cases), bench.name


def test_require_keys_names_every_missing_credential(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    monkeypatch.delenv("DOUBLEWORD_API_KEY", raising=False)
    monkeypatch.delenv("MYHOST_KEY", raising=False)
    # a satisfied key preflights silently
    _require_keys({"OPENROUTER_API_KEY"})
    with pytest.raises(SystemExit) as exc:
        _require_keys({"OPENROUTER_API_KEY", "DOUBLEWORD_API_KEY", "MYHOST_KEY"})
    msg = str(exc.value)
    assert "DOUBLEWORD_API_KEY" in msg and "MYHOST_KEY" in msg
    assert "OPENROUTER_API_KEY" not in msg  # the present one is not reported


def test_apply_tb_env_reasoning_flag_wins_over_preset(monkeypatch) -> None:
    monkeypatch.setenv("COMPOUND_REASONING", "on")
    _apply_tb_env(_tb_args(reasoning="off"))
    import os

    assert os.environ["COMPOUND_REASONING"] == "off"
    # 'default' clears any pre-set value so nothing is injected
    _apply_tb_env(_tb_args(reasoning="default"))
    assert "COMPOUND_REASONING" not in os.environ


@pytest.fixture(autouse=True)
def _isolate_pinning_env():
    """Undo the env vars ``_apply_tb_env`` writes straight into ``os.environ``.

    It sets COMPOUND_REASONING / COMPOUND_DW_CACHE / COMPOUND_TB_TIMEOUT_MULT
    directly, which is the point of the function, but it means a test calling it
    leaks that value into every test that runs later. The default args here have
    ``cache_optin=False``, so without this the whole suite downstream ran with
    cache markers disabled.
    """
    import os

    keys = ("COMPOUND_REASONING", "COMPOUND_DW_CACHE", "COMPOUND_TB_TIMEOUT_MULT")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_apply_tb_env_reasoning_omitted_honors_env(monkeypatch) -> None:
    import os

    monkeypatch.setenv("COMPOUND_REASONING", "on")
    _apply_tb_env(_tb_args(reasoning=None))
    assert os.environ["COMPOUND_REASONING"] == "on"  # untouched


def test_apply_tb_env_cache_optin_sets_env(monkeypatch) -> None:
    import os

    monkeypatch.delenv("COMPOUND_DW_CACHE", raising=False)
    _apply_tb_env(_tb_args(cache_optin=True))
    assert os.environ["COMPOUND_DW_CACHE"] == "1"


def test_apply_tb_env_timeout_mult_shell_wins(monkeypatch) -> None:
    import os

    # a pre-set (shell-exported) multiplier is not overwritten by the flag
    monkeypatch.setenv("COMPOUND_TB_TIMEOUT_MULT", "5")
    _apply_tb_env(_tb_args(tb_timeout_mult=3.0))
    assert os.environ["COMPOUND_TB_TIMEOUT_MULT"] == "5"
    # with nothing pre-set, the flag lands
    monkeypatch.delenv("COMPOUND_TB_TIMEOUT_MULT", raising=False)
    _apply_tb_env(_tb_args(tb_timeout_mult=3.0))
    assert os.environ["COMPOUND_TB_TIMEOUT_MULT"] == "3.0"


def test_sweep_config_custom_provider_key_env() -> None:
    custom = SweepConfig("m", "up", "fp8", 1.0, 2.0, provider="myhost", api_key_env="MY_KEY")
    assert custom.required_key_env() == "MY_KEY"
    assert SweepConfig("m", "up", "q", 1, 2).required_key_env() == "OPENROUTER_API_KEY"
    assert (
        SweepConfig("m", "up", "q", 1, 2, provider="doubleword").required_key_env()
        == "DOUBLEWORD_API_KEY"
    )
    with pytest.raises(ValueError):
        SweepConfig("m", "up", "q", 1, 2, provider="myhost").required_key_env()


def test_apply_tb_env_call_ledger_flag_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("COMPOUND_CALL_LEDGER", raising=False)
    _apply_tb_env(_tb_args(call_ledger="artifacts/run/calls.jsonl"))
    assert os.environ["COMPOUND_CALL_LEDGER"] == "artifacts/run/calls.jsonl"


def test_apply_tb_env_without_the_flag_leaves_recording_off(monkeypatch) -> None:
    monkeypatch.delenv("COMPOUND_CALL_LEDGER", raising=False)
    _apply_tb_env(_tb_args())
    assert "COMPOUND_CALL_LEDGER" not in os.environ


def test_parse_agent_kwargs_splits_pairs() -> None:
    from compound.bench import _parse_agent_kwargs

    assert _parse_agent_kwargs(["max_turns=30", "foo=a=b"]) == {
        "max_turns": "30",
        "foo": "a=b",  # only the first '=' separates
    }
    assert _parse_agent_kwargs(None) == {}


def test_parse_agent_kwargs_rejects_a_malformed_pair() -> None:
    from compound.bench import _parse_agent_kwargs

    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _parse_agent_kwargs(["max_turns"])
