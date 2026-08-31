"""Tests for the Harbor / Terminal-Bench 4.0 adapter.

The command builder is pure so the argv a paid run would issue is asserted
here rather than discovered by spending. The result parsing is tested against
Harbor's real schema shape, with the emphasis on keeping a trial that never
reached the verifier out of the pass rate.
"""

from __future__ import annotations

import json

import pytest

from compound.adapters.harbor import (
    DEFAULT_AGENT,
    DEFAULT_DATASET,
    build_command,
    job_result_path,
    load_job_summary,
    proxy_env,
    summarize,
    trial_rows,
)


def cmd(**over):
    base = dict(model="openai/m", jobs_dir="jobs", job_name="job-1")
    base.update(over)
    return build_command(**base)


class TestBuildCommand:
    def test_defaults_pin_the_dataset_version(self):
        # @latest would let a continuously-updated benchmark change the task set
        # between two arms of one experiment.
        assert DEFAULT_DATASET.endswith("@4.0.0")
        assert "--dataset" in cmd()
        assert cmd()[cmd().index("--dataset") + 1] == DEFAULT_DATASET

    def test_core_arguments(self):
        got = cmd(attempts=3, n_concurrent=8, agent=DEFAULT_AGENT)
        assert got[:4] == ["uvx", "--from", "harbor", "harbor"]
        assert got[4] == "run"
        for flag, value in [
            ("--agent", DEFAULT_AGENT),
            ("--model", "openai/m"),
            ("--job-name", "job-1"),
            ("--n-attempts", "3"),
            ("--n-concurrent", "8"),
            ("--env", "docker"),
        ]:
            assert got[got.index(flag) + 1] == value
        assert "--yes" in got  # a sweep must not block on a prompt

    def test_task_filters_repeat_the_flag(self):
        got = cmd(include_tasks=["cad-model", "coq-block-bound"], n_tasks=10)
        assert got.count("--include-task-name") == 2
        assert "cad-model" in got and "coq-block-bound" in got
        assert got[got.index("--n-tasks") + 1] == "10"

    def test_timeout_multiplier_is_delegated_to_harbor(self):
        # Harbor scales limits natively, so nothing on disk is patched.
        assert cmd(timeout_multiplier=3.0)[
            cmd(timeout_multiplier=3.0).index("--timeout-multiplier") + 1
        ] == "3.0"

    def test_a_multiplier_of_one_is_omitted(self):
        assert "--timeout-multiplier" not in cmd(timeout_multiplier=1.0)

    def test_pinning_an_in_sandbox_agent_is_refused(self):
        # claude-code calls the model from inside the sandbox, where a localhost
        # proxy is unreachable. Running anyway would silently produce an arm
        # that hit the public endpoint and reported a host we never chose.
        with pytest.raises(ValueError, match="inside the sandbox"):
            cmd(agent="claude-code", proxied=True)

    def test_an_in_sandbox_agent_is_fine_when_not_pinning(self):
        assert "--agent" in cmd(agent="claude-code", proxied=False)

    def test_zero_attempts_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            cmd(attempts=0)


class TestProxyEnv:
    def test_both_spellings_and_a_placeholder_key(self):
        env = proxy_env("http://127.0.0.1:8900/v1")
        assert env["OPENAI_API_BASE"] == env["OPENAI_BASE_URL"] == "http://127.0.0.1:8900/v1"
        # The proxy holds the real credential; nothing real reaches the agent.
        assert env["OPENAI_API_KEY"] == "proxy"


def trial(task="t", reward=1, error=None, rewards_override=...):
    out = {
        "task_name": task,
        "trial_name": f"{task}.1",
        "agent_info": {"name": "terminus-2", "model_info": {"name": "openai/m"}},
        "verifier_result": {"rewards": {"resolved": reward}},
    }
    if rewards_override is not ...:
        out["verifier_result"] = rewards_override
    if error:
        out["exception_info"] = {"exception_type": error}
    return out


class TestTrialRows:
    def test_pass_and_fail_are_read_from_the_single_reward(self):
        rows = trial_rows({"trial_results": [trial(reward=1), trial(reward=0)]})
        assert [r["resolved"] for r in rows] == [True, False]

    def test_a_trial_that_never_verified_has_no_verdict(self):
        # Not a model failure: a missing measurement. Collapsing the two reports
        # infrastructure noise as a quality difference.
        rows = trial_rows(
            {"trial_results": [trial(error="TimeoutError", rewards_override=None)]}
        )
        assert rows[0]["resolved"] is None
        assert rows[0]["error"] == "TimeoutError"

    def test_multi_key_or_non_binary_rewards_are_not_guessed(self):
        rows = trial_rows(
            {
                "trial_results": [
                    trial(rewards_override={"rewards": {"a": 1, "b": 0}}),
                    trial(rewards_override={"rewards": {"partial": 0.5}}),
                ]
            }
        )
        assert [r["resolved"] for r in rows] == [None, None]

    def test_empty_job_yields_no_rows(self):
        assert trial_rows({}) == []


class TestSummarize:
    def test_rate_is_over_verdicts_not_all_trials(self):
        rows = trial_rows(
            {
                "trial_results": [
                    trial(reward=1),
                    trial(reward=0),
                    trial(error="ContainerError", rewards_override=None),
                ]
            }
        )
        summary = summarize(rows)
        assert summary["trials"] == 3
        assert summary["verdicts"] == 2
        assert summary["resolve_rate"] == 0.5      # not 1/3
        assert summary["unverified"] == 1
        assert summary["errors"] == {"ContainerError": 1}

    def test_no_verdicts_leaves_the_rate_undefined(self):
        rows = trial_rows(
            {"trial_results": [trial(error="TimeoutError", rewards_override=None)]}
        )
        assert summarize(rows)["resolve_rate"] is None


class TestLoadJobSummary:
    def test_reads_a_job_and_keeps_harbor_totals(self, tmp_path):
        job_dir = tmp_path / "job-1"
        job_dir.mkdir()
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [trial(reward=1), trial(reward=0)],
                    "stats": {
                        "n_input_tokens": 1000,
                        "n_cache_tokens": 800,
                        "n_output_tokens": 200,
                        "cost_usd": 0.05,
                    },
                }
            )
        )
        summary = load_job_summary(tmp_path, "job-1")
        assert summary["resolve_rate"] == 0.5
        # Carried as a cross-check against the ledger, which is authoritative.
        assert summary["harbor_stats"]["n_cache_tokens"] == 800
        assert summary["harbor_stats"]["cost_usd"] == 0.05

    def test_missing_job_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="no Harbor job result"):
            load_job_summary(tmp_path, "nope")

    def test_job_result_path_layout(self, tmp_path):
        assert job_result_path(tmp_path, "j").name == "result.json"
        assert job_result_path(tmp_path, "j").parent.name == "j"
