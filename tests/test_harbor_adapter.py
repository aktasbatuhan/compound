"""Tests for the Harbor / Terminal-Bench 4.0 adapter.

The command builder is pure so the argv a paid run would issue is asserted
here rather than discovered by spending. The result parsing is tested against
Harbor's real schema shape, with the emphasis on keeping a trial that never
reached the verifier out of the pass rate.
"""

from __future__ import annotations

import json
import os

import pytest

from compound.adapters.harbor import (
    DEFAULT_AGENT,
    DEFAULT_DATASET,
    build_command,
    job_result_path,
    load_job_summary,
    load_trial_results,
    proxy_env,
    qualify_task,
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
        # Bare names are widened to match Harbor's namespaced task ids.
        assert "*/cad-model" in got and "*/coq-block-bound" in got
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


class TestQualifyTask:
    def test_bare_name_is_widened_to_match_harbors_namespaced_ids(self):
        # Harbor task ids are 'terminal-bench/data-anonymization'; a bare name
        # matches nothing and the job dies with "No tasks matched the filter(s)".
        got = cmd(include_tasks=["data-anonymization"])
        assert "*/data-anonymization" in got

    def test_qualify_task_directly(self):
        assert qualify_task("cad-model") == "*/cad-model"
        assert qualify_task("terminal-bench/cad-model") == "terminal-bench/cad-model"
        assert qualify_task("*-model") == "*-model"

    def test_already_qualified_or_globbed_names_pass_through(self):
        got = cmd(include_tasks=["terminal-bench/cad-model", "*-model"])
        assert "terminal-bench/cad-model" in got
        assert "*-model" in got


class TestSweepHarborWiring:
    """Where each signal has to be set for a pinned run to record anything."""

    def _run(self, tmp_path, monkeypatch, ledger_dir=None):
        import contextlib

        from compound import provider_sweep
        from compound.adapters import harbor as harbor_mod
        from compound.providers_registry import ProviderSpec

        seen: dict = {}

        @contextlib.contextmanager
        def fake_serve(spec, port=0):
            # The proxy reads its signals from THIS process while it serves.
            seen["ledger_env"] = os.environ.get("COMPOUND_CALL_LEDGER")
            seen["label_env"] = os.environ.get("COMPOUND_RUN_LABEL")
            yield "http://127.0.0.1:9999/v1"

        def fake_run(command, extra_env=None, cwd=None):
            seen["extra_env"] = extra_env
            job_dir = tmp_path / "jobs" / command[command.index("--job-name") + 1]
            job_dir.mkdir(parents=True)
            (job_dir / "result.json").write_text(
                json.dumps({"trial_results": [trial(reward=1)], "stats": {}})
            )
            return 0

        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setattr(provider_sweep, "serve_provider", fake_serve)
        monkeypatch.setattr(harbor_mod, "run_harbor", fake_run)
        spec = ProviderSpec(
            token="openrouter/auto", kind="openrouter",
            base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
            upstream=None,
        )
        summaries = provider_sweep.sweep_harbor(
            [spec], model="m", jobs_dir=tmp_path / "jobs",
            dataset="d@1", agent="terminus-2", ledger_dir=ledger_dir,
        )
        return seen, summaries

    def test_ledger_reaches_the_proxy_not_the_subprocess(self, tmp_path, monkeypatch):
        # Regression: the proxy runs in-process, so a ledger path passed only to
        # the subprocess env records nothing and the run yields no call data.
        seen, _ = self._run(tmp_path, monkeypatch, ledger_dir=tmp_path / "led")
        assert seen["ledger_env"] == str(tmp_path / "led" / "openrouter-auto.jsonl")
        assert seen["label_env"] == "openrouter-auto"
        assert "COMPOUND_CALL_LEDGER" not in seen["extra_env"]

    def test_only_the_endpoint_and_key_go_to_the_agent(self, tmp_path, monkeypatch):
        seen, _ = self._run(tmp_path, monkeypatch, ledger_dir=tmp_path / "led")
        assert set(seen["extra_env"]) == {
            "OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_KEY"
        }

    def test_process_env_is_restored_after_the_arm(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMPOUND_CALL_LEDGER", raising=False)
        self._run(tmp_path, monkeypatch, ledger_dir=tmp_path / "led")
        # A sweep must not leak one arm's ledger path into the next.
        assert "COMPOUND_CALL_LEDGER" not in os.environ

    def test_summary_is_returned_per_host(self, tmp_path, monkeypatch):
        _, summaries = self._run(tmp_path, monkeypatch)
        assert summaries["openrouter-auto"]["resolve_rate"] == 1.0


class TestAgentTimeoutMultiplier:
    def test_agent_only_cap_is_a_separate_flag(self):
        # --timeout-multiplier scales every phase including the environment
        # build, so shrinking it to bound a run kills the container before it
        # starts (observed live: EnvironmentStartTimeoutError).
        got = cmd(agent_timeout_multiplier=0.02)
        assert got[got.index("--agent-timeout-multiplier") + 1] == "0.02"
        assert "--timeout-multiplier" not in got

    def test_both_can_be_set_independently(self):
        got = cmd(timeout_multiplier=3.0, agent_timeout_multiplier=0.5)
        assert "--timeout-multiplier" in got and "--agent-timeout-multiplier" in got

    def test_a_multiplier_of_one_is_omitted(self):
        assert "--agent-timeout-multiplier" not in cmd(agent_timeout_multiplier=1.0)


class TestErroredTrialsAreVisible:
    def test_an_arm_where_every_trial_errored_is_not_reported_as_empty(self, tmp_path):
        # Harbor counts a trial that died before producing a result, but writes
        # no entry in trial_results. Reading only the rows made a fully failed
        # arm look like an empty successful one.
        job_dir = tmp_path / "job-1"
        job_dir.mkdir()
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [],
                    "n_total_trials": 1,
                    "stats": {"n_errored_trials": 1},
                }
            )
        )
        summary = load_job_summary(tmp_path, "job-1")
        assert summary["trials"] == 0
        assert summary["errored_trials"] == 1
        assert summary["total_trials"] == 1


class TestPerTrialResults:
    def _job(self, tmp_path, job_trials):
        job = tmp_path / "job-1"
        (job / "task-a").mkdir(parents=True)
        (job / "task-b").mkdir(parents=True)
        (job / "result.json").write_text(
            json.dumps({"trial_results": job_trials, "n_total_trials": 2,
                        "stats": {"n_errored_trials": 1}})
        )
        (job / "task-a" / "result.json").write_text(json.dumps(trial(task="a", reward=1)))
        (job / "task-b" / "result.json").write_text(json.dumps(trial(task="b", reward=0)))
        return tmp_path

    def test_trial_files_are_used_when_the_job_level_list_is_empty(self, tmp_path):
        # Observed live: a finished 5-trial job carried trial_results: [] while
        # every trial's outcome sat in its own directory, so the arm reported
        # as having run nothing at all.
        summary = load_job_summary(self._job(tmp_path, []), "job-1")
        assert summary["trials"] == 2
        assert summary["resolve_rate"] == 0.5

    def test_the_job_level_list_wins_when_present(self, tmp_path):
        summary = load_job_summary(self._job(tmp_path, [trial(task="z", reward=1)]), "job-1")
        assert summary["trials"] == 1

    def test_load_trial_results_reads_each_trial_directory(self, tmp_path):
        results = load_trial_results(self._job(tmp_path, []), "job-1")
        assert sorted(r["task_name"] for r in results) == ["a", "b"]


class TestAgentKwargs:
    def test_max_turns_reaches_the_agent_constructor(self):
        # Equal turns, not equal wall clock: a clock cap hands the faster host
        # more turns and records a slow host's truncation as a failure.
        got = cmd(agent_kwargs={"max_turns": "30"})
        assert got[got.index("--agent-kwarg") + 1] == "max_turns=30"

    def test_multiple_kwargs_repeat_the_flag(self):
        got = cmd(agent_kwargs={"max_turns": "30", "max_thinking_tokens": "2048"})
        assert got.count("--agent-kwarg") == 2

    def test_no_kwargs_adds_nothing(self):
        assert "--agent-kwarg" not in cmd()


def test_build_command_runs_a_local_task_directory():
    """A benchmark whose own runner is unreleased still runs from a checkout."""
    command = build_command(
        task_path="repo/tasks/crash-proof-flash-filesystem",
        dataset=None,
        model="openai/z-ai/glm-5.3-flash",
        jobs_dir="jobs",
        job_name="arm",
    )
    assert "--path" in command
    assert command[command.index("--path") + 1] == "repo/tasks/crash-proof-flash-filesystem"
    assert "--dataset" not in command


def test_build_command_refuses_both_sources():
    with pytest.raises(ValueError, match="not both"):
        build_command(
            task_path="repo/tasks/x",
            dataset="some/other@1.0",
            model="m",
            jobs_dir="jobs",
            job_name="arm",
        )


def test_build_command_refuses_no_source():
    with pytest.raises(ValueError, match="one of dataset or task_path"):
        build_command(dataset=None, model="m", jobs_dir="jobs", job_name="arm")
