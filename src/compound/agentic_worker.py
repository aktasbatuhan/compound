"""One isolated pilot episode, using upstream benchmark agents and graders."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from compound.agentic_gateway import ProviderResponseError, check_response_errors


class LocalClient:
    def __init__(self, base, model):
        self.base, self.model = base, model

    def complete_messages(self, messages, **kwargs):
        kwargs.pop("timeout", None)
        body = {"model": self.model, "messages": messages, **kwargs}
        request = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=650) as response:
            raw = json.load(response)
        check_response_errors(raw)
        return raw["choices"][0]["message"].get("content") or "", raw


def retail(spec, episode, base, directory):
    import litellm
    import tau2.utils.llm_utils as tau_llm
    from tau2.data_model.simulation import RunConfig
    from tau2.run import run_domain

    original = litellm.completion

    def local_only(*args, **kwargs):
        # Judges with default models are also forced through the auxiliary route.
        kwargs["api_base"] = kwargs.get("api_base") or base.replace("/agent/", "/auxiliary/")
        if not kwargs["api_base"].startswith("http://127.0.0.1:"):
            raise ValueError("retail model call escaped the metered gateway")
        kwargs.update(api_key="local-meter", num_retries=0, timeout=650)
        response = original(*args, **kwargs)
        role = "auxiliary" if "/auxiliary/" in kwargs["api_base"] else "agent"
        check_response_errors(response.model_dump(), role)
        return response

    litellm.completion = local_only
    tau_llm.completion = local_only
    config = RunConfig(
        domain="retail",
        task_ids=[episode["task_id"]],
        task_split_name=None,
        agent="llm_agent",
        llm_agent="openai/gpt-5.6-sol",
        llm_args_agent={"api_base": base, "api_key": "local-meter", "max_tokens": 8192},
        llm_user="openai/gpt-5.6-sol",
        llm_args_user={
            "api_base": base.replace("/agent/", "/auxiliary/"),
            "api_key": "local-meter",
            "max_tokens": 2048,
        },
        num_trials=1,
        max_steps=spec["sources"]["retail"]["max_steps"],
        max_concurrency=1,
        seed=episode["trial"],
        save_to=str(directory / "official"),
        log_level="ERROR",
    )
    result = run_domain(config)
    if len(result.simulations) != 1 or result.simulations[0].reward_info is None:
        raise ValueError("missing retail grade")
    sim = result.simulations[0]
    return {
        "status": "graded",
        "success": sim.reward_info.reward == 1.0,
        "duration_s": sim.duration,
        "metrics": sim.reward_info.model_dump(mode="json"),
        "grader": "tau2-verified@" + spec["sources"]["retail"]["revision"],
    }


def finance(spec, episode, base, directory):
    from finbalance.benchmark.dataset import load_records

    from compound.adapters.finbalance import run_case

    record = next(
        r
        for r in load_records(spec["sources"]["finance"]["data_path"])
        if r.record_id == episode["task_id"]
    )
    result = run_case(
        record,
        LocalClient(base, "metered-model"),
        max_steps=spec["sources"]["finance"]["max_steps"],
    )
    (directory / "official.json").write_text(json.dumps(result, indent=2) + "\n")
    return {k: result[k] for k in ("status", "success", "metrics", "duration_s")} | {
        "grader": "finbalance@" + spec["sources"]["finance"]["revision"]
    }


def coding(spec, episode, base, directory, oracle=False):
    from swebench.harness.constants import RUN_EVALUATION_LOG_DIR

    setup_start = time.monotonic()
    instances = json.loads(Path(spec["sources"]["coding"]["data_path"]).read_text())
    instance = next(x for x in instances if x["instance_id"] == episode["task_id"])
    grader_path = Path(".compound/sources/swe-verified-grader.json")
    grader_rows = json.loads(grader_path.read_text())
    grading = next(x for x in grader_rows if x["instance_id"] == instance["instance_id"])
    for key in ("problem_statement", "patch", "test_patch", "base_commit"):
        if grading[key] != instance[key]:
            raise ValueError("grading metadata changed frozen task: " + key)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        a, b = grading[key], instance[key]
        a = json.loads(a) if isinstance(a, str) else a
        b = json.loads(b) if isinstance(b, str) else b
        if set(a) != set(b):
            raise ValueError("grading metadata changed frozen tests: " + key)
    instance["image_name"] = grading["image"]
    if oracle:
        patch = instance["patch"]
        duration = 0
    else:
        import litellm
        import yaml
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.models.litellm_model import LitellmModel
        from minisweagent.run.benchmarks.swebench import get_sb_environment

        original_completion = litellm.completion

        def checked_completion(*args, **kwargs):
            response = original_completion(*args, **kwargs)
            check_response_errors(response.model_dump())
            return response

        litellm.completion = checked_completion
        cfg = yaml.safe_load(
            Path(
                ".compound/sources/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml"
            ).read_text()
        )
        cfg["agent"].update(step_limit=spec["sources"]["coding"]["max_steps"], cost_limit=0)
        model = LitellmModel(
            model_name="openai/gpt-5.6-sol",
            cost_tracking="ignore_errors",
            model_kwargs={
                "api_base": base,
                "api_key": "local-meter",
                "max_tokens": 8192,
                "timeout": 650,
                "num_retries": 0,
            },
        )
        env = get_sb_environment(cfg, instance)
        agent = DefaultAgent(model, env, **cfg["agent"])
        agent_start = time.monotonic()
        try:
            info = agent.run(instance["problem_statement"])
            duration = time.monotonic() - agent_start
            patch = info.get("submission") or ""
            agent.save(directory / "trajectory.json")
        finally:
            if hasattr(env, "cleanup"):
                env.cleanup()
    prediction = {
        "instance_id": instance["instance_id"],
        "model_name_or_path": "compound-flex",
        "model_patch": patch,
    }
    setup_duration = time.monotonic() - setup_start - duration
    grading_start = time.monotonic()
    (directory / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(grader_path),
        "--predictions_path",
        str(directory / "predictions.jsonl"),
        "--instance_ids",
        instance["instance_id"],
        "--max_workers",
        "1",
        "--run_id",
        episode["episode_id"],
        "--timeout",
        "300",
    ]
    subprocess.run(cmd, check=True, timeout=1200)
    report = (
        RUN_EVALUATION_LOG_DIR
        / episode["episode_id"]
        / "compound-flex"
        / instance["instance_id"]
        / "report.json"
    )
    if not report.is_file():
        aggregate_path = RUN_EVALUATION_LOG_DIR / episode["episode_id"] / "results.json"
        if aggregate_path.is_file():
            aggregate = json.loads(aggregate_path.read_text())
            if instance["instance_id"] in aggregate.get("empty_patch_ids", []):
                (directory / "official.json").write_text(json.dumps(aggregate, indent=2) + "\n")
                return {
                    "status": "graded",
                    "success": False,
                    "failure_reason": "empty_patch",
                    "duration_s": duration,
                    "setup_duration_s": setup_duration,
                    "grading_duration_s": time.monotonic() - grading_start,
                    "grader": "swebench@" + spec["sources"]["coding"]["grader_revision"],
                }
        raise ValueError("SWE-bench grader did not produce a report")
    raw = json.loads(report.read_text())
    (directory / "official.json").write_text(json.dumps(raw, indent=2) + "\n")
    if raw[instance["instance_id"]].get("infra_failure"):
        raise ValueError("SWE-bench reported a grading infrastructure failure")
    return {
        "status": "graded",
        "success": raw[instance["instance_id"]]["resolved"] is True,
        "duration_s": duration,
        "setup_duration_s": setup_duration,
        "grading_duration_s": time.monotonic() - grading_start,
        "grader": "swebench@" + spec["sources"]["coding"]["grader_revision"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--oracle", action="store_true")
    args = p.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    episode = json.loads(args.episode)
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        if args.oracle and episode["suite"] != "coding":
            raise ValueError("oracle mode is only for local coding grader validation")
        if episode["suite"] == "coding":
            result = coding(spec, episode, args.base, directory, args.oracle)
        elif episode["suite"] == "retail":
            result = retail(spec, episode, args.base, directory)
        else:
            result = finance(spec, episode, args.base, directory)
    except ProviderResponseError as exc:
        result = {
            "status": "provider_error" if exc.role == "agent" else "infrastructure_error",
            "success": None,
            "error_type": type(exc).__name__,
            "provider_error_code": exc.code,
            "failure_role": exc.role,
            "embedded_provider_error": True,
        }
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result = {
            "status": "infrastructure_error",
            "success": None,
            "error_type": type(exc).__name__,
        }
        if (
            episode["suite"] == "retail"
            and isinstance(exc, ValueError)
            and str(exc).startswith("AssistantMessage must have either content or tool calls.")
        ):
            result.update(
                status="provider_error",
                failure_role="agent",
                failure_reason="empty_agent_message",
            )
    result.setdefault("duration_s", time.monotonic() - start)
    result["worker_duration_s"] = time.monotonic() - start
    (directory / "outcome.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
