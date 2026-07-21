from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from compound.contracts import Partition
from compound.telemetry import ingest_tau_results


@dataclass(frozen=True, slots=True)
class TauModel:
    provider: str
    model: str
    api_base: str | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = None

    def litellm_name(self) -> str:
        if self.provider == "openrouter":
            return f"openrouter/{self.model}"
        if self.provider == "doubleword":
            return f"openai/{self.model}"
        raise ValueError(f"unsupported tau provider: {self.provider}")

    def llm_args(self) -> dict:
        args: dict = {}
        if self.api_base:
            args["api_base"] = self.api_base
        if self.reasoning_effort:
            args["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            args["max_tokens"] = self.max_tokens
        return args


def task_ids_by_domain(
    manifest_path: str | Path, partition: Partition
) -> dict[str, list[str]]:
    manifest = json.loads(Path(manifest_path).read_text())
    grouped: dict[str, list[str]] = defaultdict(list)
    for case in manifest["cases"]:
        if case["partition"] != partition.value:
            continue
        domain, task_id = case["case_id"].split(":", 1)
        grouped[domain].append(task_id)
    return dict(grouped)


def run_tau_partition(
    *,
    manifest_path: str | Path,
    partition: Partition,
    agent_model: TauModel,
    user_model: TauModel,
    candidate_instruction: str,
    trials: int,
    max_steps: int,
    output_dir: str | Path,
    telemetry_path: str | Path | None = None,
) -> list[Path]:
    """Run selected tau tasks inside tau's own pinned environment.

    This function intentionally imports tau lazily so the Compound core stays isolated from
    benchmark dependencies.
    """
    if agent_model.provider == "doubleword":
        key = os.getenv("DOUBLEWORD_API_KEY")
        if not key:
            raise RuntimeError("DOUBLEWORD_API_KEY is required")
        os.environ["OPENAI_API_KEY"] = key

    from tau2.agent.llm_agent import SYSTEM_PROMPT, LLMAgent
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.run import run_domain

    class CompoundAgent(LLMAgent):
        @property
        def system_prompt(self) -> str:
            return SYSTEM_PROMPT.format(
                domain_policy=self.domain_policy,
                agent_instruction=candidate_instruction,
            )

    def create_agent(tools, domain_policy, **kwargs):
        return CompoundAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    agent_name = "compound_prompt_agent"
    if agent_name not in registry.get_agents():
        registry.register_agent_factory(create_agent, agent_name)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for domain, task_ids in task_ids_by_domain(manifest_path, partition).items():
        model_slug = agent_model.model.replace("/", "--")
        if agent_model.reasoning_effort:
            model_slug += f"--reasoning-{agent_model.reasoning_effort}"
        output = destination / f"{domain}-{partition.value}-{model_slug}.json"
        config = TextRunConfig(
            domain=domain,
            task_ids=task_ids,
            num_trials=trials,
            max_steps=max_steps,
            agent=agent_name,
            llm_agent=agent_model.litellm_name(),
            llm_args_agent=agent_model.llm_args(),
            llm_user=user_model.litellm_name(),
            llm_args_user=user_model.llm_args(),
            save_to=str(output),
            auto_resume=True,
        )
        run_domain(config)
        results_path = output / "results.json"
        if telemetry_path and results_path.exists():
            ingest_tau_results(
                results_path,
                telemetry_path=telemetry_path,
                agent_provider=agent_model.provider,
                agent_model=agent_model.model,
                user_provider=user_model.provider,
                user_model=user_model.model,
            )
        outputs.append(output)
    return outputs
