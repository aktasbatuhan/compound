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
    #: "openrouter", "doubleword", or any label for an OpenAI-compatible host
    #: (vLLM, Fireworks direct, Groq, a local server, ...). Non-openrouter
    #: providers speak the OpenAI chat API at ``api_base`` and authenticate with
    #: the key found in ``api_key_env``.
    provider: str
    model: str
    api_base: str | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    #: Env var holding the API key for a custom OpenAI-compatible host.
    api_key_env: str | None = None
    #: OpenRouter upstream tag (e.g. "baseten/fp8") to pin the serving host; the
    #: request then refuses fallbacks so every episode is served by that host.
    openrouter_provider: str | None = None
    #: Service tier flag forwarded in the request body (e.g. Doubleword "flex").
    #: CAUTION: Doubleword's chat route accepts but does not echo this — treat a
    #: tier-flagged run as unverified until reconciled against billing.
    service_tier: str | None = None

    def litellm_name(self) -> str:
        if self.provider == "openrouter":
            return f"openrouter/{self.model}"
        # Every other provider is an OpenAI-compatible host addressed by api_base.
        if self.provider != "doubleword" and not self.api_base:
            raise ValueError(
                f"provider {self.provider!r} is OpenAI-compatible and needs api_base"
            )
        return f"openai/{self.model}"

    def resolve_api_key_env(self) -> str | None:
        """Env var whose value must land in OPENAI_API_KEY before the run.

        OpenRouter is handled natively by litellm (OPENROUTER_API_KEY), so it
        needs no override; every OpenAI-compatible host authenticates through
        the OPENAI_API_KEY litellm sends to api_base.
        """
        if self.provider == "openrouter":
            return None
        if self.api_key_env:
            return self.api_key_env
        if self.provider == "doubleword":
            return "DOUBLEWORD_API_KEY"
        raise ValueError(f"provider {self.provider!r} needs api_key_env")

    def llm_args(self) -> dict:
        args: dict = {}
        if self.api_base:
            args["api_base"] = self.api_base
        if self.reasoning_effort:
            args["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            args["max_tokens"] = self.max_tokens
        extra_body: dict = {}
        if self.openrouter_provider:
            if self.provider != "openrouter":
                raise ValueError("openrouter_provider requires provider='openrouter'")
            # litellm forwards extra_body verbatim; OpenRouter honors provider.only
            # and, with allow_fallbacks off, errors rather than silently rerouting.
            extra_body["provider"] = {
                "only": [self.openrouter_provider],
                "allow_fallbacks": False,
            }
        if self.service_tier:
            extra_body["service_tier"] = self.service_tier
        if extra_body:
            args["extra_body"] = extra_body
        return args

    def slug(self) -> str:
        """Filesystem-safe identity for output naming, including the pinned host."""
        parts = [self.model.replace("/", "--")]
        if self.reasoning_effort:
            parts.append(f"reasoning-{self.reasoning_effort}")
        if self.openrouter_provider:
            parts.append(f"at-{self.openrouter_provider.replace('/', '-')}")
        if self.provider == "doubleword":
            parts.append(f"at-doubleword-{self.service_tier or 'realtime'}")
        elif self.provider != "openrouter":
            tier = f"-{self.service_tier}" if self.service_tier else ""
            parts.append(f"at-{self.provider}{tier}")
        return "--".join(parts)


def task_ids_by_domain(
    manifest_path: str | Path, partition: Partition | None
) -> dict[str, list[str]]:
    """Group manifest task ids by domain; `partition=None` selects every partition."""
    manifest = json.loads(Path(manifest_path).read_text())
    grouped: dict[str, list[str]] = defaultdict(list)
    for case in manifest["cases"]:
        if partition is not None and case["partition"] != partition.value:
            continue
        domain, task_id = case["case_id"].split(":", 1)
        grouped[domain].append(task_id)
    return dict(grouped)


NL_ASSERTIONS_JUDGE = "openrouter/openai/gpt-4.1"


def route_nl_judge_via_openrouter() -> None:
    """Point tau's nl_assertions judge at OpenRouter.

    tau grades nl_assertions tasks with an OpenAI-billed judge; the same model
    snapshot is served on OpenRouter, so rerouting it removes the extra
    credential (and survives the OPENAI_API_KEY override doubleword runs need).
    The evaluator imports the constant by value, so both bindings are patched.
    """
    import tau2.config
    import tau2.evaluator.evaluator_nl_assertions as nl_evaluator

    tau2.config.DEFAULT_LLM_NL_ASSERTIONS = NL_ASSERTIONS_JUDGE
    nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS = NL_ASSERTIONS_JUDGE


def run_tau_partition(
    *,
    manifest_path: str | Path,
    partition: Partition | None,
    agent_model: TauModel,
    user_model: TauModel,
    candidate_instruction: str,
    trials: int,
    max_steps: int,
    output_dir: str | Path,
    telemetry_path: str | Path | None = None,
    task_split_overrides: dict[str, str] | None = None,
    domains: set[str] | None = None,
    case_ids: set[str] | None = None,
) -> list[Path]:
    """Run selected tau tasks inside tau's own pinned environment.

    This function intentionally imports tau lazily so the Compound core stays isolated from
    benchmark dependencies.
    """
    key_env = agent_model.resolve_api_key_env()
    if key_env:
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(f"{key_env} is required")
        os.environ["OPENAI_API_KEY"] = key

    from tau2.agent.llm_agent import SYSTEM_PROMPT, LLMAgent
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.run import run_domain

    route_nl_judge_via_openrouter()

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

    # tau resolves a RELATIVE save_to under its own data directory; resolve so
    # results land here and the telemetry ingest below can actually find them.
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    partition_label = partition.value if partition is not None else "all"
    for domain, task_ids in task_ids_by_domain(manifest_path, partition).items():
        if domains is not None and domain not in domains:
            continue
        if case_ids is not None:
            task_ids = [t for t in task_ids if f"{domain}:{t}" in case_ids]
            if not task_ids:
                continue
        output = destination / f"{domain}-{partition_label}-{agent_model.slug()}.json"
        config = TextRunConfig(
            domain=domain,
            # Domains default to their "base" task split; the manifest may
            # reference ids that only exist in another split (telecom → "full"),
            # so allow a per-domain split override. (The *_full task-set aliases
            # are broken in this tau revision — the split is the working lever.)
            task_split_name=(task_split_overrides or {}).get(domain, "base"),
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
                agent_upstream=agent_model.openrouter_provider,
            )
        outputs.append(output)
    return outputs
