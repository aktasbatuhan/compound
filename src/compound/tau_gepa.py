"""GEPA prompt optimization over live tau-bench episodes.

Optimizes the ONE mutable component of the tau agent (its `agent_instruction`,
injected into tau's system prompt) for a fixed candidate model/route, with the
same anti-overfitting discipline as the DS-1000 arc:

  - GEPA explores on `optimizer_train` and Pareto-selects on
    `optimizer_validation`;
  - the sealed `decision_test` partition is NEVER touched during optimization;
  - the final claim comes from one baseline-vs-best run on the decision tasks
    (`--decision`), executed after the search is closed.

Every metric call is a full interactive tau episode (agent + user simulator +
live domain tools, official reward), run against the declared candidate route.
Money safety mirrors the sweep runner: `--estimate` is the free default, a hard
USD cap guards `--run`, and every episode's cost is ledgered at declared prices.

Run from the tau venv:
    PYTHONPATH=src .compound/sources/tau2-bench/.venv/bin/python -m compound.tau_gepa --estimate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gepa import EvaluationBatch

from compound.budget import BudgetLedger
from compound.costs import TokenPrices
from compound.gepa_v2 import ExperimentCap, MeteredV2ReflectionLM, NamespacedTrialLoader
from compound.providers import OpenAICompatibleProvider

# ---------------------------------------------------------------------------
# Declared decision context (docs.doubleword.ai/inference-api/models, async tier)
# ---------------------------------------------------------------------------
CANDIDATE_MODEL = "zai-org/GLM-5.2-FP8"
CANDIDATE_ROUTE = "doubleword/flex"
CANDIDATE_API_BASE = "https://api.doubleword.ai/v1"
CANDIDATE_SERVICE_TIER = "flex"
CANDIDATE_PRICES = TokenPrices(input_per_million=0.70, output_per_million=2.25)
USER_MODEL = "openai/gpt-5.6-luna"
USER_PRICES = TokenPrices(input_per_million=0.10, output_per_million=0.60)
REFLECTION_MODEL = "openai/gpt-5.6-sol"
REFLECTION_PRICES = TokenPrices(input_per_million=5.00, output_per_million=30.00)

MANIFEST_PATH = Path("benchmarks/manifests/tau_bench.json")
MAX_STEPS = 30
AGENT_MAX_TOKENS = 8192
# The Doubleword flex (async) tier can legitimately queue a request for minutes
# (observed tail ~200s), but with no client timeout a wedged request hangs the
# whole run forever (one episode stalled 4.6h before we killed it). Cap each
# request generously so a stuck call raises and tau2's retries can recover.
AGENT_TIMEOUT_S = 600
USER_TIMEOUT_S = 120
SEED_CANDIDATE = {"agent_instruction": ""}
REFLECTION_TEMPLATE = (
    "Rewrite this customer-service agent instruction using the evidence below. Keep only "
    "reusable behavioral principles (tool discipline, when to act vs ask, policy adherence, "
    "closing the loop with the user); never include case-specific names, values, or solutions. "
    "Replace weak rules instead of appending. The replacement must be under 200 words.\n\n"
    "CURRENT COMPONENT:\n<curr_param>\n\nEVIDENCE:\n<side_info>\n\n"
    "Return only the complete replacement instruction in a fenced text block."
)

# The registered tau agent factory reads the CURRENT candidate here, so one
# registration serves every GEPA candidate.
_CURRENT_INSTRUCTION: dict[str, str] = {"text": ""}
_AGENT_NAME = "compound_gepa_agent"


@dataclass(frozen=True, slots=True)
class TauTask:
    domain: str
    task_id: str

    @property
    def case_id(self) -> str:  # NamespacedTrialLoader keys off this
        return f"{self.domain}:{self.task_id}"


@dataclass(slots=True)
class TauEpisodeTrace:
    task: TauTask
    reward: float
    termination: str
    feedback: str
    first_user_message: str


def load_partition_tasks(partition: str) -> list[TauTask]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    tasks = []
    for case in manifest["cases"]:
        if case["partition"] != partition:
            continue
        domain, task_id = case["case_id"].split(":", 1)
        tasks.append(TauTask(domain, task_id))
    return tasks


def _register_agent() -> None:
    from tau2.agent.llm_agent import SYSTEM_PROMPT, LLMAgent
    from tau2.registry import registry

    class CompoundGepaAgent(LLMAgent):
        @property
        def system_prompt(self) -> str:
            return SYSTEM_PROMPT.format(
                domain_policy=self.domain_policy,
                agent_instruction=_CURRENT_INSTRUCTION["text"],
            )

    def create_agent(tools, domain_policy, **kwargs):
        return CompoundGepaAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    if _AGENT_NAME not in registry.get_agents():
        registry.register_agent_factory(create_agent, _AGENT_NAME)


def _candidate_hash(instruction: str) -> str:
    return hashlib.sha256(instruction.encode()).hexdigest()[:12]


def _group_task_ids(tasks: list[TauTask]) -> dict[str, list[str]]:
    """Group task ids by domain, deduped, order preserved.

    GEPA's reflective minibatches can sample the same task twice; tau2's
    get_tasks rejects a task_ids list longer than the unique tasks it loads
    (len(tasks) != len(task_ids), with an empty "missing" set), which crashed a
    run mid-reflection. Dedup here; evaluate() maps every batch item back by
    (domain, task_id), so duplicates still resolve to the single run.
    """
    by_domain: dict[str, list[str]] = {}
    for task in tasks:
        ids = by_domain.setdefault(task.domain, [])
        if task.task_id not in ids:
            ids.append(task.task_id)
    return by_domain


class EpisodeRunner:
    """Runs tau episodes for (candidate, tasks); caches per candidate on disk.

    tau's auto_resume keys on (trial, task_id, seed) inside a save dir, so one
    directory per candidate hash makes re-evaluation of a known candidate free.
    Episode costs are ledgered once per simulation id at declared prices.
    """

    def __init__(self, run_root: Path, ledger: BudgetLedger, cap: ExperimentCap) -> None:
        self.run_root = run_root
        self.ledger = ledger
        self.cap = cap
        self._charged_path = run_root / "charged-episodes.json"
        self._charged: set[str] = (
            set(json.loads(self._charged_path.read_text()))
            if self._charged_path.exists()
            else set()
        )
        self.metric_calls = 0

    def _charge(self, sim_key: str, sims: list[dict]) -> None:
        if sim_key in self._charged:
            return
        cost = 0.0
        for sim in sims:
            for message in sim.get("messages", []):
                usage = message.get("usage") or {}
                if not usage:
                    continue
                prices = CANDIDATE_PRICES if message.get("role") == "assistant" else USER_PRICES
                cost += (
                    (usage.get("prompt_tokens", 0) or 0) * prices.input_per_million
                    + (usage.get("completion_tokens", 0) or 0) * prices.output_per_million
                ) / 1e6
        self.ledger.record(cost, label=f"tau-episodes:{sim_key}")
        self._charged.add(sim_key)
        self._charged_path.parent.mkdir(parents=True, exist_ok=True)
        self._charged_path.write_text(json.dumps(sorted(self._charged)))

    def run(
        self, instruction: str, tasks: list[TauTask], *, trials: int = 1, tag: str | None = None
    ) -> dict[tuple[str, str], list[dict]]:
        from tau2.data_model.simulation import TextRunConfig
        from tau2.run import run_domain

        _register_agent()
        _CURRENT_INSTRUCTION["text"] = instruction
        chash = tag or _candidate_hash(instruction)
        by_domain = _group_task_ids(tasks)

        results: dict[tuple[str, str], list[dict]] = {}
        for domain, ids in sorted(by_domain.items()):
            self.cap.require_headroom(0.10 * len(ids) * trials)
            # tau's resume refuses a save dir whose task set differs, and GEPA
            # evaluates one candidate on many minibatches - so the task-id set
            # joins the path (and the charge key, or reruns would be uncharged).
            idhash = hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:8]
            save_to = self.run_root / "episodes" / chash / f"{domain}-{idhash}.json"
            save_to.parent.mkdir(parents=True, exist_ok=True)
            config = TextRunConfig(
                domain=domain,
                task_split_name="full" if domain == "telecom" else "base",
                task_ids=ids,
                num_trials=trials,
                max_steps=MAX_STEPS,
                agent=_AGENT_NAME,
                llm_agent=f"openai/{CANDIDATE_MODEL}",
                llm_args_agent={
                    "api_base": CANDIDATE_API_BASE,
                    "max_tokens": AGENT_MAX_TOKENS,
                    "timeout": AGENT_TIMEOUT_S,
                    "extra_body": {"service_tier": CANDIDATE_SERVICE_TIER},
                },
                llm_user=f"openrouter/{USER_MODEL}",
                llm_args_user={"max_tokens": 2048, "timeout": USER_TIMEOUT_S},
                save_to=str(save_to.resolve()),
                auto_resume=True,
            )
            run_domain(config)
            payload = json.loads((save_to / "results.json").read_text())
            sims = payload.get("simulations", [])
            self._charge(f"{chash}:{domain}:{idhash}", sims)
            for sim in sims:
                key = (domain, str(sim.get("task_id")))
                results.setdefault(key, []).append(sim)
        self.metric_calls += len(tasks) * trials
        return results


def _sim_feedback(sim: dict) -> tuple[float, str, str, str]:
    info = sim.get("reward_info") or {}
    reward = float(info.get("reward") or 0.0)
    termination = str(sim.get("termination_reason") or "unknown")
    parts = [f"reward={reward:.2f}", f"termination={termination}"]
    note = (info.get("info") or {}).get("note")
    if note:
        parts.append(f"note={note}")
    for field in ("action_checks", "communicate_checks", "nl_assertions", "db_check"):
        value = info.get(field)
        if value:
            parts.append(f"{field}={json.dumps(value)[:400]}")
    first_user = ""
    tail = []
    for message in sim.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if role == "user" and not first_user and isinstance(content, str):
            first_user = content[:400]
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            tail.append(f"{role}: {content[:280]}")
    if tail:
        parts.append("final exchanges:\n" + "\n".join(tail[-4:]))
    return reward, termination, "\n".join(parts), first_user


class TauGEPAAdapter:
    # gepa probes this optional hook; None selects its reflection-LM proposal path.
    propose_new_texts = None

    def __init__(self, runner: EpisodeRunner) -> None:
        self.runner = runner

    def evaluate(
        self,
        batch: list[TauTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TauEpisodeTrace, dict[str, Any]]:
        instruction = candidate.get("agent_instruction", "")
        results = self.runner.run(instruction, list(batch))
        traces: list[TauEpisodeTrace] = []
        for task in batch:
            sims = results.get((task.domain, task.task_id), [])
            if not sims:
                traces.append(TauEpisodeTrace(task, 0.0, "missing", "episode did not run", ""))
                continue
            reward, termination, feedback, first_user = _sim_feedback(sims[0])
            traces.append(TauEpisodeTrace(task, reward, termination, feedback, first_user))
        return EvaluationBatch(
            outputs=[
                {"case_id": t.task.case_id, "reward": t.reward, "termination": t.termination}
                for t in traces
            ],
            scores=[t.reward for t in traces],
            trajectories=traces if capture_traces else None,
            objective_scores=[{"task_success": t.reward} for t in traces],
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TauEpisodeTrace, dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        del candidate
        traces = eval_batch.trajectories
        if traces is None:
            raise ValueError("captured trajectories are required for reflection")
        records = [
            {
                "case_id": t.task.case_id,
                "Inputs": {"domain": t.task.domain, "customer_opening": t.first_user_message},
                "Generated Outputs": f"episode ended: {t.termination}",
                "Feedback": t.feedback,
            }
            for t in traces
        ]
        return {component: records for component in components_to_update}


def _rate(results: dict, tasks: list[TauTask]) -> dict[str, Any]:
    per_domain: dict[str, list[float]] = {}
    rewards = []
    for task in tasks:
        sims = results.get((task.domain, task.task_id), [])
        task_rewards = [float((s.get("reward_info") or {}).get("reward") or 0.0) for s in sims]
        mean = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
        rewards.append(mean)
        per_domain.setdefault(task.domain, []).append(mean)
    return {
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "solved": sum(1 for r in rewards if r == 1.0),
        "n": len(rewards),
        "per_domain": {
            d: f"{sum(1 for r in v if r == 1.0)}/{len(v)}" for d, v in sorted(per_domain.items())
        },
    }


def run_optimization(output: Path, max_metric_calls: int, cap_usd: float, seed: int) -> int:
    import gepa

    train = load_partition_tasks("optimizer_train")
    val = load_partition_tasks("optimizer_validation")
    ledger = BudgetLedger.load(output / "budget.json", cap_usd)
    cap = ExperimentCap(ledger, ledger.spent_usd, cap_usd)
    runner = EpisodeRunner(output, ledger, cap)
    adapter = TauGEPAAdapter(runner)
    reflection = MeteredV2ReflectionLM(
        provider=OpenAICompatibleProvider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        ),
        model=REFLECTION_MODEL,
        prices=REFLECTION_PRICES,
        ledger=ledger,
        experiment_cap=cap,
        cache_dir=output / "cache" / "reflection",
    )
    minibatch = 3
    engine_limit = max(len(val), max_metric_calls - (len(val) + 2 * minibatch) + 1)
    manifest = {
        "algorithm": "gepa-tau",
        "candidate_model": CANDIDATE_MODEL,
        "candidate_route": CANDIDATE_ROUTE,
        "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": max_metric_calls,
        "engine_metric_call_limit": engine_limit,
        "seed_candidate": SEED_CANDIDATE,
        "train_case_ids": [t.case_id for t in train],
        "validation_case_ids": [t.case_id for t in val],
        "decision_case_ids_exposed": [],
        "started_at": datetime.now(UTC).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=NamespacedTrialLoader(train, "train"),
        valset=NamespacedTrialLoader(val, "validation"),
        adapter=adapter,
        reflection_lm=reflection,
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=minibatch,
        reflection_prompt_template={"agent_instruction": REFLECTION_TEMPLATE},
        perfect_score=1.0,
        max_metric_calls=engine_limit,
        run_dir=str(output / "gepa"),
        cache_evaluation=False,
        seed=seed,
        display_progress_bar=True,
    )
    best = dict(result.best_candidate)
    baseline_val = _rate(runner.run(SEED_CANDIDATE["agent_instruction"], val, tag="seed-val"), val)
    best_val = _rate(runner.run(best["agent_instruction"], val, tag="best-val"), val)
    summary = {
        "best_candidate": best,
        "baseline_validation": baseline_val,
        "best_validation": best_val,
        "metric_calls_episodes": runner.metric_calls,
        "spent_usd": round(ledger.spent_usd, 4),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nnext: --decision {output} runs baseline vs best ONCE on the sealed decision tasks")
    return 0


def run_decision(run_dir: Path, trials: int, cap_usd: float) -> int:
    summary = json.loads((run_dir / "summary.json").read_text())
    best = summary["best_candidate"]["agent_instruction"]
    decision = load_partition_tasks("decision_test")
    gate_path = run_dir / "decision.json"
    if gate_path.exists():
        print(f"error: decision already ran once ({gate_path}); refusing a second look",
              file=sys.stderr)
        return 2
    ledger = BudgetLedger.load(run_dir / "budget.json", cap_usd)
    cap = ExperimentCap(ledger, ledger.spent_usd, cap_usd)
    runner = EpisodeRunner(run_dir, ledger, cap)
    baseline = _rate(
        runner.run("", decision, trials=trials, tag="decision-baseline"), decision
    )
    optimized = _rate(
        runner.run(best, decision, trials=trials, tag="decision-best"), decision
    )
    report = {
        "decision_tasks": [t.case_id for t in decision],
        "trials": trials,
        "baseline": baseline,
        "optimized": optimized,
        "delta_mean_reward": round(optimized["mean_reward"] - baseline["mean_reward"], 4),
        "spent_usd_total": round(ledger.spent_usd, 4),
        "decided_at": datetime.now(UTC).isoformat(),
    }
    gate_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/optimization") /
                        f"tau-gepa-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--max-metric-calls", type=int, default=60)
    parser.add_argument("--cap-usd", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decision-trials", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="print cost plan (default)")
    mode.add_argument("--run", action="store_true", help="run the optimization (spends money)")
    mode.add_argument("--decision", type=Path, metavar="RUN_DIR",
                      help="one-shot baseline-vs-best on the sealed decision tasks")
    args = parser.parse_args()

    if args.run or args.decision:
        for env in ("DOUBLEWORD_API_KEY", "OPENROUTER_API_KEY"):
            if not os.getenv(env):
                print(f"error: {env} is required", file=sys.stderr)
                return 2
        os.environ["OPENAI_API_KEY"] = os.environ["DOUBLEWORD_API_KEY"]

    if args.decision:
        return run_decision(args.decision, args.decision_trials, args.cap_usd)
    if args.run:
        return run_optimization(args.output, args.max_metric_calls, args.cap_usd, args.seed)

    train, val, decision = (load_partition_tasks(p) for p in
                            ("optimizer_train", "optimizer_validation", "decision_test"))
    ep_usd = 0.056  # ~24k in / 0.5k out per episode at declared flex + user-sim prices
    search = args.max_metric_calls * ep_usd
    final = 2 * len(decision) * args.decision_trials * ep_usd
    print(f"target: {CANDIDATE_MODEL} @ {CANDIDATE_ROUTE} (reflection: {REFLECTION_MODEL})")
    print(f"partitions: train={len(train)} val={len(val)} decision={len(decision)} (sealed)")
    print(f"search: <= {args.max_metric_calls} episodes  ~${search:.2f}"
          f" + reflection ~$0.40")
    print(f"decision gate: 2 candidates x {len(decision)} tasks x {args.decision_trials} trials"
          f" = {2 * len(decision) * args.decision_trials} episodes  ~${final:.2f}")
    print(f"ceiling: ~${search + final + 0.4:.2f}   hard cap: ${args.cap_usd:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
