import json
import random
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_environment
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from compound.agentic_study import fingerprint

base = Path("benchmarks/flex-agentic/budget-curve-v2.json")
spec = json.loads(base.read_text())
source = spec["sources"]["retail"]
old = {t["id"] for t in source["tasks"]}
rows = json.loads(Path(source["data_path"]).read_text())
results = []
eligible = []
for row in rows:
    t = Task.model_validate(row)
    if t.id in old:
        continue
    try:
        if not t.evaluation_criteria or not t.evaluation_criteria.actions:
            raise ValueError("no reference actions")
        history = list(t.initial_state.message_history or []) if t.initial_state else []
        env = get_environment()
        init = t.initial_state
        env.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=history,
        )
        for a in t.evaluation_criteria.actions:
            cls = AssistantMessage if a.requestor == "assistant" else UserMessage
            history.append(
                cls(
                    role=a.requestor,
                    tool_calls=[
                        ToolCall(
                            id=a.action_id,
                            name=a.name,
                            arguments=a.arguments,
                            requestor=a.requestor,
                        )
                    ],
                )
            )
            response = env.get_response(history[-1].tool_calls[0])
            if response.error:
                raise ValueError(response.content)
            history.append(response)
        history.append(
            AssistantMessage(
                role="assistant",
                content=" ".join(t.evaluation_criteria.communicate_info or ["Done."]),
            )
        )
        good = EnvironmentEvaluator.calculate_reward(get_environment, t, history).reward
        bad = EnvironmentEvaluator.calculate_reward(get_environment, t, []).reward
        comm = CommunicateEvaluator.calculate_reward(t, history).reward
        ok = good == 1 and bad == 0 and comm == 1
        results.append(
            {"task_id": t.id, "gold_db": good, "empty_db": bad, "gold_communicate": comm, "ok": ok}
        )
        if ok:
            eligible.append(row)
    except Exception as e:
        results.append({"task_id": t.id, "ok": False, "error": str(e)})
Path("artifacts/tier-validation-v3/pool-audit.json").write_text(json.dumps(results, indent=2))
print("eligible held-out tasks", len(eligible))
if len(eligible) < 30:
    raise ValueError("Not enough qualified held-out tasks")
chosen = random.Random("tier-validation-v3-2026-09-21").sample(
    sorted(eligible, key=lambda r: r["id"]), 30
)
source["tasks"] = [{"id": r["id"], "sha256": fingerprint(r)} for r in chosen]
source.pop("qualification_exclusions", None)
source["selection"] = (
    "Seeded random 30 tasks passing offline gold DB/communication and "
    "negative empty-trajectory checks; excludes all ten v2 pilot tasks."
)
source["selection_rule"] = source["selection"]
spec["selection_seed"] = "tier-validation-v3-2026-09-21"
spec["trials"] = 3
spec.pop("budget_levels_usd", None)
spec["models"][0]["episode_budget_usd"] = 0.10
spec["controls"].update(
    simulator_episode_budget_usd=0.06,
    role_budget_usd={"agent": 7.0, "auxiliary": 2.4},
    temperature=0,
    require_admission_qualification=True,
    stop_on_budget_exhaustion=True,
    budget_policy=(
        "Safety ceilings only; budget stops invalidate performance interpretation. "
        "No budget-scaling claim."
    ),
)
spec["study_intent"] = (
    "Fresh held-out paired retail validation of requested realtime vs async: correctness, "
    "cost per successful task, and success by deadline. Descriptive task-clustered uncertainty; "
    "no claim of equivalence or general benchmark superiority."
)
Path("benchmarks/flex-agentic/tier-validation-v3.json").write_text(
    json.dumps(spec, indent=2) + "\n"
)
print("selected", [r["id"] for r in chosen])
