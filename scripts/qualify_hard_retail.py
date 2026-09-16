"""Offline positive and negative checks for the three selected retail tasks."""

import json
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from compound.agentic_study import verify_sources


def main():
    spec = json.loads(Path("benchmarks/flex-agentic/budget-hard-three.json").read_text())
    assert not verify_sources(spec)
    ids = {t["id"] for t in spec["sources"]["retail"]["tasks"]}
    rows = json.loads(Path(spec["sources"]["retail"]["data_path"]).read_text())
    results = []
    for raw in rows:
        if raw["id"] not in ids:
            continue
        task = Task.model_validate(raw)
        env = get_environment()
        init = task.initial_state
        env.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=init.message_history if init else [],
        )
        trajectory = list(init.message_history if init else [])
        for i, action in enumerate(task.evaluation_criteria.actions):
            call = ToolCall(
                id=str(i), name=action.name, arguments=action.arguments, requestor=action.requestor
            )
            result = env.get_response(call)
            assert not result.error, result
            trajectory.append(
                AssistantMessage(
                    role="assistant",
                    tool_calls=[call],
                )
            )
            trajectory.append(result)

        def grade(messages, task=task):
            return EnvironmentEvaluator.calculate_reward(get_environment, task, messages).reward

        positive = grade(trajectory)
        negative = grade([])
        assert positive == 1 and negative == 0, (task.id, positive, negative)
        visible = json.dumps([message.model_dump() for message in trajectory])
        expected_communication = task.evaluation_criteria.communicate_info or []
        assert all(value in visible for value in expected_communication)
        results.append(
            {
                "task_id": task.id,
                "reference_calls": len(task.evaluation_criteria.actions),
                "reference_reward": positive,
                "empty_trajectory_reward": negative,
                "communication_items_visible": len(expected_communication),
                "communication_judge_checked_offline": False,
            }
        )
    assert {r["task_id"] for r in results} == ids
    Path("artifacts/flex-hard-20260910/qualification.json").write_text(
        json.dumps(results, indent=2)
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
