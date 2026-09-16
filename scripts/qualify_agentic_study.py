"""Replay reference answers without making inference calls.

Run with the selected benchmark checkout on PYTHONPATH; see benchmarks/flex-agentic/README.md.
"""

import argparse


def qualify_retail():
    import json
    from pathlib import Path

    from tau2.data_model.tasks import Task
    from tau2.domains.retail.environment import get_environment

    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    ids = {x["id"] for x in spec["sources"]["retail"]["tasks"]}
    rows = json.loads(Path(spec["sources"]["retail"]["data_path"]).read_text())
    results = []
    for row in rows:
        t = Task.model_validate(row)
        e = get_environment()
        init = t.initial_state
        e.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=init.message_history if init else [],
        )
        before = e.get_db_hash()
        errors = []
        for a in t.evaluation_criteria.actions:
            try:
                result = e.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
            except Exception as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
                continue
            if getattr(result, "error", None):
                errors.append(str(result.error))
            if "error" in str(getattr(result, "content", "")).lower():
                errors.append(str(result.content))
        changed = before != e.get_db_hash()
        results.append(
            {
                "task_id": t.id,
                "gold_replay_errors": errors,
                "state_changed": changed,
                "reward_basis": [x.value for x in t.evaluation_criteria.reward_basis],
            }
        )
    Path("artifacts/flex-agentic-preflight/retail-qualification.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(
        {
            "checked": len(results),
            "valid_mutating": sum(
                not r["gold_replay_errors"] and r["state_changed"] for r in results
            ),
            "selected": [r for r in results if r["task_id"] in ids],
        }
    )
    selected = [r for r in results if r["task_id"] in ids]
    if len(selected) != len(ids) or any(
        r["gold_replay_errors"] or not r["state_changed"] for r in selected
    ):
        raise ValueError("Selected retail tasks failed reference replay qualification")


def qualify_finance():
    import copy
    import json
    from pathlib import Path

    from finbalance.benchmark.analysis import posting_to_dict, serialize_balance_sheet
    from finbalance.benchmark.dataset import load_records
    from finbalance.benchmark.parser import parse_submission
    from finbalance.benchmark.prompt import build_prompt
    from finbalance.benchmark.scoring import score_submission
    from finbalance.benchmark.tools import ledger_check_tool

    from compound.adapters.finbalance import run_case
    from compound.agentic_study import finance_success

    spec = json.loads(Path("benchmarks/flex-agentic/pilot.json").read_text())
    ids = {x["id"] for x in spec["sources"]["finance"]["tasks"]}
    results = []
    for r in load_records(spec["sources"]["finance"]["data_path"]):
        if r.record_id not in ids:
            continue
        gold = {
            "has_inconsistency": r.expected_inconsistency,
            "inconsistency_codes": r.expected_inconsistency_codes,
            "inconsistency_notes": [],
            "entries": []
            if r.expected_inconsistency
            else [posting_to_dict(x) for x in r.expected_entries],
            "balance_sheet": {"assets": {}, "liabilities": {}, "equity": {}}
            if r.expected_inconsistency
            else serialize_balance_sheet(r.expected_balance_sheet),
        }
        parsed = parse_submission(json.dumps(gold))
        score = score_submission(r, parsed, parse_success=True)
        assert finance_success(score), (r.record_id, score)
        bad = copy.deepcopy(gold)
        if r.expected_inconsistency:
            bad["has_inconsistency"] = False
            bad["inconsistency_codes"] = []
        else:
            bad["entries"][0]["amount"] += 100
        assert not finance_success(
            score_submission(r, parse_submission(json.dumps(bad)), parse_success=True)
        ), r.record_id
        # Hidden expected answers must not influence visible prompt/tools.
        stripped = copy.deepcopy(r)
        stripped.expected_entries = []
        stripped.expected_inconsistency_codes = []
        stripped.expected_inconsistency = False
        stripped.expected_balance_sheet = None
        stripped.inconsistency_reasons = []
        assert build_prompt(r) == build_prompt(stripped)
        assert ledger_check_tool({"entries": gold["entries"]}, r) == ledger_check_tool(
            {"entries": gold["entries"]}, stripped
        )

        class ReferenceClient:
            """Test double: exercise tool execution and grading without a provider."""

            def __init__(self, answer):
                self.answer = answer
                self.calls = 0

            def complete_messages(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    assert kwargs["tools"]
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "test-calculator",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"2+2"}',
                                },
                            }
                        ],
                    }
                    text = ""
                else:
                    assert messages[-1]["role"] == "tool"
                    text = json.dumps(self.answer)
                    message = {"role": "assistant", "content": text}
                return text, {"choices": [{"message": message}], "usage": {}}

        client = ReferenceClient(gold)
        trial = run_case(r, client)
        assert trial["success"] and client.calls == 2 and len(trial["tool_calls"]) == 1
        results.append(
            {
                "task_id": r.record_id,
                "gold_passes": True,
                "wrong_answer_fails": True,
                "hidden_answer_invariance": True,
                "native_tool_loop_reference_passes": True,
            }
        )
    Path("artifacts/flex-agentic-preflight/finance-qualification.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(results)
    if {r["task_id"] for r in results} != ids:
        raise ValueError("Finance qualification did not cover all selected tasks")


if __name__ == "__main__":
    from pathlib import Path

    Path("artifacts/flex-agentic-preflight").mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("retail", "finance"))
    args = parser.parse_args()
    if args.suite == "retail":
        qualify_retail()
    else:
        qualify_finance()
