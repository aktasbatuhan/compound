"""Turn a real interactive trace into a scenario a simulation can replay.

A scenario keeps what a candidate model needs to meet the same situation:
the system prompt and tools it ran with, every user turn in order, the tool
results that were recorded, and what the production model actually did (its
tool calls), which is the reference the candidate is graded against.

It is read from a ``compound.trace`` document's last model call, whose input
holds the conversation so far and whose output is the final assistant turn.
Tool results come from two places: tool messages in that history (matched to
the assistant tool call that asked for them) and ``tool_execution`` steps,
which also carry each tool's replay policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from compound.shape.records import lang_hint

POLICIES = ("recorded", "mocked", "live_read_only", "blocked")


def canonical_args(args: Any) -> str:
    """One stable string per set of arguments, whatever order keys came in."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    return json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Action:
    """One tool call, compared by name and arguments."""

    tool: str
    args: str  # canonical JSON

    def label(self) -> str:
        return f"{self.tool}({self.args})"


@dataclass
class Scenario:
    trace_id: str
    task_key: str | None
    incumbent_model: str | None
    system: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    user_turns: list[str]
    reference_actions: list[Action]
    recorded: dict[Action, str]
    policies: dict[str, str]
    lang: str | None
    outcome: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def interactive(self) -> bool:
        return len(self.user_turns) > 1 or bool(self.reference_actions)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("text")
        )
    return ""


def wire_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Contract tool definitions as chat-completions ``tools``."""
    out = []
    for tool in tools or []:
        if tool.get("type") == "function" and "function" in tool:
            out.append(tool)
            continue
        out.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            },
        })
    return out


def _calls(message: dict[str, Any]) -> list[tuple[str | None, str, Any]]:
    """(id, name, arguments) for each tool call, in contract or wire form."""
    out = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = call.get("name") or fn.get("name")
        args = call.get("arguments", fn.get("arguments"))
        if name:
            out.append((call.get("id"), name, args))
    return out


def scenario_from_trace(doc: dict[str, Any], default_policy: str = "recorded") -> Scenario | None:
    """The scenario in one trace, or None when it has no model call to replay."""
    calls = [s for s in doc.get("steps") or [] if s.get("type") == "model_call"]
    if not calls:
        return None
    last = calls[-1]
    history = [m for m in last.get("input") or [] if isinstance(m, dict)]
    if isinstance(last.get("output"), dict):
        history = history + [{**last["output"], "role": "assistant"}]

    system = [m for m in history if m.get("role") == "system"]
    user_turns = [_text(m.get("content")) for m in history if m.get("role") == "user"]
    reference: list[Action] = []
    by_id: dict[str, Action] = {}
    recorded: dict[Action, str] = {}
    for message in history:
        if message.get("role") == "assistant":
            for call_id, name, args in _calls(message):
                action = Action(name, canonical_args(args))
                reference.append(action)
                if call_id:
                    by_id[call_id] = action
        elif message.get("role") == "tool":
            action = by_id.get(message.get("tool_call_id") or "")
            if action is not None:
                recorded.setdefault(action, _text(message.get("content")))

    policies: dict[str, str] = {}
    notes: list[str] = []
    for step in doc.get("steps") or []:
        if step.get("type") != "tool_execution" or not step.get("name"):
            continue
        action = Action(step["name"], canonical_args(step.get("input")))
        output = step.get("output")
        if action not in recorded and output is not None:
            recorded[action] = output if isinstance(output, str) else json.dumps(
                output, ensure_ascii=False
            )
        policy = step.get("replay_policy")
        if policy in POLICIES:
            policies[step["name"]] = policy
    for action in reference:
        policies.setdefault(action.tool, default_policy)
        if policies[action.tool] == "recorded" and action not in recorded:
            notes.append(f"no recorded result for {action.label()}")

    tools = wire_tools(last.get("tools_available"))
    for tool in tools:
        policies.setdefault(tool["function"]["name"], default_policy)
    return Scenario(
        trace_id=str(doc.get("trace_id") or ""),
        task_key=doc.get("task_key"),
        incumbent_model=last.get("resolved_model") or last.get("model"),
        system=[{"role": "system", "content": _text(m.get("content"))} for m in system],
        tools=tools,
        user_turns=[t for t in user_turns if t],
        reference_actions=reference,
        recorded=recorded,
        policies=policies,
        lang=lang_hint(" ".join(user_turns)),
        outcome=doc.get("outcome"),
        notes=notes,
    )
