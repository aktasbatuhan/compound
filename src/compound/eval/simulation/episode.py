"""Play one scenario: a model, a simulated user and replayed tools, turn by turn.

The model answers; if it calls tools, the environment answers them and the
model goes again; once it replies in text, the simulated user speaks or ends
the conversation. The episode stops when the user is done (``user_done``,
the normal end), when a tool cannot be answered under its policy, when the
model-call budget runs out (``truncated``), or when the spend cap is reached
(``budget``). Every stop reason is kept, because a model that wanders into a
tool call nobody recorded has already diverged from what the real one did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from compound.eval.simulation.chat import ChatFn
from compound.eval.simulation.environment import Environment
from compound.eval.simulation.scenario import Action, Scenario, canonical_args
from compound.eval.simulation.users import User

DEFAULT_MAX_MODEL_CALLS = 24


@dataclass
class Episode:
    trace_id: str
    model: str
    stop: str
    actions: list[Action]
    transcript: list[dict[str, Any]]
    model_calls: int = 0
    user_turns: int = 0
    cost_usd: float = 0.0
    cost_complete: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    stop_tool: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in self.__dict__.items() if k != "actions"},
            "actions": [a.label() for a in self.actions],
        }


class SpendCap:
    """Stops new model calls once settled spend reaches the cap.

    Episodes run one call at a time, so at most one call can be in flight when
    the cap is reached; the overshoot is bounded by a single call.
    """

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent = 0.0

    @property
    def reached(self) -> bool:
        return self.spent >= self.max_usd

    def add(self, usd: float | None) -> None:
        self.spent += usd or 0.0


def _visible(transcript: list[dict[str, Any]]) -> list[dict[str, str]]:
    """What a person in the chat would see: user lines and assistant text."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in transcript
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]


def run_episode(
    scenario: Scenario,
    chat: ChatFn,
    user: User,
    env: Environment,
    *,
    model: str,
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
    cap: SpendCap | None = None,
) -> Episode:
    transcript: list[dict[str, Any]] = [*scenario.system,
                                        {"role": "user", "content": user.first(scenario)}]
    episode = Episode(scenario.trace_id, model, "truncated", [], transcript, user_turns=1)
    tools = scenario.tools or None
    while episode.model_calls < max_model_calls:
        if cap is not None and cap.reached:
            episode.stop = "budget"
            return episode
        result = chat(transcript, tools)
        episode.model_calls += 1
        episode.latency_ms += result.latency_ms
        episode.input_tokens += int(result.usage.get("prompt_tokens") or 0)
        episode.output_tokens += int(result.usage.get("completion_tokens") or 0)
        if result.cost_usd is None:
            episode.cost_complete = False
        episode.cost_usd += result.cost_usd or 0.0
        if cap is not None:
            cap.add(result.cost_usd)
        message = result.message
        transcript.append(message)
        calls = message.get("tool_calls") or []
        if calls:
            for call in calls:
                fn = call.get("function") or {}
                name, args = fn.get("name") or "", fn.get("arguments")
                episode.actions.append(Action(name, canonical_args(args)))
                answer = env.call(name, args)
                if answer.stop:
                    episode.stop, episode.stop_tool = answer.stop, name
                    return episode
                transcript.append({"role": "tool", "tool_call_id": call.get("id"),
                                   "content": answer.content})
            continue
        reply = user.next(scenario, _visible(transcript))
        if reply is None:
            episode.stop = "user_done"
            return episode
        transcript.append({"role": "user", "content": reply})
        episode.user_turns += 1
    return episode


def dumps(episode: Episode) -> str:
    return json.dumps(episode.to_json(), ensure_ascii=False, default=str)
