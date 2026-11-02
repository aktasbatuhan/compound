"""Simulated users: who speaks after the candidate model answers.

``ScriptedUser`` replays the real user's turns in order, whatever the model
said. It costs nothing and is exact when the user's next message does not
depend on the answer (form-like flows, one-shot requests with follow-ups).

``LLMUser`` plays the real user with a model. It is told what the real user
said, in order, and asked to keep pursuing the same goal in the same language,
one message at a time, reacting to what the candidate actually replied. When
the goal is reached, or the real user would have given up, it answers
``###STOP###``. This is the tau-bench pattern; whether it is faithful enough
for a given task is not assumed but checked (see ``verdict.fidelity``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from compound.eval.simulation.chat import ChatFn
from compound.eval.simulation.scenario import Scenario

STOP = "###STOP###"
LANG_NAMES = {"tr": "Turkish", "latin": "the same language as the real user"}


class User(Protocol):
    def first(self, scenario: Scenario) -> str: ...
    def next(self, scenario: Scenario, visible: list[dict[str, str]]) -> str | None: ...


class ScriptedUser:
    def first(self, scenario: Scenario) -> str:
        return scenario.user_turns[0]

    def next(self, scenario: Scenario, visible: list[dict[str, str]]) -> str | None:
        said = sum(1 for m in visible if m["role"] == "user")
        return scenario.user_turns[said] if said < len(scenario.user_turns) else None


def user_brief(scenario: Scenario) -> str:
    turns = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(scenario.user_turns))
    language = LANG_NAMES.get(scenario.lang or "", "the same language as the real user")
    return (
        "You are role-playing the user of an AI assistant, not the assistant. "
        "Below are the messages the real user sent, in order, in a real conversation. "
        "Keep pursuing the same goal with the same facts, tone and level of detail, one "
        "message at a time, reacting to what the assistant actually says. Do not invent "
        "new requests, do not help the assistant, and never say you are simulated. Write "
        f"in {language}. When your goal has been met, or the real user would have given "
        f"up, reply with exactly {STOP} and nothing else.\n\n"
        f"What the real user said:\n{turns}"
    )


@dataclass
class LLMUser:
    chat: ChatFn
    #: Spent on the simulated user, kept apart from the candidate's spend.
    usage: list[dict[str, Any]] = field(default_factory=list)

    def first(self, scenario: Scenario) -> str:
        # Opening with the real first message keeps every model on the same start.
        return scenario.user_turns[0]

    def next(self, scenario: Scenario, visible: list[dict[str, str]]) -> str | None:
        # From the simulator's side the roles flip: the assistant's replies are
        # what it is talking to, and its own earlier lines are its turns.
        flipped = [
            {"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
            for m in visible
        ]
        result = self.chat([{"role": "system", "content": user_brief(scenario)}, *flipped], None)
        self.usage.append({"cost_usd": result.cost_usd, "usage": result.usage})
        text = (result.message.get("content") or "").strip()
        if not text or STOP in text:
            return None
        return text
