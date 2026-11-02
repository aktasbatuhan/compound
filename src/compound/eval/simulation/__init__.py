"""Evaluate a candidate model on real interactive traces with a simulated user."""

from compound.eval.simulation.chat import ChatFn, ChatResult, provider_chat
from compound.eval.simulation.environment import Environment
from compound.eval.simulation.episode import Episode, SpendCap, run_episode
from compound.eval.simulation.scenario import Action, Scenario, scenario_from_trace
from compound.eval.simulation.users import LLMUser, ScriptedUser
from compound.eval.simulation.verdict import decide, fidelity, grade

__all__ = [
    "Action",
    "ChatFn",
    "ChatResult",
    "Environment",
    "Episode",
    "LLMUser",
    "Scenario",
    "ScriptedUser",
    "SpendCap",
    "decide",
    "fidelity",
    "grade",
    "provider_chat",
    "run_episode",
    "scenario_from_trace",
]
