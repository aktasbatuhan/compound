"""Answer a candidate's tool calls without touching anything real by accident.

Each tool follows its replay policy, the same four the TypeScript trajectory
runner uses:

- ``recorded``: return what the real tool returned for the same name and
  arguments. A call the recording never saw has no honest answer, so the
  episode stops with ``missing_recorded_result`` rather than inventing one.
- ``mocked``: return a fixed result (``mock_results`` per tool, else ``{}``).
- ``live_read_only``: call a function the caller supplies, for tools that only
  read. Without one, the episode stops with ``unsupported_policy``.
- ``blocked``: a tool with side effects that must never run; the episode stops.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from compound.eval.simulation.scenario import Action, Scenario, canonical_args


@dataclass(frozen=True)
class ToolResult:
    content: str | None
    #: None when the call was answered; otherwise the reason the episode stops.
    stop: str | None = None


@dataclass
class Environment:
    recorded: dict[Action, str]
    policies: dict[str, str]
    default_policy: str = "recorded"
    mock_results: dict[str, str] = field(default_factory=dict)
    live: dict[str, Callable[[dict[str, Any]], str]] = field(default_factory=dict)

    @classmethod
    def for_scenario(cls, scenario: Scenario, **kwargs: Any) -> Environment:
        return cls(recorded=scenario.recorded, policies=dict(scenario.policies), **kwargs)

    def call(self, tool: str, args: Any) -> ToolResult:
        policy = self.policies.get(tool, self.default_policy)
        if policy == "recorded":
            result = self.recorded.get(Action(tool, canonical_args(args)))
            if result is None:
                return ToolResult(None, "missing_recorded_result")
            return ToolResult(result)
        if policy == "mocked":
            return ToolResult(self.mock_results.get(tool, "{}"))
        if policy == "live_read_only":
            fn = self.live.get(tool)
            if fn is None:
                return ToolResult(None, "unsupported_policy")
            parsed = args
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    parsed = {}
            return ToolResult(str(fn(parsed if isinstance(parsed, dict) else {})))
        return ToolResult(None, "tool_blocked")
