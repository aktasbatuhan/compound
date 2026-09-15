"""Thin adapter to FinBalance's existing native tool loop and official scorer.

Install the pinned source separately. A caller supplies the instrumented model
client; this module never discovers keys or creates a provider implicitly.
"""

from __future__ import annotations

import time
from typing import Any

from compound.agentic_study import finance_success


def run_case(
    record: Any,
    client: Any,
    *,
    max_steps: int = 8,
    max_tokens: int = 8192,
    timeout: int = 600,
) -> dict:
    from finbalance.benchmark.parser import SubmissionParseError, parse_submission
    from finbalance.benchmark.prompt import build_prompt
    from finbalance.benchmark.scoring import score_submission
    from finbalance.benchmark.tools import (
        TOOL_VARIANT_NATIVE_FULL_TOOL_AGENT,
        TOOLS_BY_VARIANT,
        run_native_tool_agent_completion,
    )

    if min(max_steps, max_tokens, timeout) < 1:
        raise ValueError("step, output, and timeout limits must be positive")
    started = time.monotonic()
    # Upstream builds the visible prompt and tools. Hidden answers are used only
    # by its grader; no oracle visibility variants or answer repair are enabled.
    completion = run_native_tool_agent_completion(
        record,
        client,
        build_prompt(record),
        allowed_tools=TOOLS_BY_VARIANT[TOOL_VARIANT_NATIVE_FULL_TOOL_AGENT],
        agent_max_steps=max_steps,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    try:
        parsed = parse_submission(completion.response_text)
    except SubmissionParseError:
        metrics = {"parse_success": False}
    else:
        metrics = score_submission(record, parsed, parse_success=True)
    return {
        "task_id": record.record_id,
        "status": "graded",
        "success": finance_success(metrics),
        "duration_s": time.monotonic() - started,
        "metrics": metrics,
        "tool_calls": completion.tool_calls,
        "tool_call_failures": completion.tool_call_failures,
        "response_payload": completion.response_payload,
        "response_text": completion.response_text,
    }
