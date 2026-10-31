"""A LiteLLM logger that records the shape of every call routed through LiteLLM.

Works for the LiteLLM SDK and the LiteLLM proxy:

    import litellm
    from compound_sdk.litellm import compound_logger
    litellm.callbacks = [compound_logger()]

``record_from_litellm`` holds the mapping and does not import LiteLLM, so it
can be tested and reused without it installed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from compound_sdk.recorder import Recorder


def _epoch(value: Any) -> float | None:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, int | float):
        return float(value)
    return None


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def record_from_litellm(
    recorder: Recorder,
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    *,
    error: bool = False,
) -> None:
    params = kwargs.get("optional_params") or {}
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    choices = _get(response_obj, "choices") or []
    recorder.record(
        messages=kwargs.get("messages") or [],
        tools=params.get("tools") or kwargs.get("tools"),
        model=_get(response_obj, "model") or kwargs.get("model"),
        started_at=_epoch(start_time),
        ended_at=_epoch(end_time),
        usage=_get(response_obj, "usage"),
        provider=litellm_params.get("custom_llm_provider") or kwargs.get("custom_llm_provider"),
        task_key=metadata.get("task_key"),
        trace_id=metadata.get("trace_id") or kwargs.get("litellm_call_id"),
        finish_reason=_get(choices[0], "finish_reason") if choices else None,
        error=error,
    )


def compound_logger(recorder: Recorder | None = None) -> Any:
    """A ``litellm`` CustomLogger instance that writes shapes via ``recorder``."""
    try:
        from litellm.integrations.custom_logger import CustomLogger
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("compound_logger needs litellm installed") from exc

    rec = recorder or Recorder()

    class CompoundLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_from_litellm(rec, kwargs, response_obj, start_time, end_time)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            record_from_litellm(rec, kwargs, response_obj, start_time, end_time, error=True)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_from_litellm(rec, kwargs, response_obj, start_time, end_time)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            record_from_litellm(rec, kwargs, response_obj, start_time, end_time, error=True)

    return CompoundLogger()
