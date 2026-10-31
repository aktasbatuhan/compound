"""Record calls made through an OpenAI (or OpenAI-compatible) Python client.

``wrap(client, recorder)`` replaces ``client.chat.completions.create`` with a
version that calls the original and then records the call's shape. The
response comes back unchanged. Streams are handed back as a thin proxy that
behaves like the original stream and records when it is exhausted or closed;
usage is only present if the app asked for it with
``stream_options={"include_usage": True}``, as the API requires.

The Responses API (``client.responses``) is not wrapped yet; use
``Recorder.record`` for it.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from compound_sdk.recorder import Recorder


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _finish(response: Any) -> str | None:
    choices = _get(response, "choices") or []
    return _get(choices[0], "finish_reason") if choices else None


def _host(client: Any) -> str | None:
    base = getattr(client, "base_url", None)
    return getattr(base, "host", None) or (str(base) if base else None)


class _Call:
    def __init__(self, recorder: Recorder, kwargs: dict[str, Any], host: str | None) -> None:
        self.recorder, self.kwargs, self.host = recorder, kwargs, host
        self.started = time.time()
        self.usage: Any = None
        self.model: str | None = kwargs.get("model")
        self.finish: str | None = None
        self.done = False

    def observe_chunk(self, chunk: Any) -> None:
        usage = _get(chunk, "usage")
        if usage:
            self.usage = usage
        self.model = _get(chunk, "model") or self.model
        self.finish = _finish(chunk) or self.finish

    def finish_call(self, *, response: Any = None, error: bool = False) -> None:
        if self.done:
            return
        self.done = True
        if response is not None:
            self.usage = _get(response, "usage")
            self.model = _get(response, "model") or self.model
            self.finish = _finish(response)
        self.recorder.record(
            messages=self.kwargs.get("messages") or [],
            tools=self.kwargs.get("tools"),
            model=self.model,
            started_at=self.started,
            ended_at=time.time(),
            usage=self.usage,
            provider=self.host,
            finish_reason=self.finish,
            error=error,
        )


class _StreamProxy:
    """Looks like the SDK's stream; records the call once it ends."""

    def __init__(self, stream: Any, call: _Call) -> None:
        self._stream, self._call = stream, call

    def __iter__(self):
        try:
            for chunk in self._stream:
                self._call.observe_chunk(chunk)
                yield chunk
        except BaseException:
            self._call.finish_call(error=True)
            raise
        self._call.finish_call()

    def __enter__(self):
        enter = getattr(self._stream, "__enter__", None)
        if enter:
            enter()
        return self

    def __exit__(self, *exc):
        self._call.finish_call(error=exc[0] is not None)
        exit_ = getattr(self._stream, "__exit__", None)
        return exit_(*exc) if exit_ else False

    def close(self):
        self._call.finish_call()
        close = getattr(self._stream, "close", None)
        if close:
            close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _AsyncStreamProxy:
    def __init__(self, stream: Any, call: _Call) -> None:
        self._stream, self._call = stream, call

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                self._call.observe_chunk(chunk)
                yield chunk
        except BaseException:
            self._call.finish_call(error=True)
            raise
        self._call.finish_call()

    async def __aenter__(self):
        enter = getattr(self._stream, "__aenter__", None)
        if enter:
            await enter()
        return self

    async def __aexit__(self, *exc):
        self._call.finish_call(error=exc[0] is not None)
        exit_ = getattr(self._stream, "__aexit__", None)
        return await exit_(*exc) if exit_ else False

    async def close(self):
        self._call.finish_call()
        close = getattr(self._stream, "close", None)
        if close:
            await close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def wrap(client: Any, recorder: Recorder) -> Any:
    completions = client.chat.completions
    original = completions.create
    if getattr(original, "_compound_wrapped", False):
        return client
    host = _host(client)

    if inspect.iscoroutinefunction(original):

        async def create(*args: Any, **kwargs: Any) -> Any:
            call = _Call(recorder, kwargs, host)
            try:
                result = await original(*args, **kwargs)
            except BaseException:
                call.finish_call(error=True)
                raise
            if kwargs.get("stream"):
                return _AsyncStreamProxy(result, call)
            call.finish_call(response=result)
            return result

    else:

        def create(*args: Any, **kwargs: Any) -> Any:
            call = _Call(recorder, kwargs, host)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                call.finish_call(error=True)
                raise
            if kwargs.get("stream"):
                return _StreamProxy(result, call)
            call.finish_call(response=result)
            return result

    create._compound_wrapped = True  # type: ignore[attr-defined]
    completions.create = create
    return client
