"""Record the shape of every model call an app makes, off the request path.

The recorder never sits between the app and the model. A call is made as
usual; afterwards its shape (sizes, timing, salted prefix hashes, no text) is
put on a queue and a background thread appends it to a JSONL file. If the
queue is full, the record is dropped and counted; if anything in recording
fails, the app never sees it. Losing a sample is acceptable, slowing or
breaking the app is not.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from compound_sdk.shape import build_record

DEFAULT_DIR = Path(os.environ.get("COMPOUND_HOME", "~/.compound")).expanduser()


def load_or_create_salt(path: Path) -> bytes:
    """A per-install secret, readable by this user only.

    Records from one install share it, so their prefix hashes compare across
    files and days. It never leaves the machine, which is what keeps the
    hashes from being reversed by guessing common prompts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    salt = secrets.token_bytes(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another process won the race; use theirs
        return path.read_bytes()
    with os.fdopen(fd, "wb") as handle:
        handle.write(salt)
    return salt


class Recorder:
    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        task_key: str | None = None,
        salt: bytes | None = None,
        max_queue: int = 10_000,
    ) -> None:
        self.path = Path(path).expanduser() if path else DEFAULT_DIR / "shapes.jsonl"
        self.task_key = task_key
        self.salt = salt if salt is not None else load_or_create_salt(DEFAULT_DIR / "salt")
        self.dropped = 0
        self.errors = 0
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, name="compound-sdk", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # -- recording -------------------------------------------------------

    def record(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None,
        started_at: float | None,
        ended_at: float | None = None,
        usage: Any = None,
        tools: list[Any] | None = None,
        provider: str | None = None,
        task_key: str | None = None,
        trace_id: str | None = None,
        finish_reason: str | None = None,
        error: bool = False,
    ) -> None:
        """Queue one call's shape. Never raises."""
        try:
            record = build_record(
                messages=list(messages or []),
                tools=tools,
                salt=self.salt,
                started_at=started_at,
                ended_at=ended_at if ended_at is not None else time.time(),
                model=model,
                usage=usage,
                provider=provider,
                task_key=task_key or self.task_key,
                trace_id=trace_id,
                finish_reason=finish_reason,
                error=error,
            )
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.errors += 1

    def wrap_openai(self, client: Any) -> Any:
        """Record every ``chat.completions.create`` call on an OpenAI client."""
        from compound_sdk.openai import wrap

        return wrap(client, self)

    # -- writing ---------------------------------------------------------

    def _drain(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                with self.path.open("a") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception:
                self.errors += 1
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until queued records are written, up to ``timeout`` seconds."""
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.01)

    def close(self) -> None:
        if self._thread.is_alive():
            self.flush()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                return
            self._thread.join(timeout=2)
