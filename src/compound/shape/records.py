"""One model call reduced to its shape: sizes, timing and prefix structure, no text.

A workload's cost and speed on another host depend on how big its requests are,
when they arrive, and how much of each prompt repeats. None of that needs the
words themselves, so a shape record keeps none of them. That is what lets a team
under KVKK or a bank's outsourcing rules hand us their traffic shape when they
cannot hand us their traffic.

Prompt reuse is kept as a *prefix chain*: one hash per boundary in the prompt
(the tool list, then each message in order), each hash covering everything
before it. Two calls that share their first k elements share their first k
hashes, which is exactly the condition a provider's prompt cache keys on. Every
hash is salted, and the salt never leaves the machine that made the export, so
nobody can recover a common system prompt by hashing guesses at it. The cost is
that hashes only compare within one export (or one SDK install).
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = "compound.shape"
SCHEMA_VERSION = 1

#: Hex characters kept per prefix hash. 64 bits is far beyond collision range
#: for any one workload and keeps a long conversation's chain small.
HASH_CHARS = 16

#: Letters that only Turkish uses among Latin-script languages we expect.
TURKISH_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")


def new_salt() -> bytes:
    """A fresh salt for one export. Kept in memory only, never written out."""
    return secrets.token_bytes(32)


@dataclass(slots=True)
class CallShape:
    """What one model call looked like, with no prompt or completion text.

    ``None`` always means "not reported". A measured zero and an unknown are
    different claims, and only measured values belong in a rate's denominator.
    """

    started_at: float | None
    ended_at: float | None = None
    latency_ms: float | None = None
    model: str | None = None
    provider: str | None = None
    task_key: str | None = None
    trace_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    message_count: int | None = None
    tool_count: int | None = None
    prompt_chars: int | None = None
    lang_hint: str | None = None
    #: ``[[hash, cumulative_chars], ...]``: tool list first, then each message.
    prefix_chain: list[list[Any]] | None = None
    status: str | None = None
    finish_reason: str | None = None
    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION}
        out.update({k: v for k, v in asdict(self).items() if v is not None and v != {}})
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CallShape:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} record")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported {SCHEMA} version {data.get('schema_version')!r}")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def completed_at(self) -> float | None:
        if self.ended_at is not None:
            return self.ended_at
        if self.started_at is not None and self.latency_ms is not None:
            return self.started_at + self.latency_ms / 1000
        return None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _message_text(message: dict[str, Any]) -> str:
    """The parts of a message a provider tokenizes, in a stable form."""
    keep = {k: message.get(k) for k in ("role", "content", "tool_calls", "tool_call_id", "name")}
    return _canonical({k: v for k, v in keep.items() if v is not None})


def _plain_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


def lang_hint(text: str) -> str | None:
    """A cheap guess at the script and language, for tokenizer cost ratios.

    ``tr`` when Turkish-only letters make up at least 1% of letters, ``latin``
    for other mostly-Latin text, ``other`` for everything else. It is a hint for
    picking a tokenizer ratio, not language detection, and is labelled as such.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    turkish = sum(ch in TURKISH_LETTERS for ch in letters)
    if turkish / len(letters) >= 0.01:
        return "tr"
    latin = sum(ch.isascii() or "À" <= ch <= "ɏ" for ch in letters)
    return "latin" if latin / len(letters) >= 0.9 else "other"


def prefix_chain(
    messages: list[dict[str, Any]], tools: list[Any] | None, salt: bytes
) -> list[list[Any]]:
    """Salted cumulative hashes at each prompt boundary, with cumulative sizes.

    The tool list comes first because providers place tool definitions ahead of
    the messages in the cached prefix; a changed tool list breaks every hit
    after it, and the chain shows that the same way.
    """
    digest = hashlib.sha256(salt)
    chain: list[list[Any]] = []
    chars = 0
    elements = [_canonical(tools or [])] + [_message_text(m) for m in messages]
    for text in elements:
        digest.update(b"\x00" + text.encode("utf-8"))
        chars += len(text)
        chain.append([digest.copy().hexdigest()[:HASH_CHARS], chars])
    return chain


def shape_from_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[Any] | None,
    salt: bytes,
    **fields: Any,
) -> CallShape:
    """Build a shape from a request's messages, then drop the messages."""
    chain = prefix_chain(messages, tools, salt)
    return CallShape(
        message_count=len(messages),
        tool_count=len(tools or []),
        prompt_chars=chain[-1][1] if chain else 0,
        lang_hint=lang_hint(_plain_text(messages)),
        prefix_chain=chain,
        **fields,
    )
