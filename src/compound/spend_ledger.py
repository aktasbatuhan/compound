"""The shared paid-call contract, Python side (#52).

The TypeScript trace pipeline keeps a durable spend ledger in SQLite:
``spend_records`` (committed charges) and ``spend_reservations`` (estimates
reserved before a provider call, settled after it). Any Python code that spends
money on that pipeline's behalf, today the GEPA optimizer, must go through the
same tables under the same rules, or the per-run cap and the global hard limit
mean nothing. This module is the Python half of that contract. It creates no
tables: the schema is owned by the TypeScript migrations, and a database without
it is refused rather than silently extended.

The rules, mirrored from ``packages/storage/src/execution.ts``:

* A call reserves its estimate inside an IMMEDIATE transaction, against the
  committed ledger plus every live reservation, so two processes cannot both
  pass a check only one could afford.
* Completion settles the reservation at the actual charge in one transaction.
* A call that fails releases its reservation.
* A reservation older than the TTL is treated as left by a dead process.
* A completed call whose usage is missing is charged at the estimate, not $0.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

RESERVATION_TTL_MS = 15 * 60 * 1000
REQUIRED_TABLES = ("spend_records", "spend_reservations")


class BudgetExceededError(RuntimeError):
    def __init__(self, attempted: float, limit: float, scope: str) -> None:
        super().__init__(
            f"{scope} budget exceeded: ${attempted:.4f} would exceed the ${limit:.4f} limit"
        )
        self.attempted = attempted
        self.limit = limit
        self.scope = scope


@dataclass(frozen=True, slots=True)
class TokenPrice:
    """USD per million input and output tokens."""

    input: float
    output: float


def estimate_cost(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    price: TokenPrice,
) -> float:
    """Conservative pre-call estimate: ~4 chars per token, full output budget."""
    chars = len(json.dumps(messages, ensure_ascii=False))
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False))
    prompt_tokens = chars / 4
    return (prompt_tokens / 1e6) * price.input + (max_tokens / 1e6) * price.output


def charge_for(usage: Any, estimate: float, price: TokenPrice) -> tuple[float, bool]:
    """Actual charge from a usage block, else the estimate. Returns (usd, usage_known)."""
    if usage is None:
        return estimate, False
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0) + int(
        getattr(usage, "reasoning_tokens", 0) or 0
    )
    if inp == 0 and out == 0:
        return estimate, False
    return (inp / 1e6) * price.input + (out / 1e6) * price.output, True


def fingerprint(model: str, messages: list[dict[str, Any]], tools: Any, max_tokens: int) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "tools": tools, "max_tokens": max_tokens},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class SpendLedger:
    """Reserve / settle / release against the TypeScript pipeline's ledger."""

    def __init__(
        self,
        db_path: str,
        *,
        experiment_id: str,
        cap_usd: float,
        global_hard_limit_usd: float,
    ) -> None:
        if not (cap_usd > 0) or not (global_hard_limit_usd > 0):
            raise ValueError("cap_usd and global_hard_limit_usd must be positive")
        self.conn = sqlite3.connect(db_path, timeout=5.0, isolation_level=None)
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.experiment_id = experiment_id
        self.cap_usd = cap_usd
        self.global_hard_limit_usd = global_hard_limit_usd
        self.total_charged_usd = 0.0
        self.calls = 0
        self.calls_usage_unknown = 0
        present = {
            row[0]
            for row in self.conn.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
        missing = [t for t in REQUIRED_TABLES if t not in present]
        if missing:
            raise RuntimeError(
                f"{db_path} lacks {missing}; run the TypeScript migrations (any `compound` "
                "command) before optimizing. Refusing to spend outside the ledger."
            )

    # -- reads (inside a caller's transaction) --------------------------------

    def _committed(self, experiment_id: str | None = None) -> float:
        if experiment_id is None:
            row = self.conn.execute("select coalesce(sum(cost_usd), 0) from spend_records")
        else:
            row = self.conn.execute(
                "select coalesce(sum(cost_usd), 0) from spend_records where experiment_id = ?",
                (experiment_id,),
            )
        return float(row.fetchone()[0])

    def _open(self, experiment_id: str | None = None) -> float:
        cutoff = int(time.time() * 1000) - RESERVATION_TTL_MS
        self.conn.execute("delete from spend_reservations where created_at < ?", (cutoff,))
        if experiment_id is None:
            row = self.conn.execute("select coalesce(sum(reserved_usd), 0) from spend_reservations")
        else:
            row = self.conn.execute(
                "select coalesce(sum(reserved_usd), 0) from spend_reservations "
                "where experiment_id = ?",
                (experiment_id,),
            )
        return float(row.fetchone()[0])

    # -- the contract ----------------------------------------------------------

    def reserve(self, fp: str, estimated_usd: float) -> str:
        reservation_id = str(uuid.uuid4())
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            global_after = self._committed() + self._open() + estimated_usd
            if global_after > self.global_hard_limit_usd:
                raise BudgetExceededError(global_after, self.global_hard_limit_usd, "global")
            exp_after = (
                self._committed(self.experiment_id)
                + self._open(self.experiment_id)
                + estimated_usd
            )
            if exp_after > self.cap_usd:
                raise BudgetExceededError(exp_after, self.cap_usd, "experiment")
            self.conn.execute(
                "insert into spend_reservations (id, experiment_id, fingerprint, reserved_usd, "
                "created_at) values (?, ?, ?, ?, ?)",
                (reservation_id, self.experiment_id, fp, estimated_usd, int(time.time() * 1000)),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return reservation_id

    def settle(self, reservation_id: str, fp: str, cost_usd: float) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("delete from spend_reservations where id = ?", (reservation_id,))
            self.conn.execute(
                "insert into spend_records (id, experiment_id, fingerprint, cost_usd, created_at) "
                "values (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), self.experiment_id, fp, cost_usd, int(time.time() * 1000)),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.total_charged_usd += cost_usd
        self.calls += 1

    def release(self, reservation_id: str) -> None:
        self.conn.execute("delete from spend_reservations where id = ?", (reservation_id,))

    def paid_call(self, provider: Any, *, model: str, messages: list[dict[str, Any]],
                  tools: Any, max_tokens: int, price: TokenPrice) -> Any:
        """Reserve, call ``provider.complete``, settle. The only way to spend."""
        fp = fingerprint(model, messages, tools, max_tokens)
        estimate = estimate_cost(messages, tools or None, max_tokens, price)
        reservation_id = self.reserve(fp, estimate)
        try:
            response = provider.complete(
                model=model, messages=messages, tools=tools, max_tokens=max_tokens
            )
        except BaseException:
            self.release(reservation_id)
            raise
        cost, known = charge_for(getattr(response, "usage", None), estimate, price)
        if not known:
            self.calls_usage_unknown += 1
        self.settle(reservation_id, fp, cost)
        return response

    def close(self) -> None:
        self.conn.close()
