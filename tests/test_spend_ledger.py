"""The Python half of the paid-call contract must behave like the TypeScript half."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import pytest

from compound.spend_ledger import (
    RESERVATION_TTL_MS,
    BudgetExceededError,
    SpendLedger,
    TokenPrice,
    charge_for,
    estimate_cost,
)

# Mirrors packages/storage/drizzle: the TypeScript migrations own this schema.
SCHEMA = """
create table spend_records (
  id text primary key not null,
  experiment_id text,
  fingerprint text not null,
  cost_usd real not null,
  created_at integer default (unixepoch() * 1000) not null
);
create table spend_reservations (
  id text primary key not null,
  experiment_id text,
  fingerprint text not null,
  reserved_usd real not null,
  created_at integer default (unixepoch() * 1000) not null
);
"""

PRICE = TokenPrice(input=1.0, output=2.0)


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class _Response:
    output: dict
    usage: _Usage | None


_DEFAULT_USAGE = _Usage(1000, 500)


class _Provider:
    def __init__(self, usage: _Usage | None = _DEFAULT_USAGE, fail: bool = False) -> None:
        self.usage = usage
        self.fail = fail
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return _Response(output={"role": "assistant", "content": "ok"}, usage=self.usage)


def _db(tmp_path):
    path = tmp_path / "compound.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.close()
    return str(path)


def _ledger(path, cap=1.0, limit=1.0):
    return SpendLedger(path, experiment_id="opt:test", cap_usd=cap, global_hard_limit_usd=limit)


def test_refuses_a_database_without_the_ledger_tables(tmp_path):
    path = tmp_path / "bare.db"
    sqlite3.connect(path).close()
    with pytest.raises(RuntimeError, match="lacks"):
        _ledger(str(path))


def test_paid_call_reserves_then_settles_at_actual_charge(tmp_path):
    ledger = _ledger(_db(tmp_path))
    provider = _Provider(_Usage(input_tokens=1000, output_tokens=500))
    ledger.paid_call(provider, model="m", messages=[{"role": "user", "content": "hi"}],
                     tools=None, max_tokens=100, price=PRICE)
    assert provider.calls == 1
    assert ledger.calls == 1
    assert ledger.total_charged_usd == pytest.approx(1000 / 1e6 * 1.0 + 500 / 1e6 * 2.0)
    conn = sqlite3.connect(ledger.conn.execute("pragma database_list").fetchone()[2])
    assert conn.execute("select count(*) from spend_reservations").fetchone()[0] == 0
    rows = conn.execute("select experiment_id, cost_usd from spend_records").fetchall()
    assert rows == [("opt:test", pytest.approx(0.002))]


def test_reservation_blocks_a_call_the_cap_cannot_afford(tmp_path):
    path = _db(tmp_path)
    ledger = _ledger(path, cap=0.0005, limit=10.0)
    provider = _Provider()
    with pytest.raises(BudgetExceededError) as err:
        ledger.paid_call(provider, model="m", messages=[{"role": "user", "content": "x" * 4000}],
                         tools=None, max_tokens=100, price=PRICE)
    assert err.value.scope == "experiment"
    assert provider.calls == 0, "refused before the provider was called"


def test_other_processes_reservations_count_against_the_global_limit(tmp_path):
    path = _db(tmp_path)
    other = SpendLedger(path, experiment_id="other", cap_usd=5.0, global_hard_limit_usd=1.0)
    other.reserve("fp-other", 0.9)
    mine = _ledger(path, cap=5.0, limit=1.0)
    with pytest.raises(BudgetExceededError) as err:
        mine.reserve("fp-mine", 0.2)
    assert err.value.scope == "global"


def test_provider_failure_releases_the_reservation(tmp_path):
    ledger = _ledger(_db(tmp_path))
    with pytest.raises(RuntimeError, match="provider down"):
        ledger.paid_call(_Provider(fail=True), model="m", messages=[{"role": "user", "content": "hi"}],
                         tools=None, max_tokens=100, price=PRICE)
    assert ledger._open() == 0
    assert ledger.total_charged_usd == 0


def test_missing_usage_is_charged_at_the_estimate_not_zero(tmp_path):
    ledger = _ledger(_db(tmp_path))
    ledger.paid_call(_Provider(usage=None), model="m", messages=[{"role": "user", "content": "hi"}],
                     tools=None, max_tokens=100, price=PRICE)
    assert ledger.total_charged_usd > 0
    assert ledger.calls_usage_unknown == 1


def test_stale_reservations_are_ignored(tmp_path):
    path = _db(tmp_path)
    ledger = _ledger(path)
    ledger.reserve("fp", 0.9)
    stale = int(time.time() * 1000) - RESERVATION_TTL_MS - 1000
    ledger.conn.execute("update spend_reservations set created_at = ?", (stale,))
    ledger.reserve("fp2", 0.9)


def test_estimate_counts_tools_and_full_output_budget():
    msgs = [{"role": "user", "content": "a" * 400}]
    without = estimate_cost(msgs, None, 100, PRICE)
    with_tools = estimate_cost(msgs, [{"type": "function", "function": {"name": "t" * 400}}], 100, PRICE)
    assert with_tools > without
    assert charge_for(None, 0.5, PRICE) == (0.5, False)
    assert charge_for(_Usage(0, 0), 0.5, PRICE) == (0.5, False)
