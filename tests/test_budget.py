import pytest

from compound.budget import BudgetExceededError, BudgetLedger


def test_budget_ledger_persists_and_enforces_limit(tmp_path) -> None:
    path = tmp_path / "budget.json"
    ledger = BudgetLedger.load(path, 1.0)
    ledger.record(0.4, label="first")
    restored = BudgetLedger.load(path, 1.0)
    assert restored.remaining_usd == pytest.approx(0.6)
    restored.record(0.4, label="first")
    assert restored.remaining_usd == pytest.approx(0.6)
    with pytest.raises(BudgetExceededError):
        restored.require_headroom(0.61)
