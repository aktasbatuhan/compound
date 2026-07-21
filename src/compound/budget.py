from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class BudgetLedger:
    hard_limit_usd: float
    path: Path
    spent_usd: float = 0.0
    entries: list[dict[str, float | str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, hard_limit_usd: float) -> BudgetLedger:
        ledger_path = Path(path)
        spent = 0.0
        entries: list[dict[str, float | str]] = []
        if ledger_path.exists():
            payload = json.loads(ledger_path.read_text())
            spent = float(payload.get("spent_usd", 0.0))
            entries = payload.get("entries", [])
        return cls(
            hard_limit_usd=hard_limit_usd,
            path=ledger_path,
            spent_usd=spent,
            entries=entries,
        )

    @property
    def remaining_usd(self) -> float:
        return self.hard_limit_usd - self.spent_usd

    def require_headroom(self, estimated_max_usd: float) -> None:
        if estimated_max_usd < 0:
            raise ValueError("estimated cost cannot be negative")
        if self.spent_usd + estimated_max_usd > self.hard_limit_usd:
            raise BudgetExceededError(
                f"estimated call would exceed ${self.hard_limit_usd:.2f} hard limit"
            )

    def record(self, cost_usd: float, *, label: str) -> None:
        if cost_usd < 0:
            raise ValueError("cost cannot be negative")
        existing = next((entry for entry in self.entries if entry["label"] == label), None)
        if existing is not None:
            if abs(float(existing["cost_usd"]) - cost_usd) > 1e-9:
                raise ValueError(f"budget label already recorded with a different cost: {label}")
            return
        if self.spent_usd + cost_usd > self.hard_limit_usd:
            raise BudgetExceededError(f"recorded cost would exceed hard limit: {label}")
        self.spent_usd += cost_usd
        self.entries.append({"label": label, "cost_usd": cost_usd})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hard_limit_usd": self.hard_limit_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_usd,
            "last_label": label,
            "entries": self.entries,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
