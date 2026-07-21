from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

from compound.budget import BudgetLedger
from compound.contracts import Usage
from compound.costs import TokenPrices
from compound.ds1000_optimizer import (
    CachedDS1000Adapter,
    DS1000Example,
    MeteredReflectionLM,
    response_text,
)
from compound.providers import ProviderResponse


def _response(text: str) -> ProviderResponse:
    return ProviderResponse(
        output={"choices": [{"message": {"content": text}}]},
        usage=Usage(input_tokens=100, output_tokens=20),
        latency_ms=10,
    )


def test_response_text_supports_text_parts() -> None:
    payload = {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]}
    assert response_text(payload) == "ok"


def test_adapter_uses_gepa_default_proposer() -> None:
    assert CachedDS1000Adapter.propose_new_texts is None


@patch("compound.ds1000_optimizer.subprocess.run")
def test_adapter_caches_model_and_grader_result(run: Mock, tmp_path: Path) -> None:
    run.return_value = Mock(
        returncode=0,
        stdout=json.dumps({"passed": True, "feedback": "passed"}),
        stderr="",
    )
    provider = Mock()
    provider.complete.return_value = _response("answer = x + 1")
    ledger = BudgetLedger.load(tmp_path / "budget.json", 25)
    adapter = CachedDS1000Adapter(
        provider=provider,
        model="candidate",
        prices=TokenPrices(1, 2),
        ledger=ledger,
        cache_dir=tmp_path / "cache",
        docker_image="test-image",
    )
    example = DS1000Example("ds1000_1", "task", "def test_execution(code): pass", {})

    first = adapter.evaluate([example], {"system_prompt": "solve"}, capture_traces=True)
    second = adapter.evaluate([example], {"system_prompt": "solve"}, capture_traces=True)

    assert first.scores == [1.0]
    assert second.scores == [1.0]
    provider.complete.assert_called_once()
    run.assert_called_once()
    assert len(ledger.entries) == 1


def test_reflection_lm_is_metered_and_cached(tmp_path: Path) -> None:
    provider = Mock()
    provider.complete.return_value = _response("new prompt")
    ledger = BudgetLedger.load(tmp_path / "budget.json", 25)
    reflection = MeteredReflectionLM(
        provider=provider,
        model="teacher",
        prices=TokenPrices(1, 2),
        ledger=ledger,
        cache_dir=tmp_path / "reflection",
    )

    assert reflection("reflect") == "new prompt"
    assert reflection("reflect") == "new prompt"
    provider.complete.assert_called_once()
    assert len(ledger.entries) == 1
