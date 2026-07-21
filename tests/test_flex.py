from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from compound.budget import BudgetLedger
from compound.costs import TokenPrices
from compound.flex import (
    CachedFlexRunner,
    FlexRequest,
    flex_fingerprint,
    responses_output_text,
    responses_usage,
)
from compound.providers import BackgroundResponseSnapshot


def _completed(response_id: str = "resp_1") -> BackgroundResponseSnapshot:
    return BackgroundResponseSnapshot(
        response_id=response_id,
        status="completed",
        request_latency_ms=8,
        output={
            "id": response_id,
            "status": "completed",
            "model": "resolved-model",
            "service_tier": "flex",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "result = 42"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 7},
            },
        },
    )


@dataclass
class FakeProvider:
    name: str = "doubleword"
    response_id: str = "resp_1"
    retrievals: list[BackgroundResponseSnapshot | BaseException] = field(default_factory=list)
    submissions: int = 0
    retrieves: int = 0
    telemetry: list[dict[str, Any]] = field(default_factory=list)

    def submit_background_response(self, **_: Any) -> BackgroundResponseSnapshot:
        self.submissions += 1
        return BackgroundResponseSnapshot(
            response_id=self.response_id,
            status="queued",
            output={"id": self.response_id, "status": "queued"},
            request_latency_ms=4,
        )

    def retrieve_background_response(self, response_id: str) -> BackgroundResponseSnapshot:
        self.retrieves += 1
        item = self.retrievals.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert item.response_id == response_id
        return item

    def record_telemetry(self, record: dict[str, Any]) -> None:
        self.telemetry.append(record)


def _runner(tmp_path, provider: FakeProvider) -> CachedFlexRunner:
    return CachedFlexRunner(
        provider=provider,  # type: ignore[arg-type]
        prices_by_model={"candidate": TokenPrices(1.0, 2.0)},
        ledger=BudgetLedger.load(tmp_path / "budget.json", 25.0),
        cache_dir=tmp_path / "cache",
        poll_interval_seconds=0,
        timeout_seconds=10,
    )


def test_responses_text_and_usage_parsing() -> None:
    raw = _completed().output
    usage = responses_usage(raw)

    assert responses_output_text(raw) == "result = 42"
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.reasoning_tokens == 7
    assert usage.cached_tokens == 3


def test_flex_fingerprint_separates_reasoning_and_trials() -> None:
    base = FlexRequest(model="candidate", input="solve", reasoning_effort="low")
    repeated = FlexRequest(
        model="candidate",
        input="solve",
        reasoning_effort="low",
        cache_discriminator="trial:1",
    )
    higher_effort = FlexRequest(model="candidate", input="solve", reasoning_effort="high")

    assert flex_fingerprint(base) != flex_fingerprint(repeated)
    assert flex_fingerprint(base) != flex_fingerprint(higher_effort)


def test_flex_runner_caches_terminal_result_and_telemetry(tmp_path) -> None:
    provider = FakeProvider(retrievals=[_completed()])
    request = FlexRequest(model="candidate", input="solve", max_output_tokens=100)

    result = _runner(tmp_path, provider).run_many([request])[0]

    assert result["status"] == "completed"
    assert result["output_text"] == "result = 42"
    assert result["billing_tier_confirmed_from_response"] is True
    assert result["estimated_cost_usd"] == pytest.approx(0.000047)
    assert provider.submissions == 1
    assert provider.retrieves == 1
    assert provider.telemetry[0]["api_surface"] == "responses"
    assert provider.telemetry[0]["requested_service_tier"] == "flex"

    cached_provider = FakeProvider()
    cached = _runner(tmp_path, cached_provider).run_many([request])[0]

    assert cached == result
    assert cached_provider.submissions == 0
    assert cached_provider.retrieves == 0


def test_flex_runner_resumes_response_id_after_poll_failure(tmp_path) -> None:
    request = FlexRequest(model="candidate", input="solve")
    interrupted = FakeProvider(retrievals=[RuntimeError("temporary poll failure")])

    with pytest.raises(RuntimeError, match="temporary poll failure"):
        _runner(tmp_path, interrupted).run_many([request])
    assert interrupted.submissions == 1

    resumed = FakeProvider(retrievals=[_completed()])
    result = _runner(tmp_path, resumed).run_many([request])[0]

    assert result["response_id"] == "resp_1"
    assert resumed.submissions == 0
    assert resumed.retrieves == 1
