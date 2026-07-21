from __future__ import annotations

from dataclasses import dataclass

from compound.contracts import Usage


@dataclass(frozen=True, slots=True)
class TokenPrices:
    input_per_million: float
    output_per_million: float


def estimate_cost(usage: Usage, prices: TokenPrices) -> float:
    uncached_input = max(0, usage.input_tokens - usage.cached_tokens)
    input_cost = uncached_input * prices.input_per_million / 1_000_000
    output_cost = usage.output_tokens * prices.output_per_million / 1_000_000
    return input_cost + output_cost

