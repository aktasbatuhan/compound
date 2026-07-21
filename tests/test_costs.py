from compound.contracts import Usage
from compound.costs import TokenPrices, estimate_cost


def test_cost_estimate_excludes_cached_input_tokens() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000, cached_tokens=200_000)
    prices = TokenPrices(input_per_million=5.0, output_per_million=30.0)
    assert estimate_cost(usage, prices) == 7.0

