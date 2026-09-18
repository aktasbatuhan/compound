import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pricing_probe', Path('scripts/probe_dw_pricing.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_precise_cache_read_rates_match_observed_meter_within_rounding():
    usage = {'prompt_tokens': 11255, 'completion_tokens': 32,
             'cache_read_input_tokens': 11255, 'cache_creation_input_tokens': 0}
    assert probe.estimate(usage, 'priority') == pytest.approx(.00005297, abs=1e-8)
    usage.update(prompt_tokens=11257, completion_tokens=39, cache_read_input_tokens=11257)
    assert probe.estimate(usage, 'flex') == pytest.approx(.00004574, abs=1e-8)


def test_invalid_usage_never_becomes_a_price():
    with pytest.raises(ValueError):
        probe.estimate({'prompt_tokens': 10, 'completion_tokens': 2,
                        'cache_read_input_tokens': 11}, 'flex')
