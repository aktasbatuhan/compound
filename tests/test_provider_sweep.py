from __future__ import annotations

from compound.provider_sweep import plan
from compound.providers_registry import parse_providers


def test_plan_lists_one_line_per_host_with_pinning():
    specs = parse_providers("openrouter/deepinfra,doubleword/flex,doubleword/realtime")
    lines = plan(specs, "deepseek/deepseek-v4-flash-0731")
    assert len(lines) == 3
    assert "deepinfra" in lines[0] and "allow_fallbacks" in lines[0]
    assert "service_tier" in lines[1]  # flex
    assert "host default" in lines[2]  # realtime, no injection
    assert all("deepseek/deepseek-v4-flash-0731" in ln for ln in lines)
