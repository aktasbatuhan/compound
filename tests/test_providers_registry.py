from __future__ import annotations

import pytest

from compound.providers_registry import (
    DOUBLEWORD_BASE,
    OPENROUTER_BASE,
    parse_provider,
    parse_providers,
)


def test_openrouter_token_pins_upstream():
    spec = parse_provider("openrouter/deepinfra")
    assert spec.kind == "openrouter"
    assert spec.upstream == "deepinfra"
    assert spec.base_url == OPENROUTER_BASE
    assert spec.required_key_env() == "OPENROUTER_API_KEY"
    assert spec.label == "deepinfra"
    assert spec.proxy_injection() == {
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False, "require_parameters": True}
    }


def test_openrouter_upstream_may_carry_quant_tag():
    spec = parse_provider("openrouter/baseten/fp8")
    assert spec.upstream == "baseten/fp8"  # label keeps the quant tag
    # provider.only pins the base provider slug (the quant-tagged id 404s)
    assert spec.proxy_injection()["provider"]["only"] == ["baseten"]


def test_doubleword_realtime_has_no_service_tier():
    spec = parse_provider("doubleword/realtime")
    assert spec.kind == "doubleword"
    assert spec.service_tier is None
    assert spec.base_url == DOUBLEWORD_BASE
    assert spec.label == "doubleword-realtime"
    assert spec.proxy_injection() == {}


def test_doubleword_flex_forwards_service_tier():
    spec = parse_provider("doubleword/flex")
    assert spec.service_tier == "flex"
    assert spec.label == "doubleword-flex"
    assert spec.proxy_injection() == {"service_tier": "flex"}


def test_doubleword_defaults_to_realtime_when_bare():
    assert parse_provider("doubleword/").service_tier is None


def test_reasoning_pin_injects_each_hosts_dialect(monkeypatch):
    monkeypatch.setenv("COMPOUND_REASONING", "on")
    assert parse_provider("openrouter/deepinfra").proxy_injection()["reasoning"] == {"enabled": True}
    assert parse_provider("doubleword/realtime").proxy_injection() == {"reasoning_effort": "medium"}
    assert parse_provider("doubleword/flex").proxy_injection() == {
        "service_tier": "flex",
        "reasoning_effort": "medium",
    }
    monkeypatch.setenv("COMPOUND_REASONING", "off")
    assert parse_provider("openrouter/deepinfra").proxy_injection()["reasoning"] == {"enabled": False}
    assert parse_provider("doubleword/realtime").proxy_injection() == {"reasoning_effort": "none"}


def test_openrouter_auto_is_unpinned(monkeypatch):
    spec = parse_provider("openrouter/auto")
    assert spec.kind == "openrouter"
    assert spec.upstream is None
    assert spec.label == "openrouter-auto"
    monkeypatch.delenv("COMPOUND_REASONING", raising=False)
    assert spec.proxy_injection() == {}  # no provider block: OpenRouter routes freely
    monkeypatch.setenv("COMPOUND_REASONING", "on")
    assert spec.proxy_injection() == {"reasoning": {"enabled": True}}


def test_no_reasoning_pin_without_env(monkeypatch):
    monkeypatch.delenv("COMPOUND_REASONING", raising=False)
    assert "reasoning" not in parse_provider("openrouter/deepinfra").proxy_injection()
    assert parse_provider("doubleword/realtime").proxy_injection() == {}


def test_direct_token_reads_config_block():
    config = {"groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"}}
    spec = parse_provider("direct/groq", providers_config=config)
    assert spec.kind == "direct"
    assert spec.base_url == "https://api.groq.com/openai/v1"
    assert spec.required_key_env() == "GROQ_API_KEY"
    assert spec.label == "groq"
    assert spec.proxy_injection() == {}


def test_cache_strategy_defaults_per_kind():
    # openrouter majors cache implicitly; doubleword needs an explicit marker;
    # a direct host assumes no usable cache until it declares one.
    assert parse_provider("openrouter/deepinfra").cache_strategy == "implicit"
    assert parse_provider("openrouter/auto").cache_strategy == "implicit"
    assert parse_provider("doubleword/flex").cache_strategy == "explicit_marker"
    assert parse_provider("doubleword/realtime").cache_strategy == "explicit_marker"
    config = {"groq": {"base_url": "https://x/v1", "api_key_env": "GROQ_API_KEY"}}
    assert parse_provider("direct/groq", providers_config=config).cache_strategy == "none"


def test_cache_strategy_overridable_by_direct_config():
    config = {
        "vllm": {
            "base_url": "https://x/v1",
            "api_key_env": "VLLM_KEY",
            "cache_strategy": "explicit_marker",
        }
    }
    spec = parse_provider("direct/vllm", providers_config=config)
    assert spec.cache_strategy == "explicit_marker"


def test_to_tau_model_openrouter_injects_provider_only():
    spec = parse_provider("openrouter/deepinfra")
    model = spec.to_tau_model("deepseek/deepseek-v4-flash-0731", max_tokens=8192)
    args = model.llm_args()
    assert args["extra_body"]["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False, "require_parameters": True}
    assert args["max_tokens"] == 8192
    assert model.litellm_name() == "openrouter/deepseek/deepseek-v4-flash-0731"


def test_to_tau_model_doubleword_flex_matches_proxy_injection():
    spec = parse_provider("doubleword/flex")
    model = spec.to_tau_model("deepseek/deepseek-v4-flash-0731")
    args = model.llm_args()
    # The in-process tau path and the proxy path must inject the same thing.
    assert args["extra_body"]["service_tier"] == "flex"
    assert spec.proxy_injection()["service_tier"] == "flex"
    assert model.resolve_api_key_env() == "DOUBLEWORD_API_KEY"


def test_parse_providers_preserves_order_and_dedupes():
    specs = parse_providers("openrouter/deepinfra, openrouter/baseten, openrouter/deepinfra")
    assert [s.upstream for s in specs] == ["deepinfra", "baseten"]


@pytest.mark.parametrize("bad", ["deepinfra", "openrouter/", "", "unknown/x"])
def test_malformed_tokens_raise(bad):
    with pytest.raises(ValueError):
        parse_provider(bad)


def test_direct_without_config_block_raises():
    with pytest.raises(ValueError):
        parse_provider("direct/nope", providers_config={})


def test_apply_host_models_matches_kind_label_and_token():
    from compound.providers_registry import apply_host_models, parse_providers

    specs = parse_providers("openrouter/deepinfra,doubleword/realtime,doubleword/flex")
    out = apply_host_models(specs, {"doubleword": "zai-org/GLM-5.3-Flash", "doubleword/flex": "other"})
    assert out[0].wire_model is None
    assert out[1].wire_model == "zai-org/GLM-5.3-Flash"
    assert out[2].wire_model == "other"


def test_apply_host_models_rejects_unknown_key():
    import pytest

    from compound.providers_registry import apply_host_models, parse_providers

    with pytest.raises(ValueError):
        apply_host_models(parse_providers("openrouter/auto"), {"fireworks": "x"})
