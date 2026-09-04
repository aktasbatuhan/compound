from __future__ import annotations

import io

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


def test_apply_host_models_ignores_a_kind_this_arm_does_not_use():
    """One grid fans the same mapping to every arm; only some arms are Doubleword."""
    from compound.providers_registry import apply_host_models, parse_providers

    specs = parse_providers("openrouter/deepinfra")
    out = apply_host_models(specs, {"doubleword": "zai-org/GLM-5.3-Flash"})
    assert out[0].wire_model is None


def test_apply_host_models_rejects_a_misspelled_kind():
    import pytest

    from compound.providers_registry import apply_host_models, parse_providers

    with pytest.raises(ValueError, match="no known provider"):
        apply_host_models(parse_providers("openrouter/auto"), {"doubelword": "x"})


def test_probe_endpoint_reports_a_rate_limited_host_as_not_answering(monkeypatch):
    """OpenRouter's "up" is its own belief; only a real call settles it."""
    import urllib.error

    from compound import openrouter_discovery as disc

    def raise_429(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {}, io.BytesIO(b'{"error":"rate-limited upstream"}')
        )

    monkeypatch.setattr(disc.urllib.request, "urlopen", raise_429, raising=False)
    status, seconds, detail = disc.probe_endpoint("deepinfra/fp8", "z-ai/glm-5.3-flash", "k")
    assert status == 429
    assert seconds >= 0
    assert "rate-limited" in detail


def test_apply_host_models_accepts_a_configured_direct_host_this_arm_lacks():
    # A mixed grid fans one mapping across every arm, so `zai` is the target on
    # the z.ai arm and unused on all the others. Without known_names the typo
    # guard would reject it and kill those arms, which is how six arms died on
    # the 2026-09-03 grid via the `doubleword` key.
    from compound.providers_registry import apply_host_models, parse_providers

    config = {"zai": {"base_url": "https://api.z.ai/api/paas/v4", "api_key_env": "ZAI_API_KEY"}}
    specs = parse_providers("openrouter/novita", providers_config=config)
    out = apply_host_models(specs, {"zai": "glm-5.3-flash"}, known_names=config)
    assert out[0].wire_model is None  # no-op on an arm that is not z.ai

    zai = parse_providers("direct/zai", providers_config=config)
    out = apply_host_models(zai, {"zai": "glm-5.3-flash"}, known_names=config)
    assert out[0].wire_model == "glm-5.3-flash"


def test_apply_host_models_still_rejects_a_typo_among_configured_names():
    from compound.providers_registry import apply_host_models, parse_providers

    config = {"zai": {"base_url": "https://x", "api_key_env": "K"}}
    with pytest.raises(ValueError, match="no known provider"):
        apply_host_models(
            parse_providers("openrouter/auto"), {"zia": "x"}, known_names=config
        )


# --- wire dialect and per-host timeout ---------------------------------------

ANTHROPIC_CFG = {
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "type": "anthropic",
        "cache_strategy": "explicit_marker",
    },
    "openai-flex": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "type": "openai_compatible",
        "service_tier": "flex",
        "timeout_s": 900,
        "max_tokens_field": "max_completion_tokens",
    },
}


def test_proxy_renames_the_output_cap_for_hosts_that_need_it():
    from compound.orproxy import inject

    spec = parse_provider("direct/openai-flex", providers_config=ANTHROPIC_CFG)
    out = inject({"model": "m", "max_tokens": 50, "messages": []}, spec)
    assert out["max_completion_tokens"] == 50 and "max_tokens" not in out
    plain = parse_provider("openrouter/deepinfra")
    assert inject({"max_tokens": 50, "messages": []}, plain)["max_tokens"] == 50


def test_direct_type_anthropic_selects_the_messages_dialect():
    spec = parse_provider("direct/anthropic", providers_config=ANTHROPIC_CFG)
    assert spec.dialect == "anthropic"
    assert spec.cache_strategy == "explicit_marker"
    assert spec.timeout_s is None
    # litellm speaks the Messages API natively; no api_base, no key override
    tau = spec.to_tau_model("claude-sonnet-5")
    assert tau.litellm_name() == "anthropic/claude-sonnet-5"
    assert tau.resolve_api_key_env() is None


def test_openai_compatible_hosts_default_to_chat_completions():
    assert parse_provider("openrouter/deepinfra").dialect == "openai"
    assert parse_provider("doubleword/flex").dialect == "openai"
    spec = parse_provider("direct/openai-flex", providers_config=ANTHROPIC_CFG)
    assert spec.dialect == "openai"
    assert spec.service_tier == "flex"
    assert spec.timeout_s == 900


def test_proxy_refuses_a_messages_dialect_host():
    from compound.orproxy import serve_provider

    spec = parse_provider("direct/anthropic", providers_config=ANTHROPIC_CFG)
    with pytest.raises(RuntimeError, match="chat completions only"):
        with serve_provider(spec):
            pass
