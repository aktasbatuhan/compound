"""Named provider tokens: one grammar for picking a serving host.

A *provider token* names where a model is served, independent of which model it
is. It is the unit a user picks on the CLI (``--providers a,b,c``) and the unit a
sweep iterates. Three forms, all model-agnostic:

    openrouter/<upstream>   pin one OpenRouter upstream (fallbacks disabled).
                            <upstream> is the routable slug, optionally with a
                            quant tag: ``openrouter/deepinfra``,
                            ``openrouter/baseten/fp8``.
    doubleword/<tier>       Doubleword, addressed directly. tier is ``realtime``
                            or ``flex`` (flex is forwarded as service_tier).
    direct/<name>           any host declared in compound.yaml ``providers.<name>``
                            (base_url + api_key_env). ``type: anthropic`` marks a
                            host that speaks the Messages API instead of chat
                            completions; everything else is OpenAI-compatible.

The same ``ProviderSpec`` drives two consumers:

* in-process benchmarks (tau2, mmlu) via :meth:`ProviderSpec.to_tau_model`, which
  injects OpenRouter ``provider.only`` / the Doubleword service tier itself; and
* external harnesses (terminal-bench) via :mod:`compound.orproxy`, which reads
  :attr:`forward_base_url`, :meth:`required_key_env`, and
  :meth:`proxy_injection` to reproduce the exact same pinning behind a localhost
  endpoint the harness points at.

Because the pinning lives in the token, "same model, many hosts" is one list on
the CLI, and every benchmark inherits it the same way.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from compound.adapters.tau import TauModel

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DOUBLEWORD_BASE = "https://api.doubleword.ai/v1"


def openrouter_only(upstream: str) -> str:
    """The value OpenRouter's ``provider.only`` expects: the base provider slug.

    Our upstream tokens carry the endpoint tag from ``/endpoints`` (e.g.
    ``deepinfra/fp4``, ``baseten/fp8``), but ``provider.only`` matches on the
    *provider* slug (``deepinfra``), not the quant-tagged endpoint id — passing
    the full tag 404s ("no allowed providers"). The quant suffix is kept only for
    labelling; here we take the provider slug before the first slash.
    """
    return upstream.split("/", 1)[0]


def openrouter_provider_block(upstream: str) -> dict[str, Any]:
    """The OpenRouter routing block every pinned request carries.

    ``require_parameters`` makes OpenRouter refuse to route to the pinned host
    when it cannot honor a parameter the request actually uses, so a capability
    gap fails fast as "no allowed providers" instead of burning a whole run:
    novita accepted 42 terminal-bench episodes it could never serve because the
    agent's ``response_format: json_schema`` isn't implemented on its endpoint.
    """
    return {
        "only": [openrouter_only(upstream)],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


# The two credentials the project ships with; a direct/<name> token may name any
# other env var through compound.yaml.
_KNOWN = {
    "openrouter": (OPENROUTER_BASE, "OPENROUTER_API_KEY"),
    "doubleword": (DOUBLEWORD_BASE, "DOUBLEWORD_API_KEY"),
}


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """A resolved serving host: where to send, how to authenticate, how to pin."""

    token: str
    kind: str  # "openrouter" | "doubleword" | "direct"
    base_url: str
    api_key_env: str
    #: OpenRouter upstream slug to pin (openrouter tokens only).
    upstream: str | None = None
    #: Service tier forwarded in the request body (e.g. Doubleword "flex").
    service_tier: str | None = None
    #: Human/config name for direct hosts, used only for labelling.
    name: str | None = None
    #: How this host serves its prompt cache, which decides whether the proxy
    #: must inject anything to get a hit. One of:
    #:   ``"implicit"``        host caches prompt prefixes on its own (OpenRouter
    #:                         majors); the proxy adds nothing.
    #:   ``"explicit_marker"`` host caches only when the request carries an
    #:                         Anthropic-style ``cache_control`` marker
    #:                         (Doubleword); the proxy injects one on opt-in.
    #:   ``"none"``            no usable prompt cache (default for direct hosts).
    #: Left ``None`` at construction, it is filled from :attr:`kind` in
    #: ``__post_init__``; a direct host may override it in compound.yaml.
    cache_strategy: str | None = None
    #: Model id to send on the wire when this host names the weights differently
    #: from the id the sweep was invoked with (Doubleword serves
    #: ``zai-org/GLM-5.3-Flash`` where OpenRouter serves ``z-ai/glm-5.3-flash``).
    #: ``None`` sends the caller's id unchanged.
    wire_model: str | None = None
    #: Wire protocol. ``"openai"`` is chat completions, which every host but one
    #: speaks. ``"anthropic"`` is the Messages API, used for Anthropic's own
    #: endpoint because its OpenAI-compatible layer drops what a comparison
    #: needs: no prompt caching, empty ``prompt_tokens_details``, ``service_tier``
    #: ignored, temperature capped at 1. Measuring Anthropic through that layer
    #: would score it at 0% cache the way an unmarked Doubleword call does.
    dialect: str = "openai"
    #: Per-call timeout this host needs, seconds. A queued tier (OpenAI flex)
    #: can legitimately wait longer than the harness default before the first
    #: byte, and a timeout that fires first records the host as failed when it
    #: was only slow. ``None`` uses the caller's default.
    timeout_s: int | None = None
    #: Name of the output-cap field this host accepts. OpenAI's current models
    #: reject ``max_tokens`` outright ("use max_completion_tokens instead"),
    #: while OpenRouter, Doubleword and most OpenAI-compatible servers take
    #: ``max_tokens``. Declared per host in compound.yaml; the wrong name is a
    #: 400 on every call, which a smoke run surfaced before a grid did.
    max_tokens_field: str = "max_tokens"

    def __post_init__(self) -> None:
        if self.cache_strategy is None:
            default = {"openrouter": "implicit", "doubleword": "explicit_marker"}.get(
                self.kind, "none"
            )
            object.__setattr__(self, "cache_strategy", default)

    @property
    def label(self) -> str:
        """Short, stable identity for tables and output directories."""
        if self.kind == "openrouter":
            return self.upstream or "openrouter-auto"
        if self.kind == "doubleword":
            return f"doubleword-{self.service_tier or 'realtime'}"
        return self.name or self.token

    @property
    def safe_label(self) -> str:
        """:attr:`label` with slashes flattened, for use as a directory name."""
        return self.label.replace("/", "-")

    def required_key_env(self) -> str:
        return self.api_key_env

    @property
    def forward_base_url(self) -> str:
        """Upstream the proxy forwards to (OpenRouter, Doubleword, or a direct host)."""
        return self.base_url

    def proxy_injection(self) -> dict[str, Any]:
        """Request-body fields the proxy must add to reproduce this host's pinning.

        Mirrors exactly what :meth:`to_tau_model` injects in-process, so a run
        through the proxy is identical to an in-process run of the same token.

        ``COMPOUND_REASONING=on|off`` additionally pins the model's reasoning
        mode in each host's own dialect. Without it, each host applies its own
        default — and hosts disagree: in the 2026-08 terminal-bench sweep the
        Doubleword deployments defaulted reasoning off while every OpenRouter
        upstream defaulted it on, which silently confounds any cross-host
        latency or throughput comparison.
        """
        body: dict[str, Any] = {}
        if self.kind == "openrouter" and self.upstream:
            body["provider"] = openrouter_provider_block(self.upstream)
        if self.service_tier:
            body["service_tier"] = self.service_tier
        pin = os.getenv("COMPOUND_REASONING", "").lower()
        if pin in ("on", "off"):
            if self.kind == "openrouter":
                body["reasoning"] = {"enabled": pin == "on"}
            else:
                # Doubleword rejects the ``reasoning`` block; its dialect is
                # OpenAI-style ``reasoning_effort``, where "none" disables.
                body["reasoning_effort"] = "medium" if pin == "on" else "none"
        return body

    def to_tau_model(self, model: str, **kwargs: Any) -> TauModel:
        """Build a :class:`TauModel` that serves ``model`` on this host."""
        from compound.adapters.tau import TauModel

        if self.kind == "openrouter":
            return TauModel(
                provider="openrouter",
                model=model,
                openrouter_provider=self.upstream,
                **kwargs,
            )
        if self.kind == "doubleword":
            return TauModel(
                provider="doubleword",
                model=model,
                api_base=self.base_url,
                api_key_env=self.api_key_env,
                service_tier=self.service_tier,
                **kwargs,
            )
        if self.dialect == "anthropic":
            # litellm speaks the Messages API natively under the ``anthropic/``
            # prefix and reads ANTHROPIC_API_KEY itself; no api_base needed.
            return TauModel(provider="anthropic", model=model, **kwargs)
        return TauModel(
            provider=self.name or "direct",
            model=model,
            api_base=self.base_url,
            api_key_env=self.api_key_env,
            service_tier=self.service_tier,
            **kwargs,
        )


def parse_provider(
    token: str, *, providers_config: dict[str, Any] | None = None
) -> ProviderSpec:
    """Resolve one provider token into a :class:`ProviderSpec`.

    ``providers_config`` is the ``providers`` block of compound.yaml, consulted
    only for ``direct/<name>`` tokens. Raises ``ValueError`` on a malformed or
    unknown token so a typo fails loudly before any spend.
    """
    token = token.strip()
    if "/" not in token:
        raise ValueError(
            f"provider token {token!r} must be kind/target, e.g. openrouter/deepinfra, "
            "doubleword/flex, or direct/<name>"
        )
    kind, target = token.split("/", 1)
    kind = kind.lower()

    if kind == "openrouter":
        if not target:
            raise ValueError("openrouter/<upstream> needs an upstream slug, or 'auto'")
        base, key = _KNOWN["openrouter"]
        if target.lower() == "auto":
            # Deliberately unpinned: no provider block at all, fallbacks allowed.
            # This is the control arm that measures what OpenRouter's default
            # routing actually does to cost, cache hits, and quality; the served
            # upstream still lands in every trace via the response's provider echo.
            return ProviderSpec(
                token=token, kind="openrouter", base_url=base, api_key_env=key, upstream=None
            )
        return ProviderSpec(
            token=token, kind="openrouter", base_url=base, api_key_env=key, upstream=target
        )

    if kind == "doubleword":
        tier = (target or "realtime").lower()
        if tier not in ("realtime", "flex"):
            raise ValueError(f"doubleword tier must be realtime or flex, got {tier!r}")
        base, key = _KNOWN["doubleword"]
        return ProviderSpec(
            token=token,
            kind="doubleword",
            base_url=base,
            api_key_env=key,
            service_tier=None if tier == "realtime" else "flex",
        )

    if kind == "direct":
        config = providers_config or {}
        if target not in config:
            raise ValueError(
                f"direct/{target}: no providers.{target} block in compound.yaml"
            )
        entry = config[target]
        dialect = "anthropic" if str(entry.get("type", "")).lower() == "anthropic" else "openai"
        timeout = entry.get("timeout_s")
        return ProviderSpec(
            token=token,
            kind="direct",
            base_url=entry["base_url"],
            api_key_env=entry["api_key_env"],
            service_tier=entry.get("service_tier"),
            name=target,
            # A direct host declares its own cache behavior; absent, it defaults
            # to "none" (no assumed prompt cache) in __post_init__.
            cache_strategy=entry.get("cache_strategy"),
            dialect=dialect,
            timeout_s=int(timeout) if timeout is not None else None,
            max_tokens_field=str(entry.get("max_tokens_field") or "max_tokens"),
        )

    raise ValueError(
        f"unknown provider kind {kind!r} in {token!r} "
        "(expected openrouter, doubleword, or direct)"
    )


def parse_providers(
    tokens: str, *, providers_config: dict[str, Any] | None = None
) -> list[ProviderSpec]:
    """Resolve a comma-separated provider list, preserving order, de-duplicating."""
    seen: set[str] = set()
    specs: list[ProviderSpec] = []
    for raw in tokens.split(","):
        raw = raw.strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        specs.append(parse_provider(raw, providers_config=providers_config))
    if not specs:
        raise ValueError("no provider tokens given")
    return specs


def apply_host_models(
    specs: list[ProviderSpec],
    mapping: dict[str, str],
    *,
    known_names: Iterable[str] | None = None,
) -> list[ProviderSpec]:
    """Return ``specs`` with :attr:`ProviderSpec.wire_model` set from ``mapping``.

    A mapping key matches a spec by exact token (``doubleword/flex``), by label
    (``doubleword-flex``), or by kind (``doubleword``), most specific first.

    A key naming a provider this arm happens not to use is a no-op, not an
    error: one grid fans the same mapping out to every arm, and only the
    Doubleword arms of that grid have a Doubleword provider. A key that names
    nothing at all is still a typo and raises, so a misspelling cannot silently
    leave an arm on the wrong model id.

    ``known_names`` are the direct-host names configured in ``compound.yaml``.
    They have to be passed in because a host like ``zai`` is a valid target on
    its own arm and unused on every other one, and without them the typo guard
    would reject the very mapping a mixed grid needs.
    """
    from dataclasses import replace

    valid = set(_KNOWN) | {str(n) for n in (known_names or ())}
    unused = set(mapping)
    out: list[ProviderSpec] = []
    for spec in specs:
        chosen = None
        for key in (spec.token, spec.label, spec.kind):
            if key in mapping:
                chosen = key
                break
        if chosen is None:
            out.append(spec)
            continue
        unused.discard(chosen)
        out.append(replace(spec, wire_model=mapping[chosen]))
    typos = sorted(k for k in unused if k not in valid and "/" not in k and "-" not in k)
    if typos:
        raise ValueError(
            f"--host-model keys name no known provider: {typos} "
            f"(known: {sorted(valid)})"
        )
    return out
