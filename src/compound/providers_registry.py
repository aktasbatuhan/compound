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
    direct/<name>           any OpenAI-compatible host declared in compound.yaml
                            ``providers.<name>`` (base_url + api_key_env).

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from compound.adapters.tau import TauModel

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DOUBLEWORD_BASE = "https://api.doubleword.ai/v1"

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

    @property
    def label(self) -> str:
        """Short, stable identity for tables and output directories."""
        if self.kind == "openrouter":
            return self.upstream or "openrouter"
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
        """
        body: dict[str, Any] = {}
        if self.kind == "openrouter" and self.upstream:
            body["provider"] = {"only": [self.upstream], "allow_fallbacks": False}
        if self.service_tier:
            body["service_tier"] = self.service_tier
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
            raise ValueError("openrouter/<upstream> needs an upstream slug")
        base, key = _KNOWN["openrouter"]
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
        return ProviderSpec(
            token=token,
            kind="direct",
            base_url=entry["base_url"],
            api_key_env=entry["api_key_env"],
            service_tier=entry.get("service_tier"),
            name=target,
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
