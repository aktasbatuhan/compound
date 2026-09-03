"""Discover which OpenRouter upstreams serve a model, as paste-ready tokens.

The whole point of a provider sweep is picking hosts, but the routable upstream
slugs are not something anyone knows by heart: they live in OpenRouter's
``/models/<model>/endpoints`` response as the ``tag`` field (e.g. ``deepinfra/fp4``,
``baseten/fp8``, ``digitalocean``). This module turns that response into the exact
``openrouter/<tag>`` tokens :mod:`compound.providers_registry` consumes, so the
"pick supported providers" step is one command instead of hand-spelunking an API.

The parser (:func:`parse_endpoints`) is pure and takes the decoded payload, so it
is testable without a network call; :func:`fetch_endpoints` wraps it with the
HTTP GET (auth header sent when ``OPENROUTER_API_KEY`` is set — the route is
public, but authenticating avoids anonymous rate limits).
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One serving host for a model, as OpenRouter reports it."""

    tag: str  # routable slug for provider.only, e.g. "deepinfra/fp4"
    provider_name: str
    quantization: str | None
    context_length: int | None
    prompt_usd_per_m: float | None  # input price, USD per 1M tokens
    completion_usd_per_m: float | None  # output price, USD per 1M tokens
    up: bool  # OpenRouter status >= 0 (a negative status means deranked/disabled)

    @property
    def token(self) -> str:
        """The ``--providers`` token that pins this exact host."""
        return f"openrouter/{self.tag}"


def _per_million(price: object) -> float | None:
    """OpenRouter prices are USD per token as strings; convert to USD per 1M."""
    try:
        return float(price) * 1_000_000
    except (TypeError, ValueError):
        return None


def parse_endpoints(payload: dict) -> list[Endpoint]:
    """Decode OpenRouter's ``/endpoints`` payload into :class:`Endpoint` rows.

    Order is preserved (OpenRouter returns endpoints in its own ranking).
    Endpoints without a ``tag`` cannot be pinned, so they are skipped.
    """
    data = payload.get("data") or {}
    out: list[Endpoint] = []
    for e in data.get("endpoints", []):
        tag = e.get("tag")
        if not tag:
            continue
        pricing = e.get("pricing") or {}
        quant = e.get("quantization")
        out.append(
            Endpoint(
                tag=tag,
                provider_name=e.get("provider_name") or e.get("name") or tag,
                quantization=None if quant in (None, "unknown") else quant,
                context_length=e.get("context_length"),
                prompt_usd_per_m=_per_million(pricing.get("prompt")),
                completion_usd_per_m=_per_million(pricing.get("completion")),
                up=e.get("status", 0) >= 0,
            )
        )
    return out


def fetch_endpoints(
    model: str,
    *,
    opener: Callable[..., object] = urllib.request.urlopen,
    timeout: float = 15.0,
) -> list[Endpoint]:
    """Fetch and parse the serving hosts for ``model`` from OpenRouter.

    ``opener`` is injectable so the fetch can be exercised without real HTTP.
    """
    req = urllib.request.Request(ENDPOINTS_URL.format(model=model))
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with opener(req, timeout=timeout) as resp:  # type: ignore[call-arg]
        payload = json.loads(resp.read())
    return parse_endpoints(payload)


def _fmt_price(p: float | None) -> str:
    return f"${p:.3f}" if p is not None else "-"


def _fmt_context(n: int | None) -> str:
    if not n:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1000}k"
    return str(n)


def format_table(endpoints: list[Endpoint]) -> str:
    """A paste-ready table: token, quant, context, in/out price, and up/down."""
    if not endpoints:
        return "no OpenRouter endpoints found for this model"
    tok_w = max(len("PROVIDER TOKEN"), max(len(e.token) for e in endpoints))
    header = (
        f"{'PROVIDER TOKEN':<{tok_w}}  {'QUANT':<7} {'CONTEXT':>7} "
        f"{'$IN/M':>7} {'$OUT/M':>7}  STATUS"
    )
    lines = [header]
    for e in endpoints:
        lines.append(
            f"{e.token:<{tok_w}}  {(e.quantization or '?'):<7} "
            f"{_fmt_context(e.context_length):>7} "
            f"{_fmt_price(e.prompt_usd_per_m):>7} "
            f"{_fmt_price(e.completion_usd_per_m):>7}  "
            f"{'up' if e.up else 'down'}"
        )
    up = [e.token for e in endpoints if e.up]
    lines.append("")
    lines.append(f"# {len(endpoints)} endpoint(s), {len(up)} up")
    lines.append("# sweep the ones that are up:")
    lines.append("#   --providers " + ",".join(up))
    return "\n".join(lines)


#: A probe asks for almost nothing: the question is whether the host answers at
#: all, not what it says.
PROBE_MAX_TOKENS = 8
PROBE_TIMEOUT_S = 90.0


def probe_endpoint(
    tag: str,
    model: str,
    api_key: str,
    *,
    timeout: float = PROBE_TIMEOUT_S,
) -> tuple[int | str, float, str]:
    """Send one tiny pinned call and report what the host actually did.

    OpenRouter's own ``status`` field says whether it believes an endpoint is
    up, which is not the same claim as "will serve me right now": without our
    own upstream key we sit on OpenRouter's shared rate-limit pool for that
    upstream, and a host listed as up can return 429 on every call for one
    model while serving another fine. Only a real call distinguishes the two.

    Returns ``(status, seconds, detail)`` where status is an HTTP code or
    ``"ERR"``, and detail is the served provider on success or the start of the
    error body otherwise.
    """
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": PROBE_MAX_TOKENS,
        "provider": {"only": [tag.split("/")[0]], "allow_fallbacks": False},
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    started = _time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = _json.load(response)
            return response.status, _time.monotonic() - started, str(payload.get("provider") or "")
    except urllib.error.HTTPError as exc:
        return exc.code, _time.monotonic() - started, exc.read()[:160].decode(errors="replace")
    except Exception as exc:  # network, timeout, malformed body
        return "ERR", _time.monotonic() - started, str(exc)[:160]


def probe_endpoints(
    endpoints: list[Endpoint],
    model: str,
    api_key: str,
    *,
    workers: int = 8,
) -> dict[str, tuple[int | str, float, str]]:
    """Probe every endpoint concurrently, keyed by tag."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda e: (e.tag, probe_endpoint(e.tag, model, api_key)), endpoints)
        return dict(results)
