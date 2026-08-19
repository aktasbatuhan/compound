"""Provider-discovery parsing and formatting (no network)."""

from __future__ import annotations

import io
import json

from compound.openrouter_discovery import (
    Endpoint,
    fetch_endpoints,
    format_table,
    parse_endpoints,
)

# A trimmed but real-shaped /endpoints payload (prices are USD per token strings).
PAYLOAD = {
    "data": {
        "id": "deepseek/deepseek-v4-flash-0731",
        "endpoints": [
            {
                "name": "DeepInfra | deepseek/...",
                "provider_name": "DeepInfra",
                "tag": "deepinfra/fp4",
                "quantization": "fp4",
                "context_length": 1048576,
                "pricing": {"prompt": "0.00000009", "completion": "0.00000018"},
                "status": 0,
            },
            {
                "name": "DigitalOcean | deepseek/...",
                "provider_name": "DigitalOcean",
                "tag": "digitalocean",
                "quantization": "unknown",
                "context_length": 1048576,
                "pricing": {"prompt": "0.000000079996", "completion": "0.000000252"},
                "status": 0,
            },
            {
                "name": "StreamLake | deepseek/...",
                "provider_name": "StreamLake",
                "tag": "streamlake",
                "quantization": "unknown",
                "context_length": 1024000,
                "pricing": {"prompt": "0.000000084", "completion": "0.000000168"},
                "status": -2,
            },
            {
                # no routable tag -> cannot be pinned -> dropped
                "name": "Mystery | deepseek/...",
                "provider_name": "Mystery",
                "quantization": "fp8",
                "pricing": {},
            },
        ],
    }
}


def test_parse_endpoints_maps_fields_and_tokens():
    eps = parse_endpoints(PAYLOAD)
    # the tag-less endpoint is skipped
    assert [e.tag for e in eps] == ["deepinfra/fp4", "digitalocean", "streamlake"]
    di = eps[0]
    assert di.token == "openrouter/deepinfra/fp4"
    assert di.quantization == "fp4"
    assert di.context_length == 1048576
    # USD per token -> USD per 1M
    assert round(di.prompt_usd_per_m, 4) == 0.09
    assert round(di.completion_usd_per_m, 4) == 0.18
    assert di.up is True


def test_unknown_quant_becomes_none_and_negative_status_is_down():
    eps = parse_endpoints(PAYLOAD)
    do, streamlake = eps[1], eps[2]
    assert do.quantization is None  # "unknown" -> None
    assert streamlake.up is False  # status -2


def test_format_table_lists_tokens_and_a_paste_ready_sweep_line():
    out = format_table(parse_endpoints(PAYLOAD))
    assert "openrouter/deepinfra/fp4" in out
    assert "PROVIDER TOKEN" in out
    # the suggested sweep line includes only the endpoints that are up
    assert "--providers openrouter/deepinfra/fp4,openrouter/digitalocean" in out
    assert "openrouter/streamlake" not in out.split("--providers")[1]
    assert "3 endpoint(s), 2 up" in out


def test_format_table_empty():
    assert "no OpenRouter endpoints" in format_table([])


def test_fetch_endpoints_uses_injected_opener():
    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=0):
        captured["url"] = req.full_url
        return _Resp(json.dumps(PAYLOAD).encode())

    eps = fetch_endpoints("deepseek/deepseek-v4-flash-0731", opener=opener)
    assert captured["url"].endswith(
        "/models/deepseek/deepseek-v4-flash-0731/endpoints"
    )
    assert isinstance(eps[0], Endpoint)
    assert eps[0].tag == "deepinfra/fp4"
