#!/usr/bin/env python3
"""Does this host's prompt cache actually serve this model, with and without markers?

Agentic workloads resend a growing transcript every turn, so the prompt cache,
not the rate card, sets the input bill. Whether a cache works is not a property
of the host alone: it depends on the host, the model, and whether the client
opts in. A host can cache one model on an endpoint and report nothing for
another on the same endpoint.

This replays a growing conversation prefix through the same pinning proxy a
sweep uses, so marker injection is the identical code path, and reports the
steady-state hit ratio over turns 2..N (turn 1 writes the cache, it cannot hit).

    python3 scripts/cache_probe.py --provider doubleword/realtime \\
        --model zai-org/GLM-5.3-Flash --model deepseek-ai/DeepSeek-V4-Flash-0731

Measured 2026-09-03 on doubleword/realtime and doubleword/flex: DeepSeek V4
Flash reported 0% unmarked and 97.7% marked; GLM 5.3 Flash reported 0% either
way, on both tiers. A run whose marked ratio is 0 is the finding, not a bug in
the probe: check it against another model on the same host before concluding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compound.orproxy import serve_provider  # noqa: E402
from compound.providers_registry import parse_provider  # noqa: E402


def load_env(path: Path = Path(".env")) -> None:
    """Read keys without sourcing the file, which clobbers PATH in a shell."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def probe(provider: str, wire_model: str, marked: bool, turns: int, filler_tokens: int) -> float:
    """Steady-state cached/prompt ratio over turns 2..N. Returns a percentage."""
    # Explicit "0": an empty value now means the default, which is markers ON.
    os.environ["COMPOUND_DW_CACHE"] = "1" if marked else "0"
    spec = replace(parse_provider(provider), wire_model=wire_model)
    filler = "The quick brown fox jumps over the lazy dog. " * max(1, filler_tokens // 10)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": f"You are a terse assistant.\n{filler}"}
    ]
    rows: list[tuple[int, int]] = []
    with serve_provider(spec) as base:
        for turn in range(1, turns + 1):
            messages.append(
                {"role": "user", "content": f"Turn {turn}. Reply with just the number."}
            )
            request = urllib.request.Request(
                base.rstrip("/") + "/chat/completions",
                data=json.dumps(
                    {"model": "placeholder", "messages": messages, "max_tokens": 16}
                ).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer probe"},
            )
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.load(response)
            except Exception as exc:  # noqa: BLE001 - one bad turn should not end the probe
                print(f"    turn {turn}: ERROR {str(exc)[:90]}")
                continue
            usage = payload.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = int(details.get("cached_tokens") or 0)
            prompt = int(usage.get("prompt_tokens") or 0)
            rows.append((prompt, cached))
            content = (payload["choices"][0]["message"].get("content") or "ok")[:20]
            messages.append({"role": "assistant", "content": content})
            pct = cached / prompt * 100 if prompt else 0.0
            print(
                f"    turn {turn}: prompt={prompt:>7,} cached={cached:>7,} "
                f"({pct:5.1f}%)  {time.time() - started:4.1f}s"
            )
    steady = rows[1:]
    total_prompt = sum(p for p, _ in steady)
    total_cached = sum(c for _, c in steady)
    return (total_cached / total_prompt * 100) if total_prompt else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", required=True, help="provider token, e.g. doubleword/realtime"
    )
    parser.add_argument(
        "--model", action="append", required=True, help="model id as the host knows it, repeatable"
    )
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--filler-tokens", type=int, default=4000)
    args = parser.parse_args()
    load_env()

    results: dict[tuple[str, bool], float] = {}
    for model in args.model:
        for marked in (False, True):
            print(f"\n{args.provider}  {model}  markers={'ON' if marked else 'OFF'}")
            results[(model, marked)] = probe(
                args.provider, model, marked, args.turns, args.filler_tokens
            )

    print(f"\nSteady-state prompt-cache hit ratio on {args.provider} (turns 2-{args.turns})")
    print(f"{'model':<40s} {'markers off':>12s} {'markers on':>12s}")
    print("-" * 66)
    for model in args.model:
        off, on = results[(model, False)], results[(model, True)]
        print(f"{model:<40s} {off:>11.1f}% {on:>11.1f}%")
    print(
        "\nA host whose marked ratio stays at 0 while another model on the same "
        "host reaches 90%+ is not caching that model, and its input bill is the "
        "full transcript on every turn."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
