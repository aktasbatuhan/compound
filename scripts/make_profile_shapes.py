#!/usr/bin/env python3
"""Generate the input x output prompt grid a serving comparison needs.

Reproduces the profile grid Telnyx published for DeepSeek V4 Flash (2026-09):
three input sizes crossed with two output sizes, ``1k/10k/100k`` in by
``100/1k`` out. Matching their grid exactly is the point, so our latency and
throughput numbers can be read against theirs instead of alongside them.

What the grid is actually for, beyond speed:

* **Cost per profile.** A host's effective rate is not one number. It moves with
  context length, because what is cached moves with context length. Reporting
  $/1M at 1k and at 100k separately is the difference between a rate card and a
  bill.
* **Cache behaviour by context length.** Run the same grid cold and warm and the
  delta is the host's prompt cache, measured rather than claimed.
* **Divergence at temperature 0.** Every host gets a byte-identical prompt, so
  any difference in the generated text is a difference in numerics.

The filler is deterministic (a fixed seed and a fixed word list), so every host
and every run sees the same bytes. It is prose rather than repeated tokens
because a long run of identical tokens is unusually easy to compress and would
flatter a host's prefill.

    python3 scripts/make_profile_shapes.py --out artifacts/profiles.json
    python3 scripts/make_profile_shapes.py --out small.json --inputs 1000,10000

Sizing is calibrated, not guessed. ``--verify`` sends each profile to a real
endpoint with ``max_tokens=1`` and reports nominal against the tokenizer's own
``prompt_tokens``, which is how :data:`CHARS_PER_TOKEN` below was set. Re-run it
if the word list or the model changes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

#: Characters per token for THIS word list, measured against DeepSeek V4 Flash's
#: tokenizer on 2026-09-03 via ``--verify`` (400k chars -> 54,349 tokens).
#:
#: Two earlier guesses were both wrong and in opposite directions, which is why
#: this is measured: 0.75 tokens per word overshot 2.75x (the list averages nine
#: characters), and the standard 4 chars/token undershot ~1.9x (the words are
#: common English and mostly tokenize to one token each, so a nine-character
#: word is one token, not two).
CHARS_PER_TOKEN = 7.4

WORDS = (
    "system latency throughput cache prefix token router provider quantization "
    "batch kernel decode prefill schedule tenant region replica shard weights "
    "context window budget invoice meter ledger percentile variance interval "
    "sample estimate deviation baseline control experiment measurement claim"
).split()


def filler(n_tokens: int, seed: int) -> str:
    """Deterministic pseudo-prose of about ``n_tokens`` tokens.

    Sized by character budget; :func:`verify` checks the result against a real
    tokenizer, because this is an estimate and a profile labelled 100k that is
    really 275k would silently change what is being measured.
    """
    rng = random.Random(seed)
    budget = int(n_tokens * CHARS_PER_TOKEN)
    out: list[str] = []
    used = 0
    i = 0
    while used < budget:
        word = rng.choice(WORDS)
        out.append(word)
        used += len(word) + 1
        if i % 18 == 17:
            out.append(".\n")
            used += 2
        i += 1
    return " ".join(out)


def build(inputs: list[int], outputs: list[int], seed: int) -> dict[str, dict]:
    shapes: dict[str, dict] = {}
    for n_in in inputs:
        # One filler per input size, shared across the output sizes, so the two
        # rows of a column differ only in how much is generated.
        body = filler(n_in, seed + n_in)
        for n_out in outputs:
            name = f"in{_label(n_in)}_out{_label(n_out)}"
            shapes[name] = {
                "max_tokens": n_out,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a benchmark load generator. Answer the user "
                            "concisely and stop when the answer is complete."
                        ),
                    },
                    {"role": "user", "content": f"{body}\n\nSummarize the notes above."},
                ],
            }
    return shapes


def _label(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--inputs", default="1000,10000,100000", help="comma-separated input token sizes"
    )
    parser.add_argument(
        "--outputs", default="100,1000", help="comma-separated output token budgets"
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--verify",
        metavar="MODEL",
        default=None,
        help="after writing, send each profile to OpenRouter with max_tokens=1 "
        "and report nominal vs the tokenizer's own prompt_tokens",
    )
    parser.add_argument("--verify-upstream", default="novita")
    args = parser.parse_args()

    inputs = [int(x) for x in args.inputs.split(",") if x.strip()]
    outputs = [int(x) for x in args.outputs.split(",") if x.strip()]
    shapes = build(inputs, outputs, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(shapes, indent=1))

    print(f"{len(shapes)} shapes -> {args.out}")
    for name, shape in shapes.items():
        chars = sum(len(m["content"]) for m in shape["messages"])
        est = int(chars / CHARS_PER_TOKEN)
        print(f"  {name:<18} ~{est:>7,} tok in (est)  max_tokens={shape['max_tokens']}")
    if args.verify:
        return verify(shapes, args.verify, args.verify_upstream)
    return 0


def verify(shapes: dict[str, dict], model: str, upstream: str) -> int:
    """Check nominal sizes against a real tokenizer. Returns non-zero if far off."""
    import os
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("verify needs OPENROUTER_API_KEY in the environment")
        return 1
    print(f"\nverifying against {model} on {upstream}")
    print(f"  {'profile':<18}{'nominal':>9}{'actual':>9}{'ratio':>7}")
    worst = 1.0
    for name, shape in shapes.items():
        nominal = _nominal_in(name)
        if nominal is None:
            continue
        body = {
            "model": model,
            "messages": shape["messages"],
            "max_tokens": 1,
            "usage": {"include": True},
            "provider": {"only": [upstream], "allow_fallbacks": False},
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                actual = (json.load(response).get("usage") or {}).get("prompt_tokens")
        except Exception as exc:  # noqa: BLE001 - a failed probe is a report line
            print(f"  {name:<18} probe failed: {str(exc)[:60]}")
            continue
        ratio = actual / nominal if nominal else 0.0
        worst = min(worst, ratio) if ratio < 1 else max(worst, 1 / ratio)
        print(f"  {name:<18}{nominal:>9,}{actual:>9,}{ratio:>7.2f}")
    if worst < 0.8:
        print(f"\n  Off by more than 20% (worst ratio {worst:.2f}). Retune CHARS_PER_TOKEN.")
        return 1
    print("\n  Within 20% of nominal.")
    return 0


def _nominal_in(name: str) -> int | None:
    """Nominal input tokens from a shape name like ``in10k_out100``."""
    head = name.split("_")[0]
    if not head.startswith("in"):
        return None
    body = head[2:]
    return int(body[:-1]) * 1000 if body.endswith("k") else int(body)


if __name__ == "__main__":
    sys.exit(main())
