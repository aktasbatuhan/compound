"""Billing-grade Doubleword cost from the ``dw`` CLI.

Doubleword's inference API does not return a per-call cost, so a provider sweep
cannot price Doubleword episodes from the transcript the way it prices
OpenRouter episodes (which carry ``raw_data.usage.cost``). Earlier reports
worked around this with a hand-entered ``--prices`` rate card, which is only as
good as the guess — and a wrong guess silently inverts the cost comparison.

The ``dw`` CLI's ``usage`` command reports the *billed* cost for a time window,
broken down by model, together with an ``estimated_realtime_cost`` (what the same
tokens would have cost entirely at the realtime tier). Scoped to a single sweep,
that anchors both tiers' cost against a real invoice.

One trap: Doubleword's OpenAI-compatible API does **not** echo per-call token
usage the way OpenRouter does, so a multi-turn episode's tokens as counted from
the transcript are far below what ``dw usage`` shows as actually billed (each
turn re-sends the whole context). The report's per-host token totals are
therefore only a *relative* signal, not an absolute one. So the effective rate is
**calibrated to the report's own token basis** — ``rate * report_tokens`` is made
to equal the billed total — rather than derived from ``dw usage``'s token count.
That way ``cost_per_task`` comes out equal to ``billed_total / episodes``, which
is the only figure the invoice actually pins down. Rates are blended over
input/output (``dw usage`` bills one number across both).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

# Doubleword host labels as ProviderSpec.label emits them.
REALTIME_LABEL = "doubleword-realtime"
FLEX_LABEL = "doubleword-flex"


@dataclass(frozen=True, slots=True)
class DWUsage:
    """One model's billed usage over a window, from ``dw usage --output json``."""

    model: str
    input_tokens: int
    output_tokens: int
    total_cost: float
    estimated_realtime_cost: float
    request_count: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def parse_usage(payload: dict, model: str) -> DWUsage:
    """Pull one model's row out of a decoded ``dw usage --output json`` payload.

    Raises ``KeyError`` if the model is absent from the window (a clear signal
    that the ``--since``/``--until`` window does not cover the run).
    """
    for row in payload.get("by_model", []):
        if row.get("model") == model:
            return DWUsage(
                model=model,
                input_tokens=int(row.get("input_tokens", 0)),
                output_tokens=int(row.get("output_tokens", 0)),
                total_cost=float(row.get("cost", 0.0)),
                # per-model rows omit the realtime estimate; fall back to the
                # window total, which equals the row when the window is scoped.
                estimated_realtime_cost=float(
                    row.get("estimated_realtime_cost")
                    or payload.get("estimated_realtime_cost", 0.0)
                ),
                request_count=int(row.get("request_count", 0)),
            )
    raise KeyError(f"model {model!r} not found in dw usage window")


def fetch_usage(
    model: str,
    *,
    since: str,
    until: str | None = None,
    dw_bin: str = "dw",
    runner: Callable[[list[str]], str] | None = None,
) -> DWUsage:
    """Run ``dw usage`` for ``model`` over the window and parse the result.

    ``runner`` maps an argv list to the command's stdout; it defaults to a real
    subprocess call and is injectable so the parse path is testable offline.
    """
    argv = [dw_bin, "usage", "--since", since, "--output", "json"]
    if until:
        argv += ["--until", until]
    run = runner or _default_runner
    return parse_usage(json.loads(run(argv)), model)


def _default_runner(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(argv)}` failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def derive_tier_rates(
    usage: DWUsage, *, realtime_tokens: int, flex_tokens: int
) -> dict[str, float]:
    """Per-tier effective rates in **USD per million tokens** (blended in/out).

    ``realtime_tokens`` / ``flex_tokens`` are the sweep's per-tier token totals as
    the *report* counts them. The rate is calibrated to that basis — not to
    ``dw usage``'s token count, which is larger (DW under-reports per-call usage) —
    so that ``rate * report_tokens`` reproduces the billed total and
    ``cost_per_task`` equals ``billed_total / episodes``.

    With both tiers present, realtime is priced against the billed all-realtime
    estimate and flex takes the remainder of the actual bill; with a single tier
    present it simply absorbs the whole bill.
    """
    tot = realtime_tokens + flex_tokens
    rates: dict[str, float] = {}
    if tot <= 0:
        return rates
    rt_rate = usage.estimated_realtime_cost / tot  # $/report-token, calibrated
    if realtime_tokens > 0:
        rates[REALTIME_LABEL] = rt_rate * 1e6
    if flex_tokens > 0:
        if realtime_tokens > 0:
            # billed total = rt_rate*realtime_tokens + flex_rate*flex_tokens
            flex_rate = (usage.total_cost - rt_rate * realtime_tokens) / flex_tokens
        else:
            flex_rate = usage.total_cost / flex_tokens
        rates[FLEX_LABEL] = max(flex_rate, 0.0) * 1e6
    return rates
