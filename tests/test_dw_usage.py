"""Doubleword billing parse + per-tier rate recovery (no CLI call)."""

from __future__ import annotations

import json

import pytest

from compound.dw_usage import (
    FLEX_LABEL,
    REALTIME_LABEL,
    derive_tier_rates,
    fetch_usage,
    parse_usage,
)

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

# Shape returned by `dw usage --since ... --output json` (real values from the run).
PAYLOAD = {
    "total_input_tokens": 88463840,
    "total_output_tokens": 3613079,
    "total_request_count": 12325,
    "total_cost": "7.647321660000000",
    "estimated_realtime_cost": "8.61209982",
    "by_model": [
        {
            "model": MODEL,
            "input_tokens": 88463840,
            "output_tokens": 3613079,
            "cost": "7.647321660000000",
            "request_count": 12325,
        }
    ],
}


def test_parse_usage_reads_the_model_row():
    u = parse_usage(PAYLOAD, MODEL)
    assert u.total_tokens == 88463840 + 3613079
    assert u.total_cost == pytest.approx(7.64732166)
    assert u.estimated_realtime_cost == pytest.approx(8.61209982)
    assert u.request_count == 12325


def test_parse_usage_missing_model_raises():
    with pytest.raises(KeyError):
        parse_usage(PAYLOAD, "some/other-model")


def test_rates_reproduce_the_billed_total_on_the_report_token_basis():
    # The report counts far fewer tokens than dw usage bills (DW under-reports
    # per-call usage). The rate must be calibrated to the report's basis so that
    # rate * report_tokens == the billed total — regardless of the report count.
    u = parse_usage(PAYLOAD, MODEL)
    for basis in (u.total_tokens, u.total_tokens // 6, 12_228_657):  # incl. real run
        rt = int(basis * 0.5005)
        flex = basis - rt
        rates = derive_tier_rates(u, realtime_tokens=rt, flex_tokens=flex)
        rebilled = (rates[REALTIME_LABEL] * rt + rates[FLEX_LABEL] * flex) / 1e6
        assert rebilled == pytest.approx(u.total_cost, rel=1e-6)
        # flex is cheaper, ~22% on this window, independent of the token basis
        assert 0.15 < 1 - rates[FLEX_LABEL] / rates[REALTIME_LABEL] < 0.30


def test_tier_totals_sum_to_the_invoice_and_realtime_uses_its_share():
    # Applying the rates to each tier's report tokens must (a) sum to the billed
    # invoice and (b) bill realtime at the all-realtime rate over its token share.
    u = parse_usage(PAYLOAD, MODEL)
    rt_tok, flex_tok = 6_119_841, 6_108_816
    rates = derive_tier_rates(u, realtime_tokens=rt_tok, flex_tokens=flex_tok)
    realtime_total = rates[REALTIME_LABEL] * rt_tok / 1e6
    flex_total = rates[FLEX_LABEL] * flex_tok / 1e6
    assert realtime_total + flex_total == pytest.approx(u.total_cost, rel=1e-9)
    share = rt_tok / (rt_tok + flex_tok)
    assert realtime_total == pytest.approx(u.estimated_realtime_cost * share, rel=1e-9)


def test_derive_tier_rates_single_tier():
    u = parse_usage(PAYLOAD, MODEL)
    only_flex = derive_tier_rates(u, realtime_tokens=0, flex_tokens=u.total_tokens)
    assert REALTIME_LABEL not in only_flex
    # only flex present -> whole billed total spread over flex tokens
    assert only_flex[FLEX_LABEL] == pytest.approx(u.total_cost / u.total_tokens * 1e6)


def test_fetch_usage_uses_injected_runner():
    calls = {}

    def runner(argv):
        calls["argv"] = argv
        return json.dumps(PAYLOAD)

    u = fetch_usage(MODEL, since="2026-08-09", until="2026-08-11", runner=runner)
    assert u.model == MODEL
    assert calls["argv"][:2] == ["dw", "usage"]
    assert "--since" in calls["argv"] and "2026-08-09" in calls["argv"]
    assert "--until" in calls["argv"] and "2026-08-11" in calls["argv"]


# A window that covers two models: `dw usage` still reports one window-level
# estimated_realtime_cost, which is their SUM and belongs to neither row.
TWO_MODEL_PAYLOAD = {
    "total_input_tokens": 16326089,
    "total_output_tokens": 689552,
    "total_cost": "1.990764900000000",
    "estimated_realtime_cost": "2.27913817",
    "by_model": [
        {
            "model": "zai-org/GLM-5.3-Flash",
            "input_tokens": 9734460,
            "output_tokens": 317510,
            "cost": "1.405318320000000",
            "request_count": 370,
        },
        {
            "model": MODEL,
            "input_tokens": 6591629,
            "output_tokens": 372042,
            "cost": "0.585446580000000",
            "request_count": 262,
        },
    ],
}


def test_parse_usage_refuses_to_attribute_a_shared_realtime_estimate():
    # Charging the window's estimate to one of two models inflates its derived
    # realtime rate, which silently inverts a per-tier cost comparison.
    usage = parse_usage(TWO_MODEL_PAYLOAD, MODEL)
    assert usage.total_cost == pytest.approx(0.58544658)
    assert usage.estimated_realtime_cost is None


def test_derive_tier_rates_refuses_without_a_per_model_estimate():
    usage = parse_usage(TWO_MODEL_PAYLOAD, MODEL)
    with pytest.raises(ValueError, match="cannot be separated"):
        derive_tier_rates(usage, realtime_tokens=1_000_000, flex_tokens=1_000_000)
