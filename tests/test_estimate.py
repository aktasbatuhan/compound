import json
import sys
from datetime import UTC, datetime

import pytest

from compound.estimate import (
    Catalog,
    TokenizerTable,
    estimate,
    workload_from_form,
    workload_from_shapes,
)
from compound.estimate.catalog import PriceCard, cards_from_endpoints, cards_from_models
from compound.shape import CallShape, build_profile

MODELS = {
    "data": [
        {"id": "closed/big", "architecture": {"tokenizer": "Claude"},
         "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000002"}},
        {"id": "open/small", "hugging_face_id": "org/Small", "architecture": {"tokenizer": "Other"},
         "pricing": {"prompt": "0.0000002", "completion": "0.000001",
                     "overrides": [{"min_prompt_tokens": 100000, "prompt": "0.0000004"}]}},
        {"id": "open/small:batch", "hugging_face_id": "org/Small",
         "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}},
        {"id": "open/small:free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "gpt/one", "architecture": {"tokenizer": "GPT"},
         "pricing": {"prompt": "0.000001", "completion": "0.000004"}},
    ]
}
ENDPOINTS = {
    "data": {
        "endpoints": [
            {"tag": "cheap/fp8", "provider_name": "Cheap", "quantization": "fp8", "status": 0,
             "pricing": {"prompt": "0.0000001", "completion": "0.0000008",
                         "input_cache_read": "0.00000001"}},
            {"tag": "down", "provider_name": "Down", "status": -5,
             "pricing": {"prompt": "0.00000001", "completion": "0.00000001"}},
            {"tag": "nightly", "provider_name": "Nightly", "status": 0,
             "pricing": {"prompt": "0.0000002", "completion": "0.000001",
                         "overrides": [{"utc_start": 1400, "utc_end": 0, "prompt": "0.0000001"},
                                       {"utc_days": ["saturday"], "prompt": "0.00000005"}]}},
        ]
    }
}
TOKENIZERS = TokenizerTable({
    "schema": "compound.tokenizer_ratios",
    "tokenizers": {
        "tiktoken:o200k_base": {"tr": {"tokens_per_1k_chars": 270.0}, "en": {"tokens_per_1k_chars": 185.0}},
        "org/Small": {"tr": {"tokens_per_1k_chars": 405.0}, "en": {"tokens_per_1k_chars": 185.0}},
    },
})


def _catalog() -> Catalog:
    models = cards_from_models(MODELS)
    return Catalog(fetched_at=0.0, sources=["test"], models=models,
                   hosts={"open/small": cards_from_endpoints(models["open/small"], ENDPOINTS)})


def _at(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


def test_listing_parsing_keeps_batch_and_drops_free_variants():
    models = cards_from_models(MODELS)
    assert "open/small:batch" in models and "open/small:free" not in models
    assert models["open/small:batch"].route == "openrouter:batch"
    assert models["closed/big"].cache_read == pytest.approx(0.2)
    assert models["open/small"].cache_read is None


def test_prompt_size_override_applies_only_above_its_threshold():
    card = cards_from_models(MODELS)["open/small"]
    assert card.rates_for(99_999, None).prompt == pytest.approx(0.2)
    assert card.rates_for(100_000, None).prompt == pytest.approx(0.4)
    assert card.rates_for(100_000, None).completion == pytest.approx(1.0)


def test_time_windows_wrap_midnight_and_weekdays_match():
    nightly = cards_from_endpoints(cards_from_models(MODELS)["open/small"], ENDPOINTS)[2]
    assert nightly.time_dependent
    thursday_noon = _at("2026-10-29T12:00:00")
    thursday_late = _at("2026-10-29T23:30:00")
    saturday_noon = _at("2026-10-31T12:00:00")
    assert nightly.rates_for(10, thursday_noon).prompt == pytest.approx(0.2)
    assert nightly.rates_for(10, thursday_late).prompt == pytest.approx(0.1)
    assert nightly.rates_for(10, saturday_noon).prompt == pytest.approx(0.05)
    # No timestamp, no time-based discount.
    assert nightly.rates_for(10, None).prompt == pytest.approx(0.2)


def test_tokenizer_ratios_are_exact_when_measured_and_ranged_when_not():
    models = cards_from_models(MODELS)
    gpt, small, closed = models["gpt/one"], models["open/small"], models["closed/big"]
    assert TOKENIZERS.ratio(gpt, small, "tr") == pytest.approx((1.5, 1.5))
    assert TOKENIZERS.ratio(gpt, small, "en") == pytest.approx((1.0, 1.0))
    low, high = TOKENIZERS.ratio(closed, small, "tr")
    assert low == pytest.approx(405 / 405) and high == pytest.approx(405 / 270)
    assert TOKENIZERS.basis(gpt) == "assumed"
    assert TOKENIZERS.basis(closed) == "unknown"
    assert TOKENIZERS.ratio(small, small, "tr") == (1.0, 1.0)


def test_catalog_round_trips_through_json():
    catalog = _catalog()
    again = Catalog.from_json(json.loads(json.dumps(catalog.to_json())))
    assert again.cards("open/small") == catalog.cards("open/small")


def test_hand_computed_monthly_cost_and_ranges():
    work = workload_from_form(monthly_calls=1000, input_tokens=1000, output_tokens=100,
                              cached_share=0.5, reuse_share=0.8)
    result = estimate(work, _catalog(), current="gpt/one", candidates=["open/small"],
                      tokenizers=TOKENIZERS)
    # gpt/one lists no cache price, so nothing is discounted:
    # 1000 calls x (1000 x $1 + 100 x $4) per million tokens.
    assert result["current"]["monthly_usd"] == pytest.approx(1.4)
    rows = {r["route"]: r for r in result["rows"] if r["model"] == "open/small"}
    cheap = rows["openrouter/cheap/fp8"]
    # English, same token density. Low end: 80% cached; high end: 50% cached.
    low = 1000 * (200 * 0.1 + 800 * 0.01 + 100 * 0.8) / 1e6
    high = 1000 * (500 * 0.1 + 500 * 0.01 + 100 * 0.8) / 1e6
    assert cheap["monthly_low"] == pytest.approx(low)
    assert cheap["monthly_high"] == pytest.approx(high)
    assert cheap["saving_high"] == pytest.approx(1 - low / 1.4)
    assert cheap["lever"] == "switch model, quality not yet measured"
    assert "openrouter/down" not in rows
    for row in result["rows"]:
        assert row["monthly_low"] <= row["monthly_high"] + 1e-12


def test_turkish_traffic_pays_the_tokenizer_difference():
    english = workload_from_form(monthly_calls=1000, input_tokens=1000, output_tokens=100)
    turkish = workload_from_form(monthly_calls=1000, input_tokens=1000, output_tokens=100,
                                 turkish_share=1.0)
    def batch(work):
        rows = estimate(work, _catalog(), current="gpt/one", candidates=["open/small"],
                        tokenizers=TOKENIZERS)["rows"]
        return next(r for r in rows if r["route"] == "openrouter:batch")
    assert batch(turkish)["monthly_high"] == pytest.approx(batch(english)["monthly_high"] * 1.5)


def test_levers_name_what_changes():
    work = workload_from_form(monthly_calls=10, input_tokens=100, output_tokens=10)
    rows = estimate(work, _catalog(), current="open/small", candidates=[],
                    tokenizers=TOKENIZERS)["rows"]
    levers = {r["route"]: r["lever"] for r in rows}
    assert levers["openrouter"] == "today"
    assert levers["openrouter/cheap/fp8"] == "same model, other host"
    assert levers["openrouter:batch"] == "batch tier, same model"


def test_shapes_need_a_day_or_an_explicit_volume():
    shapes = [CallShape(started_at=float(t), input_tokens=100, output_tokens=10) for t in range(5)]
    profile = build_profile(shapes)
    with pytest.raises(ValueError, match="less than a day"):
        workload_from_shapes(shapes, profile)
    work = workload_from_shapes(shapes, profile, monthly_calls=5000)
    assert work.monthly_calls == pytest.approx(5000)


def test_cli_estimates_from_a_saved_catalog(tmp_path, monkeypatch, capsys):
    from compound.cli import main

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_catalog().to_json()))
    out = tmp_path / "estimate.json"
    monkeypatch.setattr(sys, "argv", [
        "compound", "estimate", "--current", "gpt/one", "--candidates", "open/small",
        "--catalog", str(path), "--monthly-calls", "1000", "--input-tokens", "1000",
        "--output-tokens", "100", "--json", str(out),
    ])
    assert main() == 0
    result = json.loads(out.read_text())
    assert result["current"]["monthly_usd"] == pytest.approx(1.4)
    assert "not modelled" in capsys.readouterr().out


def test_unknown_models_fail_clearly(tmp_path, monkeypatch):
    from compound.cli import main

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_catalog().to_json()))
    monkeypatch.setattr(sys, "argv", [
        "compound", "estimate", "--current", "nope/model", "--catalog", str(path),
        "--monthly-calls", "1", "--input-tokens", "1", "--output-tokens", "1",
    ])
    with pytest.raises(SystemExit, match="not in the price catalog"):
        main()


def test_bundled_tokenizer_table_loads():
    table = TokenizerTable.bundled()
    assert "tiktoken:o200k_base" in table.table
    card = PriceCard(model="x", route="openrouter", provider="x", prompt=1, completion=1,
                     hugging_face_id="zai-org/GLM-5.3")
    assert table.density(card, "tr").basis == "measured"
    gpt = PriceCard(model="openai/gpt-6-sol", route="openrouter", provider="x", prompt=1,
                    completion=1, tokenizer="GPT")
    assert table.density(gpt, "tr").basis == "measured"
    newer = PriceCard(model="openai/gpt-7", route="openrouter", provider="x", prompt=1,
                      completion=1, tokenizer="GPT")
    assert table.density(newer, "tr").basis == "assumed"
