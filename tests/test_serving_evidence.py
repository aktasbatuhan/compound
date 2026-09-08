"""The public example reproduces measurements without private logs or dependencies."""

import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serving_evidence", ROOT / "scripts/serving_evidence.py"
)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def test_export_excludes_disallowed_rows_and_private_content(tmp_path):
    base = {k: None for k in evidence.FIELDS}
    base.update(
        iso="2026-09-03T00:00:00+0000",
        route="a",
        status=200,
        text="PRIVATE-OUTPUT",
        messages="PRIVATE-PROMPT",
        error="PRIVATE-ACCOUNT-DETAILS",
    )
    for source, relative in evidence.SOURCES.items():
        path = tmp_path / "private" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [base, {**base, "status": 402}]
        if source == "original":
            rows.append({**base, "route": "doubleword-flex"})
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    bundle = tmp_path / "public"
    evidence.export(tmp_path / "private", bundle)
    manifest, rows = evidence.load(bundle)
    raw = gzip.decompress((bundle / "calls.jsonl.gz").read_bytes()).decode()
    assert "PRIVATE" not in raw
    assert len(rows) == 3
    assert all(r["status"] != 402 for r in rows)
    assert manifest["sources"][0]["excluded"] == {
        "account_credit": 1,
        "unmarked_doubleword": 1,
    }
    assert all(
        set(r) == set(evidence.FIELDS) | {"error_class", "source", "source_line"} for r in rows
    )


def test_summary_uses_only_observed_cost_and_cache_denominators():
    row = {k: None for k in evidence.FIELDS}
    row.update(
        source="original",
        shape="small",
        cache_mode="warm",
        route="a",
        status=200,
        error_class=None,
        ttft_s=1,
        decode_tps=10,
    )
    cells = evidence.summarize(
        [
            {**row, "prompt_tokens": 100, "cached_tokens": 80, "cost_usd": 0.01},
            {**row, "prompt_tokens": 900, "cached_tokens": None, "cost_usd": None},
            {**row, "status": 429, "error_class": "http_429", "ttft_s": 99},
        ]
    )
    c = cells[0]
    assert c["cache_share"] == 0.8
    assert c["cost_per_m_prompt"] == 100
    assert c["failures"] == 1 and c["n"] == 3
    assert c["priced_samples"] == 1
    assert c["ttft_s"] == 1 and c["ttft_samples"] == 2
    assert c["failure_ci"][0] < 1 / 3 < c["failure_ci"][1]


def test_bundled_selection_and_reproduction_with_stdlib_only(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(evidence.BUNDLE, bundle)
    manifest, rows = evidence.load(bundle)
    assert len(rows) == 12812
    assert sum(r["source"] != "auto_followup" for r in rows) == 12795
    assert all(r["source"] == "marked" for r in rows if r["route"].startswith("doubleword-"))
    assert sum(s["excluded"].get("account_credit", 0) for s in manifest["sources"]) == 85
    # -S disables site packages; -I ignores PYTHONPATH and user packages.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "scripts/serving_evidence.py"),
            "--bundle",
            str(bundle),
            "--check",
        ],
        check=True,
    )
    (bundle / "index.html").unlink()
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "scripts/serving_evidence.py"),
            "--bundle",
            str(bundle),
        ],
        check=True,
    )
    assert (bundle / "index.html").read_bytes() == (evidence.BUNDLE / "index.html").read_bytes()


def test_corrupt_bundle_is_refused(tmp_path):
    shutil.copytree(evidence.BUNDLE, tmp_path / "bundle")
    with (tmp_path / "bundle/calls.jsonl.gz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        evidence.load(tmp_path / "bundle")


def test_p90_interpolates_and_ignores_missing_timings():
    rows = [{"t": v} for v in [None, 1, 2, 3, 4]]
    assert evidence.percentile(rows, "t", 0.9) == pytest.approx(3.7)
    assert evidence.percentile([{"t": None}], "t", 0.9) is None


def test_figures_use_primary_measurements_only():
    manifest, rows = evidence.load(evidence.BUNDLE)
    cells = evidence.summarize(rows)
    primary = [c for c in cells if c["source"] != "auto_followup"]
    before = evidence.charts(primary, rows)
    extra = {
        **next(r for r in rows if r["source"] == "auto_followup"),
        "provider_echo": "FOLLOWUP_SENTINEL",
    }
    assert evidence.charts(primary, rows + [extra]) == before
    page = evidence.render(manifest, cells, rows)
    assert "Rebuild this report" not in page
    assert "Serving study ·" not in page
    assert "First token p90" in page


def test_doubleword_costs_use_matching_calibration_without_changing_calls():
    manifest, rows = evidence.load(evidence.BUNDLE)
    cells = evidence.summarize(rows, manifest["cost_calibration"])
    marked = [c for c in cells if c["source"] == "marked"]
    assert len(marked) == 24
    for c in marked:
        calibration = next(
            x
            for x in manifest["cost_calibration"]
            if x["route"] == c["route"]
            and x["cache_mode"] == c["cache_mode"]
            and c["shape"] in x["shapes"]
        )
        assert c["cost_per_m_prompt"] == pytest.approx(
            calibration["cost_usd"] / calibration["input_tokens"] * 1e6
        )
        assert c["priced_samples"] == 0
        assert c["cost_calibration_samples"] == calibration["calls"]
    assert all(r["cost_usd"] is None for r in rows if r["source"] == "marked")
    assert next(
        c
        for c in marked
        if c["route"] == "doubleword-flex"
        and c["shape"] == "in10k_out100"
        and c["cache_mode"] == "cold"
    )["cost_per_m_prompt"] == pytest.approx(0.09094969199)
