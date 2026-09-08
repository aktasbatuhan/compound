import base64
import gzip
import json
import re
import subprocess
import sys

import pytest

from compound.serving_report import aggregate, build_report, read_results


def record(**overrides):
    base = dict(
        route="host/a",
        model="my-model",
        shape="support-ticket",
        mode="off",
        cache_mode="cold",
        temperature=0,
        max_tokens=100,
        status=200,
        ttft_s=1,
        decode_tps=20,
        prompt_tokens=100,
        cached_tokens=0,
        cost_usd=0.001,
    )
    return {**base, **overrides}


def write(tmp_path, rows, name="results.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_measured_denominators_failures_and_settings(tmp_path):
    path = write(
        tmp_path,
        [
            record(),
            record(prompt_tokens=900, cost_usd=None, cached_tokens=None, ttft_s=3),
            record(status=402, error="credit secret", ttft_s=99),
            record(status=200, error="broken stream"),
            record(mode="on"),
            record(model="other-model"),
            record(max_tokens=1000),
            record(cache_marked=True),
            record(temperature=0.7),
        ],
    )
    rows, _ = read_results([path])
    cells = aggregate(rows)
    assert len(cells) == 6
    c = cells[0]
    assert (c["n"], c["successes"], c["failures"]) == (4, 2, 2)
    assert c["cost_per_m_prompt"] == 10
    assert c["cost_per_m_prompt_samples"] == 1
    assert c["cache_share"] == 0
    assert c["cache_share_samples"] == 1
    assert c["ttft_s"] == 2 and c["ttft_s_p90"] == pytest.approx(2.8)
    assert c["failure_ci"][0] < 0.5 < c["failure_ci"][1]


def test_portable_report_sanitization_escaping_and_missing_data(tmp_path):
    route = '</script><script>alert("bad")</script>'
    path = write(
        tmp_path,
        [
            record(
                route=route,
                shape="@@TITLE@@",
                text="PRIVATE",
                messages="PRIVATE",
                api_key="PRIVATE",
            ),
            record(route="failed", status=None, error="PRIVATE", cost_usd=None, ttft_s=None),
            record(cache_mode="warm", cached_tokens=90, cost_usd=0.0001),
        ],
    )
    out = build_report([path], tmp_path / "report.html", title="<Test>")
    doc = out.read_text()
    assert "<title>&lt;Test&gt;</title>" in doc
    assert route not in doc and "PRIVATE" not in doc
    assert "@@TITLE@@" in doc  # user text is not recursively substituted
    assert "Cache cost impact" in doc and "Not reported" in doc
    payloads = re.findall(r'download="([^"]+)" href="data:[^;]+;base64,([^"]+)"', doc)
    downloads = {name: base64.b64decode(data) for name, data in payloads}
    assert set(downloads) == {"calls.jsonl.gz", "summary.json", "manifest.json"}
    exported = gzip.decompress(downloads["calls.jsonl.gz"]).decode()
    assert "PRIVATE" not in exported and "api_key" not in exported
    assert len(exported.splitlines()) == 3
    manifest = json.loads(downloads["manifest.json"])
    assert manifest["included_rows"] == 3
    assert "results.jsonl" not in downloads["manifest.json"].decode()
    # Deterministic output, including gzip download bytes.
    assert (
        build_report([path], tmp_path / "second.html", title="<Test>").read_bytes()
        == out.read_bytes()
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(ttft_s=float("nan")),
        dict(cost_usd=-1),
        dict(prompt_tokens=True),
        dict(cached_tokens=101),
        dict(status=200.1),
        dict(shape=None),
        dict(cache_marked="yes"),
    ],
)
def test_invalid_measurements_fail_with_line_number(tmp_path, changes):
    path = write(tmp_path, [record(**changes)])
    with pytest.raises(ValueError, match=r"results.jsonl:1:"):
        build_report([path], tmp_path / "report.html")
    assert not (tmp_path / "report.html").exists()


def test_multiple_files_gzip_legacy_and_overwrite(tmp_path):
    path = write(tmp_path, [record()])
    zipped = tmp_path / "more.jsonl.gz"
    zipped.write_bytes(
        gzip.compress(json.dumps(record(mode=None, model=None, max_tokens=None)).encode())
    )
    rows, sources = read_results([path, zipped])
    assert len(rows) == 2 and len(aggregate(rows)) == 2
    assert [s["rows"] for s in sources] == [1, 1]
    with pytest.raises(ValueError, match="duplicate input"):
        read_results([path, path])
    with pytest.raises(ValueError, match="overwrite an input"):
        build_report([path], path, force=True)
    out = build_report([path, zipped], tmp_path / "report.html")
    with pytest.raises(FileExistsError):
        build_report([path], out)
    build_report([path], out, force=True)


def test_empty_and_malformed_inputs(tmp_path):
    for content in ("", "{}\n", "{broken\n", "[]\n"):
        path = tmp_path / "results.jsonl"
        path.write_text(content)
        with pytest.raises(ValueError):
            read_results([path])


def test_cli_report_offline(tmp_path):
    path = write(tmp_path, [record(status=429, cost_usd=None, ttft_s=None)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compound.bench",
            "serving-report",
            str(path),
            "--out",
            str(tmp_path / "report.html"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    doc = (tmp_path / "report.html").read_text()
    assert "No provider has both" in doc
    assert "100.00%" in doc
    assert "Report:" in result.stdout
