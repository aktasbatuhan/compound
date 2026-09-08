#!/usr/bin/env python3
"""Export sanitized serving evidence or rebuild its report, with Python's stdlib only."""

# HTML/CSS templates stay readable as complete markup lines.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "site/report/serving"
SOURCES = {
    "original": "serving-1788459928/all.jsonl",
    "marked": "serving-dw-marked/all.jsonl",
    "auto_followup": "serving-auto-402-rerun/all.jsonl",
}
# Export only measurements and provenance. Never export request/response text,
# raw errors (which can contain account identifiers), credentials, or billing accounts.
FIELDS = (
    "iso",
    "route",
    "mode",
    "shape",
    "rep",
    "cache_mode",
    "temperature",
    "status",
    "total_s",
    "ttft_s",
    "finish_reason",
    "provider_echo",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cost_usd",
    "decode_tps",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exclude(source: str, row: dict) -> str | None:
    if source == "original" and row["route"].startswith("doubleword-"):
        return "unmarked_doubleword"
    if row.get("status") == 402:
        # All 402s in these archived runs were inspected as account-credit errors.
        return "account_credit"
    return None


def sanitize(source: str, line: int, row: dict) -> dict:
    out = {k: row.get(k) for k in FIELDS}
    error = row.get("error")
    status = row.get("status")
    out["error_class"] = (
        f"http_{status}"
        if status is not None and status != 200
        else "transport_or_response_error"
        if error
        else None
    )
    out.update(source=source, source_line=line)
    return out


def cost_calibration(archives: Path) -> list[dict]:
    path = archives / "serving-dw-cost/dw-cost.jsonl"
    if not path.exists():
        return []
    snapshots = [json.loads(line) for line in path.read_text().splitlines()]
    result = []
    for start, end in zip(snapshots[1::2], snapshots[2::2], strict=True):
        label = start["label"].removesuffix(":start")
        if end["label"] != label + ":end":
            raise ValueError("unmatched cost calibration interval")
        tier, _, condition = label.split("|")
        size, cache = condition.split("-")
        calls_path = archives / f"serving-dw-cost/out/{tier}-{condition}/results.jsonl"
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        tokens = sum(c["prompt_tokens"] for c in calls)
        before, after = start["usage"], end["usage"]
        if after["total_input_tokens"] - before["total_input_tokens"] != tokens or after[
            "total_request_count"
        ] - before["total_request_count"] != len(calls):
            raise ValueError("cost calibration usage does not match calls")
        charge = float(after["total_cost"]) - float(before["total_cost"])
        result.append(
            {
                "route": "doubleword-" + tier,
                "cache_mode": cache,
                "input_group": size,
                "shapes": sorted({c["shape"] for c in calls}),
                "calls": len(calls),
                "input_tokens": tokens,
                "cost_usd": charge,
                "cost_per_m_prompt": charge / tokens * 1e6,
                "start": start["ts"],
                "end": end["ts"],
                "source_sha256": digest(path.read_bytes()),
                "calls_sha256": digest(calls_path.read_bytes()),
            }
        )
    return result


def export(archives: Path, bundle: Path) -> None:
    records, sources = [], []
    for name, relative in SOURCES.items():
        data = (archives / relative).read_bytes()
        excluded: Counter = Counter()
        count = 0
        times = []
        for line, raw in enumerate(data.decode().splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            count += 1
            times.append(row["iso"])
            reason = exclude(name, row)
            if reason:
                excluded[reason] += 1
                continue
            records.append(sanitize(name, line, row))
        sources.append(
            {
                "id": name,
                "archive": relative,
                "sha256": digest(data),
                "rows": count,
                "excluded": dict(excluded),
                "included": count - sum(excluded.values()),
                "start": min(times),
                "end": max(times),
            }
        )
    payload = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records
    ).encode()
    bundle.mkdir(parents=True, exist_ok=True)
    packed = gzip.compress(payload, mtime=0)
    (bundle / "calls.jsonl.gz").write_bytes(packed)
    manifest = {
        "schema_version": 1,
        "cost_calibration": cost_calibration(archives),
        "cost_method": "Doubleword costs use matched account-meter charge deltas divided by input tokens from separate calibration calls. Small calibration pools 1k and 10k shapes; large pools 100k shapes, each across output budgets. Applied to corresponding marked-run cells without assigning costs to individual calls.",
        "model": "DeepSeek V4 Flash (0731)",
        "model_identity_basis": "Archived experiment configuration; legacy rows lack model IDs.",
        "sources": sources,
        "included_rows": len(records),
        "calls_sha256": digest(packed),
        "decoded_sha256": digest(payload),
        "selection": "Original non-Doubleword routes plus marked Doubleword rerun. "
        "All account-credit 402 rows excluded. Auto follow-up shown separately.",
        "limitations": [
            "Controlled request shapes, not production traffic or task-quality evaluation.",
            "Doubleword was rerun later; time-of-day effects are not controlled across those runs.",
            "Cold requests contain unique nonces; only warm requests repeat identical prefixes.",
            "Quantization suffixes are discovery labels, not verified per-call constraints.",
            "Excluding credit errors changes denominators and leaves gaps in affected cells.",
            "Cost uses per-call charges where reported and separate Doubleword calibration intervals where available.",
            "Repeated calls are dependent observations; intervals are descriptive, not causal tests.",
            "No raw prompt, output, or error text is included in this public measurement bundle.",
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load(bundle: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    packed = (bundle / "calls.jsonl.gz").read_bytes()
    if digest(packed) != manifest["calls_sha256"]:
        raise ValueError("evidence archive checksum mismatch")
    payload = gzip.decompress(packed)
    if digest(payload) != manifest["decoded_sha256"]:
        raise ValueError("decoded evidence checksum mismatch")
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    if len(rows) != manifest["included_rows"]:
        raise ValueError("evidence row count mismatch")
    for row in rows:
        if row.get("status") == 402 or exclude(row["source"], row):
            raise ValueError("excluded evidence present in public bundle")
    return manifest, rows


def percentile(rows: list[dict], field: str, q: float) -> float | None:
    values = sorted(r[field] for r in rows if r.get(field) is not None)
    if not values:
        return None
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def median(rows: list[dict], field: str) -> float | None:
    values = [r[field] for r in rows if r.get(field) is not None]
    return statistics.median(values) if values else None


def wilson(k: int, n: int) -> list[float]:
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, centre - half), min(1, centre + half)]


def summarize(rows: list[dict], calibration: list[dict] | None = None) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        groups[row["source"], row["shape"], row["cache_mode"], row["route"]].append(row)
    result = []
    for (source, shape, cache, route), rs in sorted(groups.items()):
        good = [r for r in rs if r["status"] == 200 and not r["error_class"]]
        cached = [
            r for r in good if r["cached_tokens"] is not None and r["prompt_tokens"] is not None
        ]
        priced = [r for r in good if r["cost_usd"] is not None and r["prompt_tokens"] is not None]
        prompt = sum(r["prompt_tokens"] for r in priced)
        cache_prompt = sum(r["prompt_tokens"] for r in cached)
        failures = len(rs) - len(good)
        result.append(
            {
                "source": source,
                "shape": shape,
                "cache_mode": cache,
                "route": route,
                "n": len(rs),
                "successes": len(good),
                "failures": failures,
                "failure_ci": wilson(failures, len(rs)),
                "ttft_s": median(good, "ttft_s"),
                "ttft_p90_s": percentile(good, "ttft_s", 0.9),
                "decode_tps": median(good, "decode_tps"),
                "total_s": median(good, "total_s"),
                "ttft_samples": sum(r["ttft_s"] is not None for r in good),
                "decode_samples": sum(r["decode_tps"] is not None for r in good),
                "cache_share": (
                    sum(r["cached_tokens"] for r in cached) / cache_prompt if cache_prompt else None
                ),
                "cache_samples": len(cached),
                "priced_samples": len(priced),
                "cost_per_m_prompt": (
                    sum(r["cost_usd"] for r in priced) / prompt * 1e6 if prompt else None
                ),
            }
        )
    for cell in result:
        if cell["source"] != "marked" or cell["cost_per_m_prompt"] is not None:
            continue
        matches = [
            c
            for c in calibration or []
            if c["route"] == cell["route"]
            and c["cache_mode"] == cell["cache_mode"]
            and cell["shape"] in c["shapes"]
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous cost calibration")
        if matches:
            cell["cost_per_m_prompt"] = matches[0]["cost_per_m_prompt"]
            cell["cost_calibration_samples"] = matches[0]["calls"]
    return result


def f(value: float | None, digits: int = 2) -> str:
    return "not reported" if value is None else f"{value:,.{digits}f}"


def charts(cells: list[dict], rows: list[dict]) -> str:
    """Render figures from the same aggregates as the detailed tables."""
    esc = html.escape
    key = {(c["shape"], c["cache_mode"], c["route"]): c for c in cells}
    routes = sorted(
        {c["route"] for c in cells},
        key=lambda r: key["in10k_out100", "cold", r]["ttft_s"] or math.inf,
    )

    def text(x, y, label, anchor="start", cls=""):
        return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" class="{cls}">{esc(str(label))}</text>'

    def line(x1, y1, x2, y2, cls="grid"):
        return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{cls}"/>'

    def dot(x, y, label, cls="mark", radius=4):
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" class="{cls}"><title>{esc(label)}</title></circle>'

    def interactive(provider, tip):
        return f'<g class="provider" data-provider="{esc(provider)}" data-tip="{esc(tip)}" tabindex="0" role="group" aria-label="{esc(tip)}">'

    def svg(title, content, width=1040, height=550):
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="group" aria-label="{esc(title)}"><title>{esc(title)}</title>{content}</svg>'

    def figure(title, description, graphic, metric, scroll=True):
        return f'<section class="figure"><h2>{title}</h2><p>{description}</p><div class="chart {"chart-scroll" if scroll else ""}" tabindex="0">{graphic}</div><p class="metric">{metric}</p></section>'

    result = []
    priced = sorted(
        [
            key["in10k_out100", "cold", r]
            for r in routes
            if key["in10k_out100", "cold", r]["cost_per_m_prompt"] is not None
        ],
        key=lambda c: c["cost_per_m_prompt"],
    )
    art = text(65, 25, "Median first-token wait (seconds)")
    for tick in range(0, 7, 2):
        y = 410 - tick / 7 * 355
        art += line(65, y, 570, y) + text(50, y + 4, f"{tick}s", "end")
    for tick in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        x = 65 + tick / 0.5 * 505
        art += text(x, 438, f"${tick:.2f}", "middle")
    for i, c in enumerate(priced):
        art += interactive(
            c["route"],
            f"{c['route']} · cost ${c['cost_per_m_prompt']:.4f}/1M input tokens · median TTFT {c['ttft_s']:.2f}s",
        )
        x, y = 65 + c["cost_per_m_prompt"] / 0.5 * 505, 410 - c["ttft_s"] / 7 * 355
        art += dot(
            x, y, f"{c['route']}: ${c['cost_per_m_prompt']:.4f}, {c['ttft_s']:.2f}s", radius=5
        )
        art += text(x, y + 19 if c["route"] == "together" else y - 9, i + 1, "middle")
        art += text(610, 60 + i * 29, f"{i + 1:2}. {c['route']}") + text(
            1030, 60 + i * 29, f"${c['cost_per_m_prompt']:.3f} · {c['ttft_s']:.1f}s", "end"
        )
        art += "</g>"
    art += text(65, 478, "Request cost per million input tokens →")
    result.append(
        figure(
            "Cost and first-token latency",
            "10k input target, 100 output-token budget, cold cache. Lower and further left means cheaper requests and a shorter wait for the first token. The list is ordered by cost.",
            svg("Cost versus median time to first token", art, height=500),
            "Cost = total request charges ÷ input tokens × 1M, including output charges. It is not the listed input-token price. Cost points cover 13 routes. Hover a point for its route and values.",
        )
    )

    for field, title, desc, unit in [
        (
            "ttft_s",
            "Time to first token",
            "The dot is the median wait; the line extends to the 90th percentile, the wait met or beaten by 90% of timed successful calls. A short line means less spread between typical and slower requests.",
            "s",
        ),
        (
            "decode_tps",
            "Generation speed",
            "Median tokens generated per second after streaming starts. This matters when the answer is long: a quick first token can still be followed by slow generation.",
            "tok/s",
        ),
    ]:
        maxval = max(
            c["ttft_p90_s"] if field == "ttft_s" else c[field]
            for c in cells
            if c["cache_mode"] == "cold" and c["shape"].endswith("_out100") and c[field] is not None
        )
        ceiling = math.ceil(maxval / (10 if field == "ttft_s" else 100)) * (
            10 if field == "ttft_s" else 100
        )
        panels = ""
        for size in ["1k", "10k", "100k"]:
            art = ""
            for tick in [0, ceiling / 2, ceiling]:
                x = 170 + tick / ceiling * 150
                art += line(x, 30, x, 458) + text(x, 18, f"{tick:g}", "middle")
            for i, r in enumerate(routes):
                c = key[f"in{size}_out100", "cold", r]
                tip = f"{r} · {size} input · {title}: {f(c[field])} {unit}"
                if field == "ttft_s":
                    tip += f" · p90 {f(c['ttft_p90_s'])}s · n={c['ttft_samples']}"
                else:
                    tip += f" · n={c['decode_samples']}"
                art += interactive(r, tip)
                y = 48 + i * 30
                art += text(160, y + 4, r, "end")
                value = c[field]
                if value is None:
                    art += text(175, y + 4, "not reported") + "</g>"
                    continue
                x = 170 + value / ceiling * 150
                if field == "ttft_s":
                    art += line(x, y, 170 + c["ttft_p90_s"] / ceiling * 150, y, "range")
                art += dot(
                    x,
                    y,
                    f"{r}: {value:.2f} {unit}; n={c['ttft_samples' if field == 'ttft_s' else 'decode_samples']}",
                )
                art += text(365, y + 4, f"{value:.1f}", "end") + "</g>"
            panels += f"<div><h3>{size} input · {unit}</h3>{svg(title + ' at ' + size + ' input, cold', art, 375, 475)}</div>"
        result.append(
            figure(
                title,
                desc
                + " All three panels use cold requests with a 100-token output budget and the same axis scale.",
                f'<div class="panels">{panels}</div>',
                "Timing includes successful calls only. "
                + (
                    "TTFT runs from request start to the first content or reasoning delta received by the client; it includes network and queueing time. P90 uses linear interpolation between ordered observations."
                    if field == "ttft_s"
                    else "Decode speed = reported completion tokens ÷ elapsed time from the first to last output delta. Non-streaming responses have no decode-speed measurement."
                ),
                False,
            )
        )

    group = sorted(
        [key["in10k_out100", "warm", r] for r in routes], key=lambda c: -(c["cache_share"] or 0)
    )
    art = ""
    for tick in [0, 25, 50, 75, 100]:
        x = 260 + tick * 6.7
        art += line(x, 30, x, 485) + text(x, 18, f"{tick}%", "middle")
    for i, c in enumerate(group):
        art += interactive(
            c["route"],
            f"{c['route']} · cache share {f(c['cache_share'] * 100 if c['cache_share'] is not None else None)}% · n={c['cache_samples']}",
        )
        y = 52 + i * 31
        art += text(242, y + 4, c["route"], "end")
        if c["cache_share"] is not None:
            x = 260 + c["cache_share"] * 670
            art += (
                line(260, y, x, y, "range")
                + dot(x, y, f"{c['route']}: {c['cache_share'] * 100:.2f}%; n={c['cache_samples']}")
                + text(x + 12, y + 4, f"{c['cache_share'] * 100:.1f}%")
            )
        art += "</g>"
    result.append(
        figure(
            "Cache hit",
            "Share of input tokens reported as cached on repeated 10k prompts, with a 100-token output budget. A high share means less input work is billed at the uncached rate when a cache discount applies.",
            svg("Token-weighted cache share for warm 10k prompts", art, height=505),
            "Cache share = sum of cached input tokens ÷ sum of input tokens for successes reporting both. This is a token share, not the percentage of requests that hit cache; the warm sequence includes its initial request.",
        )
    )

    pairs = [
        (
            r,
            key["in100k_out100", "cold", r]["cost_per_m_prompt"],
            key["in100k_out100", "warm", r]["cost_per_m_prompt"],
        )
        for r in routes
    ]
    pairs = [(r, c, w) for r, c, w in pairs if c is not None and w is not None and c > 0 and w > 0]

    def sy(v):
        return 440 - (math.log10(v) - math.log10(0.01)) / (math.log10(0.5) - math.log10(0.01)) * 385

    def label_positions(index):
        ordered = sorted(pairs, key=lambda p: -p[index])
        last = 22
        positions = {}
        for p in ordered:
            last = max(sy(p[index]), last + 27)
            positions[p[0]] = last
        return positions

    left, right = label_positions(1), label_positions(2)
    art = (
        text(320, 22, "Cold", "middle")
        + text(705, 22, "Warm", "middle")
        + line(320, 42, 320, 465)
        + line(705, 42, 705, 465)
    )
    for r, c, w in pairs:
        art += interactive(
            r, f"{r} · cold ${c:.4f} → warm ${w:.4f} per 1M input tokens · {c / w:.1f}× lower"
        )
        y1, y2 = sy(c), sy(w)
        art += (
            line(320, y1, 705, y2, "focus-line" if r == "deepseek" else "range")
            + dot(320, y1, r)
            + dot(705, y2, r)
        )
        art += line(305, left[r], 320, y1) + line(705, y2, 720, right[r])
        art += text(295, left[r] + 4, f"{r} ${c:.4f}", "end") + text(
            730, right[r] + 4, f"${w:.4f} {r}"
        )
        art += "</g>"
    result.append(
        figure(
            "What a warm cache does to cost",
            "100k input target, 100 output-token budget. Each line connects a route’s cold and warm request cost on a logarithmic scale: equal vertical distances represent equal cost ratios.",
            svg(
                "Cold versus warm normalized request cost, logarithmic scale",
                art,
                height=max(550, max(right.values()) + 35),
            ),
            "Same cost definition as the scatter plot, including output charges. Warm means a repeated prompt; the measured cache share determines how much was actually cached.",
        )
    )

    shapes = [
        "in1k_out100",
        "in1k_out1k",
        "in10k_out100",
        "in10k_out1k",
        "in100k_out100",
        "in100k_out1k",
    ]
    reliability = ""
    for cache in ["cold", "warm"]:
        art = ""
        for j, shape in enumerate(shapes):
            art += text(290 + j * 130, 24, shape.replace("in", "").replace("_out", "→"), "middle")
        for i, r in enumerate(routes):
            y = 55 + i * 31
            art += (
                interactive(r, r + " · " + cache + " requests")
                + text(220, y + 4, r, "end")
                + "</g>"
            )
            for j, shape in enumerate(shapes):
                c = key[shape, cache, r]
                x = 290 + j * 130
                rate = c["failures"] / c["n"]
                art += interactive(
                    r,
                    f"{r} · {shape} · {cache} · {c['failures']}/{c['n']} failures ({rate * 100:.1f}%)",
                )
                art += text(
                    x,
                    y + 4,
                    f"{rate * 100:.0f}%" if rate else "0",
                    "middle",
                    "fail" if rate else "",
                )
                art += "</g>"
        reliability += f"<h3>{cache.capitalize()} requests</h3>" + svg(
            f"{cache} failure percentages by route and request shape", art, height=490
        )
    result.append(
        figure(
            "Reliability by request shape",
            "Percentage of measured calls that failed. Columns show input-token targets → output-token budgets. Compare cold and warm separately; a route can behave differently when the same prompt asks for a longer response.",
            reliability,
            "Failure rate = failed calls ÷ measured calls in that condition. HTTP, transport and response errors count as failures. Exact counts and descriptive intervals are in the tables below. Zero means no observed failure in this sample.",
        )
    )

    primary_rows = [
        r for r in rows if r["source"] == "original" and r["route"] == "openrouter-auto"
    ]
    art = ""
    details = []
    for i, (shape, cache) in enumerate((s, c) for s in shapes for c in ["cold", "warm"]):
        rs = [r for r in primary_rows if r["shape"] == shape and r["cache_mode"] == cache]
        counts = Counter(r["provider_echo"] or "Not reported" for r in rs)
        y = 48 + i * 36
        x = 240
        label = shape.replace("in", "").replace("_out", "→") + " " + cache
        art += text(220, y + 17, label, "end")
        for n, (provider, count) in enumerate(sorted(counts.items(), key=lambda p: (-p[1], p[0]))):
            art += interactive(
                provider,
                f"{provider} · {label} · {count}/{len(rs)} calls ({count / len(rs) * 100:.1f}%)",
            )
            width = count / len(rs) * 650
            art += f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="25" fill="{["#396c6e", "#799694", "#b3c2bd", "#d5dcd7"][n % 4]}" stroke="#f6f6f3"><title>{esc(provider)}: {count}/{len(rs)} ({count / len(rs) * 100:.1f}%)</title></rect>'
            if width > len(provider) * 8 + 12:
                art += text(
                    x + width / 2,
                    y + 17,
                    provider,
                    "middle",
                    "segment-light" if n % 4 < 2 else "label",
                )
            art += "</g>"
            x += width
        art += text(905, y + 17, f"n={len(rs)}")
        details.append(
            "<tr><th>"
            + esc(label)
            + "</th><td>"
            + esc(
                ", ".join(
                    f"{p}: {n}" for p, n in sorted(counts.items(), key=lambda p: (-p[1], p[0]))
                )
            )
            + "</td></tr>"
        )
    for tick in [0, 25, 50, 75, 100]:
        art += text(240 + tick * 6.5, 22, f"{tick}%", "middle")
    result.append(
        figure(
            "Where automatic routing sent the calls",
            "Each bar shows the upstream names returned by OpenRouter for one request shape and cache condition. A repeated prompt may reach a different upstream, which can change its cache behavior.",
            svg(
                "OpenRouter upstream distribution by request shape and cache condition",
                art,
                height=500,
            ),
            "Segment width is the share of calls. Names are reported upstreams, not verified machines. Colours separate neighbouring segments; hover for every name and count. This distribution alone does not establish why routing changed.",
        )
        + '<details><summary>All upstream names and call counts</summary><div class="scroll"><table>'
        + "".join(details)
        + "</table></div></details>"
    )
    return "".join(result)


def render(manifest: dict, cells: list[dict], rows: list[dict]) -> str:
    esc = html.escape
    primary = [c for c in cells if c["source"] != "auto_followup"]
    by_key = {(c["shape"], c["cache_mode"], c["route"]): c for c in primary}
    cold = by_key["in100k_out100", "cold", "deepseek"]["cost_per_m_prompt"]
    warm = by_key["in100k_out100", "warm", "deepseek"]["cost_per_m_prompt"]
    shapes = sorted(
        {c["shape"] for c in primary},
        key=lambda s: (
            int(s.split("_")[0][2:].replace("k", "000")),
            int(s.split("_")[1][3:].replace("k", "000")),
        ),
    )
    leaders = []
    sections = []
    for shape in shapes:
        for cache in ("cold", "warm"):
            group = [c for c in primary if c["shape"] == shape and c["cache_mode"] == cache]
            speed = max(
                (c for c in group if c["decode_tps"] is not None), key=lambda c: c["decode_tps"]
            )
            first = min((c for c in group if c["ttft_s"] is not None), key=lambda c: c["ttft_s"])
            leaders.append(speed["route"])
            body = []
            for c in sorted(group, key=lambda c: c["route"]):
                lo, hi = c["failure_ci"]
                cost = f(c["cost_per_m_prompt"], 5)
                if c["cost_per_m_prompt"] is not None:
                    cost = "$" + cost
                cache_value = (
                    f(c["cache_share"] * 100, 1) + "%"
                    if c["cache_share"] is not None
                    else "not reported"
                )
                cost_samples = (
                    f"n={c['cost_calibration_samples']}"
                    if c.get("cost_calibration_samples")
                    else f"{c['priced_samples']}/{c['successes']} successes priced"
                )
                body.append(
                    f'<tr><th scope="row">{esc(c["route"])}</th>'
                    f"<td>{c['n']}</td><td>{c['failures']}/{c['n']}"
                    f"<small>{lo * 100:.1f}–{hi * 100:.1f}% interval</small></td>"
                    f"<td>{f(c['ttft_s'], 3)}<small>n={c['ttft_samples']}</small></td>"
                    f"<td>{f(c['ttft_p90_s'], 3)}</td>"
                    f"<td>{f(c['decode_tps'], 1)}<small>n={c['decode_samples']}</small></td>"
                    f"<td>{cache_value}<small>n={c['cache_samples']}</small></td>"
                    f"<td>{cost}<small>{cost_samples}</small></td></tr>"
                )
            label = shape.replace("in", "").replace("_out", " input / ") + " output · " + cache
            sections.append(
                f'<section class="cell" data-shape="{shape}" data-cache="{cache}">'
                f"<h3>{label}</h3><p>Lowest median first-token latency: "
                f"<b>{esc(first['route'])}</b>. Highest median decode speed: "
                f'<b>{esc(speed["route"])}</b>.</p><div class="scroll" tabindex="0" '
                f'role="region" aria-label="{label} measurements"><table>'
                "<thead><tr><th>Route</th><th>Included calls</th><th>Failures</th>"
                "<th>First token p50 · s</th><th>First token p90 · s</th><th>Decode · tok/s</th><th>Cache share</th>"
                "<th>Effective cost / 1M prompt tokens</th></tr></thead><tbody>"
                + "".join(body)
                + "</tbody></table></div></section>"
            )
    primary_n = sum(c["n"] for c in primary)
    return TEMPLATE.replace(
        "{{CONTENT}}",
        f"""
<header><a href="https://github.com/aktasbatuhan/compound">compound</a><nav><a href="#charts">Explore the results</a><a href="#experiment">Run your own experiment ↗</a></nav></header>
<main><h1>The same model.<br>Different serving tradeoffs.</h1>
<p class="intro">Where you send a prompt changes what you pay, how long you wait, and whether the call succeeds.</p>
<p>DeepSeek V4 Flash · 14 serving routes · {primary_n:,} measured calls</p>
<p class="setup">One controlled setup: 1k, 10k and 100k input-token targets, each with a 100 or 1k output-token budget. Temperature 0, reasoning off. Cold requests use a unique prefix; warm requests repeat the same prompt. These are serving measurements, not task-quality scores.</p>
<div class="finding"><h2>A repeated prompt can change the winner.</h2><p>DeepSeek’s 100k-input, 100-output-budget requests cost <strong>{cold / warm:.1f}× less</strong> warm than cold in this sample. Telnyx led median generation speed in <strong>{leaders.count("telnyx")}/{len(leaders)} conditions</strong>. The fastest first token depended on the request shape.</p></div>
<div id="charts">{charts(primary, rows)}</div>
<section id="measurements"><h2>The measurements</h2><p>Choose the request shape and cache condition. Input sizes are targets and output sizes are budgets; actual token usage is retained in the downloadable records.</p>
<div id="controls" hidden><label>Request shape <select id="shape">{"".join(f'<option value="{s}">{s.replace("in", "").replace("_out", " input / ")} output budget</option>' for s in shapes)}</select></label><label>Cache condition <select id="cache"><option value="cold">Cold</option><option value="warm">Warm</option></select></label><button id="all" type="button">Show all conditions</button></div>
<p class="note">Timings describe successful calls. n is the number of observations behind each metric. Failure intervals are descriptive 95% Wilson intervals. “Not reported” means no measurement, not zero.</p>{"".join(sections)}</section>
<section id="experiment"><h2>Which host wins on your prompts?</h2><p>A chat agent, a document pipeline and a batch job stress different parts of this comparison. Use Compound to measure the hosts you are considering with your own messages.</p>
<ol><li>Install Compound from the repository and set your provider API key.</li><li>Save your messages in <code>shapes.json</code>. Start with this structure and replace the sample prompt.</li><li>Preview the experiment, then add <code>--go</code> to execute it.</li></ol>
<pre><code>{{
  "my-workload": {{
    "messages": [{{"role": "user", "content": "Replace this with a representative prompt."}}],
    "max_tokens": 100
  }}
}}</code></pre>
<pre><code>uv sync --extra dev
# Discover available hosts for your chosen model
compound-bench providers YOUR_MODEL
# Compare automatic routing with a host from that list
compound-bench serving --model-or YOUR_MODEL \\
  --providers openrouter/auto,openrouter/HOST \\
  --shapes shapes.json --cache-mode both \\
  --reasoning-modes off --temperature 0 --reps 5</code></pre>
<p>The command previews the call count without spending. Replace <code>YOUR_MODEL</code> and <code>HOST</code> with discovery results, set <code>OPENROUTER_API_KEY</code> in <code>.env</code>, then add <code>--go</code>. Executed requests bill at provider rates; serving runs have no dollar cap.</p>
<p><a href="../../docs/serving/">Read the serving guide →</a> · <a href="https://github.com/aktasbatuhan/compound">Get Compound on GitHub ↗</a></p></section>
<section id="data"><h2>Explore the data</h2><p>Download per-call timings, token usage, status codes and reported costs, or use the computed summaries. The manifest records the run windows and source provenance.</p><p class="downloads"><a href="calls.jsonl.gz" download>Per-call measurements ↓</a><a href="summary.json" download>Summary data ↓</a><a href="manifest.json" download>Data manifest ↓</a></p>
<details><summary>How to read these measurements</summary><p>Routes describe the tested access path. Quantization suffixes are discovery labels. Calls were collected in controlled runs, so results describe this sample, not a provider-wide guarantee. Zero observed failures does not establish perfect reliability. Repeated calls are dependent observations, and runs at different times can see different capacity.</p><p>Cost uses the sum of request charges divided by the corresponding input-token total, multiplied by one million. It includes output charges. Telnyx appears in timing, cache and failure charts; its cost is not available in this report. The manifest contains full source selection details.</p></details></section></main><footer>Compound · <a href="https://github.com/aktasbatuhan/compound">Open source experiments for choosing where to run your model</a></footer>
""",
    )


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Same model, different serving tradeoffs · Compound</title><link rel="icon" href="data:,"><style>
:root{color-scheme:light;--ink:#17292d;--muted:#52656a;--line:#cbd7da;--accent:#136c69}*{box-sizing:border-box}body{margin:0;background:#f8fbfc;color:var(--ink);font:17px/1.6 system-ui,sans-serif}header,main,footer{max-width:1180px;margin:auto;padding:24px 32px}header{display:flex;justify-content:space-between;gap:24px;border-bottom:1px solid var(--line)}header>a{font-weight:750;font-size:22px;text-decoration:none}nav{display:flex;gap:24px;align-items:center}a{color:var(--accent);text-underline-offset:4px}h1{font-size:clamp(42px,6vw,78px);line-height:1.06;letter-spacing:-.055em;max-width:880px;margin:28px 0}h2{font-size:30px;line-height:1.2;letter-spacing:-.025em;margin:0 0 18px}h3{font-size:21px}p,li{max-width:78ch}.intro{font-size:22px;max-width:760px}.eyebrow{color:var(--muted);margin-top:40px}.finding{border-left:5px solid var(--accent);background:#e8f3f2;padding:28px 32px;margin:44px 0}.finding strong{color:#075c58}section{margin-top:56px}section.cell{margin-top:30px}.note{font-size:14px;color:var(--muted)}.scroll{overflow:auto;margin:20px 0;border:1px solid var(--line);border-radius:8px;background:white}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;text-align:left}th,td{padding:13px 15px;vertical-align:top;border-bottom:1px solid #e3eaec}thead th{background:#eaf0f2;font-size:13px}tbody th{white-space:nowrap}small{display:block;font-size:11px;color:var(--muted);white-space:nowrap}#controls{display:flex;gap:20px;align-items:end;flex-wrap:wrap}#controls[hidden],[hidden]{display:none!important}label{font-size:14px;display:grid;gap:6px}select,button{font:inherit;padding:10px 14px;background:white;color:var(--ink);border:1px solid #849c9f;border-radius:6px}button{cursor:pointer}pre{padding:22px;background:#eaf0f2;border-radius:8px;overflow:auto;font-size:14px}code{font-family:ui-monospace,monospace}.downloads{display:flex;gap:24px;flex-wrap:wrap}footer{border-top:1px solid var(--line);margin-top:56px;color:var(--muted);font-size:14px}:focus-visible{outline:3px solid var(--accent);outline-offset:3px}@media(max-width:600px){header,main,footer{padding:20px}header{flex-direction:column;gap:12px}nav{font-size:14px;gap:18px}.finding{padding:22px}h2{font-size:26px}.intro{font-size:19px}td,th{padding:10px}.downloads{gap:12px}}

body{background:#f6f6f3;color:#252c2b}h1,h2{font-family:Georgia,serif;font-weight:400;letter-spacing:-.035em}h2{font-size:36px}h1{margin-top:68px}.setup{color:var(--muted);font-size:16px}.finding{background:none;border-left:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:30px 0}.figure{margin:70px 0}.figure>p{font-size:18px;max-width:850px}.chart svg{width:100%;height:auto;display:block}.chart text{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#505c59}.chart .mark{fill:#286568}.chart .grid{stroke:#dfe3df;stroke-width:1}.chart .focus-line{stroke:#286568;stroke-width:2.5}.chart .range{stroke:#9ab0ad;stroke-width:2}.chart .label{fill:#252c2b}.chart .segment-light{fill:#fff}.chart .focal{fill:#286568;font-weight:600}.chart .fail{fill:#99564f}.chart a{cursor:pointer}.chart-scroll{overflow-x:auto}.chart-scroll svg{min-width:780px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}.panels h3{font:16px ui-monospace,monospace}.panels svg{width:100%}.scroll{border:0;border-radius:0;background:transparent}thead th{background:none}tbody tr:hover{background:#edf1ee}td,th{font-family:ui-monospace,monospace}details{font-size:15px;max-width:850px}summary{cursor:pointer}.metric{color:#52656a;font-size:14px!important}.chart-note{font-size:13px!important;color:#52656a}.figure h2{margin-bottom:12px}@media(max-width:700px){.panels{grid-template-columns:1fr}.figure{margin:48px 0}h1{margin-top:40px}.chart-scroll{padding-bottom:12px}.figure>p{font-size:16px}}

.provider{cursor:pointer}.provider line{pointer-events:stroke}.figure.has-provider .provider{opacity:.18}.figure.has-provider .provider.is-active{opacity:1}.provider.is-active text{font-weight:700;fill:#075c58}.provider.is-active .segment-light{fill:white}.provider.is-active .range{stroke:#075c58;stroke-width:3}.provider.is-active circle{stroke:#075c58;stroke-width:2}.provider:focus{outline:none}.provider:focus-visible{outline:2px solid #136c69;outline-offset:4px}#chart-tooltip{position:fixed;z-index:10;pointer-events:none;max-width:min(350px,calc(100vw - 24px));padding:10px 14px;background:#17292d;color:white;border-radius:5px;font:13px/1.5 ui-monospace,monospace;box-shadow:0 2px 10px #0002}
</style></head><body>{{CONTENT}}<script>
const shape=document.getElementById('shape'),cache=document.getElementById('cache');
function filter(){document.querySelectorAll('.cell').forEach(el=>{el.hidden=el.dataset.shape!==shape.value||el.dataset.cache!==cache.value;});}
document.getElementById('controls').hidden=false;shape.value='in10k_out100';cache.value='cold';shape.addEventListener('change',filter);cache.addEventListener('change',filter);document.getElementById('all').addEventListener('click',()=>document.querySelectorAll('.cell').forEach(el=>el.hidden=false));filter();

const tooltip=document.createElement('div');tooltip.id='chart-tooltip';tooltip.hidden=true;tooltip.setAttribute('role','tooltip');document.body.append(tooltip);
let current=null;
function clearProvider(){document.querySelectorAll('.has-provider,.is-active').forEach(el=>el.classList.remove('has-provider','is-active'));tooltip.hidden=true;if(current)current.removeAttribute('aria-describedby');current=null;}
function showProvider(el,event){
  clearProvider();current=el;const figure=el.closest('.figure');figure.classList.add('has-provider');
  figure.querySelectorAll('[data-provider]').forEach(item=>item.classList.toggle('is-active',item.dataset.provider===el.dataset.provider));
  tooltip.textContent=el.dataset.tip;tooltip.hidden=false;el.setAttribute('aria-describedby','chart-tooltip');
  const rect=el.getBoundingClientRect();const x=event?.clientX??Math.min(rect.right,innerWidth-24), y=event?.clientY??rect.top;
  tooltip.style.left=Math.max(12,Math.min(x+14,innerWidth-tooltip.offsetWidth-12))+'px';
  tooltip.style.top=Math.max(12,Math.min(y+14,innerHeight-tooltip.offsetHeight-12))+'px';
}
document.querySelectorAll('[data-provider]').forEach(el=>{
  // Remove native titles to avoid a second tooltip over the custom one.
  el.querySelectorAll('title').forEach(t=>t.remove());
  el.addEventListener('pointerenter',e=>showProvider(el,e));
  el.addEventListener('pointermove',e=>showProvider(el,e));
  el.addEventListener('pointerleave',()=>{if(current===el)clearProvider();});
  el.addEventListener('focus',()=>showProvider(el));
  el.addEventListener('blur',clearProvider);
  el.addEventListener('click',e=>showProvider(el,e));
});
document.addEventListener('keydown',e=>{if(e.key==='Escape')clearProvider();});
document.addEventListener('pointerdown',e=>{if(!e.target.closest('[data-provider]'))clearProvider();});
window.addEventListener('scroll',clearProvider,true);window.addEventListener('resize',clearProvider);
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-from", type=Path, help="maintainer: private artifact root")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument(
        "--check", action="store_true", help="verify generated outputs without writing"
    )
    args = parser.parse_args()
    if args.export_from:
        if args.check:
            parser.error("--check cannot be combined with --export-from")
        export(args.export_from, args.bundle)
    manifest, rows = load(args.bundle)
    cells = summarize(rows, manifest.get("cost_calibration"))
    outputs = {
        "summary.json": json.dumps(cells, indent=2) + "\n",
        "index.html": render(manifest, cells, rows),
    }
    # Portable copy: all downloads embedded; documentation links resolve on any domain.
    standalone = outputs["index.html"].replace(
        "../../docs/serving/",
        "https://github.com/aktasbatuhan/compound/blob/main/site/docs-src/serving.md",
    )
    for name, mime in [
        ("calls.jsonl.gz", "application/gzip"),
        ("manifest.json", "application/json"),
        ("summary.json", "application/json"),
    ]:
        payload = outputs[name].encode() if name in outputs else (args.bundle / name).read_bytes()
        uri = f"data:{mime};base64," + base64.b64encode(payload).decode()
        standalone = standalone.replace(
            f'href="{name}" download', f'href="{uri}" download="{name}"'
        )
    outputs["standalone.html"] = standalone
    for name, text in outputs.items():
        path = args.bundle / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f"stale generated report: {path}")
        else:
            path.write_text(text)
    print(
        f"Verified {len(rows):,} sanitized calls; {len(cells)} cells. Report: {args.bundle / 'index.html'}"
    )


if __name__ == "__main__":
    main()
