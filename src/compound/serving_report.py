"""Portable serving reports. No network, credentials, or study-specific assumptions."""

# HTML/SVG templates stay readable as complete markup lines.
# ruff: noqa: E501
from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# Only measurements are exported. Request/response text and raw errors are private.
STRINGS = (
    "iso",
    "route",
    "model",
    "mode",
    "shape",
    "cache_mode",
    "provider_echo",
    "experiment",
    "reuse_policy",
    "phase",
    "run_id",
    "prompt_sha256",
)
NUMBERS = (
    "idle_s",
    "actual_idle_s",
    "concurrency",
    "trial",
    "max_tokens",
    "temperature",
    "status",
    "round",
    "rep",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "cost_usd",
    "ttft_s",
    "total_s",
    "decode_tps",
)
CONDITION = (
    "shape",
    "mode",
    "temperature",
    "max_tokens",
    "experiment",
    "reuse_policy",
    "idle_s",
    "concurrency",
)
ARM = ("route", "model", "cache_marked")


def read_results(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    rows, sources, seen = [], [], set()
    for index, path in enumerate(paths):
        if path.resolve() in seen:
            raise ValueError(f"duplicate input: {path}")
        seen.add(path.resolve())
        data = path.read_bytes()
        decoded = gzip.decompress(data) if path.suffix == ".gz" else data
        count = 0
        for lineno, line in enumerate(decoded.decode().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                raw = json.loads(line)
            except ValueError:
                raise ValueError(f"{where}: invalid JSON") from None
            if not isinstance(raw, dict):
                raise ValueError(f"{where}: expected a serving record object")
            for key in ("route", "shape"):
                if not isinstance(raw.get(key), str) or not raw[key].strip():
                    raise ValueError(f"{where}: missing {key}")
            row = {}
            for key in STRINGS:
                value = raw.get(key)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{where}: {key} must be text or null")
                row[key] = value
            for key in NUMBERS:
                value = raw.get(key)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{where}: {key} must be a finite nonnegative number or null")
                row[key] = value
            if row["status"] is not None and (
                int(row["status"]) != row["status"] or not 100 <= row["status"] <= 599
            ):
                raise ValueError(f"{where}: invalid HTTP status")
            for key in (
                "max_tokens",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "cache_write_tokens",
                "round",
                "rep",
            ):
                if row[key] is not None and int(row[key]) != row[key]:
                    raise ValueError(f"{where}: {key} must be an integer")
            if (
                row["cached_tokens"] is not None
                and row["prompt_tokens"] is not None
                and row["cached_tokens"] > row["prompt_tokens"]
            ):
                raise ValueError(f"{where}: cached_tokens exceeds prompt_tokens")
            marker = raw.get("cache_marked")
            if marker is not None and not isinstance(marker, bool):
                raise ValueError(f"{where}: cache_marked must be boolean or null")
            row["cache_marked"] = marker
            row["error_class"] = (
                f"http_{row['status']}"
                if row["status"] is not None and row["status"] != 200
                else "transport_or_response_error"
                if raw.get("error") or raw.get("error_class")
                else None
            )
            row.update(source=index + 1, source_line=lineno)
            rows.append(row)
            count += 1
        sources.append({"id": index + 1, "sha256": hashlib.sha256(data).hexdigest(), "rows": count})
    if not rows:
        raise ValueError("no serving records found")
    return rows, sources


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def wilson(k: int, n: int) -> tuple[float, float]:
    p, z = k / n, 1.96
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, center - half), min(1, center + half)


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    keys = (*CONDITION, *ARM, "cache_mode")
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    cells = []
    for values, records in groups.items():
        good = [r for r in records if r["status"] == 200 and not r["error_class"]]
        c = dict(zip(keys, values, strict=True))
        c.update(n=len(records), successes=len(good), failures=len(records) - len(good))
        c["failure_rate"] = c["failures"] / c["n"]
        c["failure_ci"] = wilson(c["failures"], c["n"])
        for field in ("ttft_s", "decode_tps", "total_s"):
            nums = [r[field] for r in good if r[field] is not None]
            c[field] = statistics.median(nums) if nums else None
            c[field + "_samples"] = len(nums)
            c[field + "_p90"] = quantile(nums, 0.9)
        priced = [r for r in good if r["cost_usd"] is not None and r["prompt_tokens"] is not None]
        cached = [
            r for r in good if r["cached_tokens"] is not None and r["prompt_tokens"] is not None
        ]
        for selected, numerator, field in (
            (priced, "cost_usd", "cost_per_m_prompt"),
            (cached, "cached_tokens", "cache_share"),
        ):
            denominator = sum(r["prompt_tokens"] for r in selected)
            c[field] = sum(r[numerator] for r in selected) / denominator if denominator else None
            c[field + "_samples"] = len(selected)
        if c["cost_per_m_prompt"] is not None:
            c["cost_per_m_prompt"] *= 1e6
        c["upstreams"] = dict(Counter(r["provider_echo"] or "Not reported" for r in records))
        cells.append(c)
    return cells


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt(value, digits=2) -> str:
    return "Not reported" if value is None else f"{value:,.{digits}f}"


def svg_text(x, y, value, anchor="start"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}">{esc(value)}</text>'


def line(x1, y1, x2, y2, cls="grid"):
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'


def dot(x, y):
    return f'<circle cx="{x}" cy="{y}" r="5"/>'


def arm(c):
    return tuple(c[k] for k in ARM)


def group(c, tip, content, ids):
    return (
        f'<g tabindex="0" role="group" class="provider" data-provider="{ids[arm(c)]}" '
        f'data-tip="{esc(tip)}" aria-label="{esc(tip)}">{content}</g>'
    )


def figure(title, description, content, height=350):
    return (
        f'<section class="figure"><h3>{esc(title)}</h3><p>{description}</p>'
        f'<div class="scroll"><svg viewBox="0 0 960 {height}" role="group" '
        f'aria-label="{esc(title)}">{content}</svg></div></section>'
    )


def plot(cells, field, title, description, ids, scale=1, unit="", interval=None):
    available = [c for c in cells if c[field] is not None]
    if not available:
        return f'<section class="figure"><h3>{esc(title)}</h3><p>{description}</p><p>No measurements reported for this metric.</p></section>'
    available.sort(key=lambda c: c[field], reverse=field in ("decode_tps", "cache_share"))
    maximum = max((c[interval] if interval else c[field]) * scale for c in available) or 1
    if field in ("cache_share", "failure_rate"):
        maximum = 100
    height = 90 + len(available) * 34
    content = ""
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = 280 + fraction * 520
        content += line(x, 38, x, height - 25) + svg_text(
            x, 23, fmt(fraction * maximum) + unit, "middle"
        )
    for i, c in enumerate(available):
        y = 60 + i * 34
        x = 280 + c[field] * scale / maximum * 520
        label = f"{ids[arm(c)]}. {c['route']}"
        tip = f"{label} · {title}: {fmt(c[field] * scale)}{unit} · {c['n']} calls, {c['successes']} successes"
        marks = svg_text(260, y + 5, label if len(label) < 29 else label[:26] + "…", "end")
        if interval:
            marks += line(x, y, 280 + c[interval] * scale / maximum * 520, y, "range")
            tip += f" · p90 {fmt(c[interval] * scale)}{unit}"
        if field == "failure_rate":
            lo, hi = c["failure_ci"]
            marks += line(280 + lo * 520, y, 280 + hi * 520, y, "range")
            tip += f" · 95% Wilson interval {lo:.1%} to {hi:.1%}"
        marks += dot(x, y) + svg_text(820, y + 5, fmt(c[field] * scale) + unit)
        content += group(c, tip, marks, ids)
    return figure(title, description, content, height)


def scatter(cells, ids):
    available = [c for c in cells if c["ttft_s"] is not None and c["cost_per_m_prompt"] is not None]
    if not available:
        return '<section class="figure"><h3>Cost and first-token latency</h3><p>No provider has both cost and first-token measurements in this condition.</p></section>'
    xmax = max(c["cost_per_m_prompt"] for c in available) * 1.1 or 1
    ymax = max(c["ttft_s"] for c in available) * 1.1 or 1
    out = ""
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x, y = 100 + fraction * 740, 340 - fraction * 290
        out += line(x, 50, x, 340) + svg_text(x, 365, "$" + fmt(fraction * xmax, 3), "middle")
        out += line(100, y, 840, y) + svg_text(85, y + 5, fmt(fraction * ymax) + "s", "end")
    out += svg_text(470, 402, "Request cost per million input tokens →", "middle")
    out += svg_text(100, 24, "Median time to first token (seconds); lower is faster")
    for c in available:
        x, y = 100 + c["cost_per_m_prompt"] / xmax * 740, 340 - c["ttft_s"] / ymax * 290
        tip = f"{c['route']} · ${fmt(c['cost_per_m_prompt'], 4)}/1M input · TTFT {fmt(c['ttft_s'])}s · {c['n']} calls"
        out += group(c, tip, dot(x, y) + svg_text(x + 9, y - 9, ids[arm(c)]), ids)
    return figure(
        "Cost and first-token latency",
        "Lower-left means cheaper and faster to start. Numbers match the provider list. Cost and timing may cover different subsets of successful calls; see coverage below.",
        out,
        425,
    )


def cache_change(cells, ids):
    phases = (
        ("prime", "probe")
        if any(c.get("experiment") == "cache-study" for c in cells)
        else ("cold", "warm")
    )
    before, after = phases
    pairs = defaultdict(dict)
    for c in cells:
        if c["cache_mode"] in phases:
            pairs[arm(c)][c["cache_mode"]] = c
    out, i = "", 0
    available = [
        p
        for p in pairs.values()
        if all(k in p and p[k]["cost_per_m_prompt"] is not None for k in phases)
    ]
    maximum = max((p[k]["cost_per_m_prompt"] for p in available for k in phases), default=1) or 1
    for p in available:
        c, w = p[before], p[after]
        y = 65 + i * 36
        a, b = (300 + p[k]["cost_per_m_prompt"] / maximum * 350 for k in phases)
        label = f"{ids[arm(c)]}. {c['route']}"
        marks = svg_text(280, y + 5, label[:28], "end") + line(a, y, b, y, "range")
        marks += f'<circle class="cold" cx="{a}" cy="{y}" r="5"/>' + dot(b, y)
        marks += svg_text(
            720,
            y + 5,
            "$" + fmt(c["cost_per_m_prompt"], 4) + " → $" + fmt(w["cost_per_m_prompt"], 4),
        )
        out += group(
            c,
            label
            + f" · {before} $"
            + fmt(c["cost_per_m_prompt"], 4)
            + f" · {after} $"
            + fmt(w["cost_per_m_prompt"], 4),
            marks,
            ids,
        )
        i += 1
    if not available:
        return f'<section class="figure"><h3>Cache cost impact</h3><p>Needs reported costs for both {before} and {after} calls from the same provider, model and marker setting.</p></section>'
    out += svg_text(
        300, 25, f"{before.title()} ○ → {after} ●; linear cost scale from $0 to $" + fmt(maximum, 4)
    )
    return figure(
        "Cache cost impact",
        f"Request cost per million input tokens, including output charges. Compares {before} and {after} calls for matching workload settings. Priming calls are separate in cache studies; ordinary warm serving groups include their first call.",
        out,
        100 + i * 36,
    )


def render(rows, cells, manifest, title):
    cohorts = defaultdict(list)
    ids = {a: i + 1 for i, a in enumerate(dict.fromkeys(arm(c) for c in cells))}
    for c in cells:
        cohorts[tuple(c[k] for k in (*CONDITION, "cache_mode"))].append(c)
    sections, options = [], []
    labels = (
        "Workload",
        "Reasoning",
        "Temperature",
        "Output budget",
        "Experiment",
        "Prefix reuse",
        "Idle seconds",
        "Burst size",
        "Phase / cache",
    )
    for index, (condition, selected) in enumerate(cohorts.items()):
        label = " · ".join(
            f"{k}: {v if v is not None else 'not recorded'}"
            for k, v in zip(labels, condition, strict=True)
            if v is not None
            or k in ("Workload", "Reasoning", "Temperature", "Output budget", "Phase / cache")
        )
        options.append(f'<option value="condition-{index}">{esc(label)}</option>')
        body = f'<h2>{esc(condition[0])}</h2><p>{esc(label)}</p><ul class="legend">'
        for c in selected:
            number = ids[arm(c)]
            provider = f"{number}. {c['route']} · model: {c['model'] or 'not recorded'} · cache markers: {'not recorded' if c['cache_marked'] is None else 'yes' if c['cache_marked'] else 'no'}"
            body += f'<li tabindex="0" class="provider" data-provider="{number}" data-tip="{esc(provider)}">{esc(provider)}</li>'
        body += "</ul>" + scatter(selected, ids)
        body += plot(
            selected,
            "ttft_s",
            "Time to first token",
            "Median wait for the first streamed token; line extends to the 90th percentile. Successful calls with timing only. Percentiles use linear interpolation, not a confidence interval.",
            ids,
            unit="s",
            interval="ttft_s_p90",
        )
        body += plot(
            selected,
            "decode_tps",
            "Generation speed",
            "Median output tokens per second after the first token. Successful calls with reported decode speed only; this is not end-to-end request throughput.",
            ids,
            unit=" tok/s",
        )
        body += plot(
            selected,
            "cache_share",
            "Cache token share",
            "Cached input tokens divided by input tokens across successful calls reporting both counts. This is token share, not the percentage of requests that hit cache. Warm includes its first call.",
            ids,
            scale=100,
            unit="%",
        )
        peers = [c for c in cells if tuple(c[k] for k in CONDITION) == condition[:-1]]
        body += cache_change(peers, ids)
        body += plot(
            selected,
            "failure_rate",
            "Failures",
            "Calls without a successful HTTP 200 response, including transport and response errors. Lines show 95% Wilson intervals. All supplied calls count, including credit errors; no task-quality evaluation is performed.",
            ids,
            scale=100,
            unit="%",
        )
        body += '<section class="figure"><h3>Reported upstreams</h3><p>Provider echoes across all calls. Missing echoes stay visible. These names do not verify quantization or explain cache behavior.</p><div class="scroll"><table><thead><tr><th>Route / model</th><th>Upstream</th><th>Calls</th><th>Share</th></tr></thead><tbody>'
        for c in selected:
            for upstream, n in sorted(c["upstreams"].items(), key=lambda item: -item[1]):
                body += f'<tr class="provider" tabindex="0" data-provider="{ids[arm(c)]}"><th>{esc(c["route"])} / {esc(c["model"])}</th><td>{esc(upstream)}</td><td>{n}</td><td>{n / c["n"]:.1%}</td></tr>'
        body += '</tbody></table></div></section><section><h3>Values and measurement coverage</h3><p>Each metric lists its measured sample count. Cost is total reported request charges divided by input tokens for those same successful calls, multiplied by one million. It includes output charges and is not an input-token rate card. Failed-call charges are not included. Missing values are not zero.</p><div class="scroll"><table><thead><tr><th>Provider</th><th>Calls / successful</th><th>TTFT p50 / p90 (s); n</th><th>Decode tok/s; n</th><th>Cache share; n</th><th>$/1M input; n</th></tr></thead><tbody>'
        for c in selected:
            body += f'<tr tabindex="0" class="provider" data-provider="{ids[arm(c)]}"><th>{ids[arm(c)]}. {esc(c["route"])}</th><td>{c["n"]} / {c["successes"]}</td>'
            body += f"<td>{fmt(c['ttft_s'])} / {fmt(c['ttft_s_p90'])}; {c['ttft_s_samples']}</td><td>{fmt(c['decode_tps'])}; {c['decode_tps_samples']}</td>"
            body += f"<td>{fmt(c['cache_share'] * 100 if c['cache_share'] is not None else None)}%; {c['cache_share_samples']}</td><td>{fmt(c['cost_per_m_prompt'], 4)}; {c['cost_per_m_prompt_samples']}</td></tr>"
        body += "</tbody></table></div></section>"
        sections.append(f'<section class="condition" id="condition-{index}">{body}</section>')
    downloads = ""
    for name, data, mime in (
        (
            "calls.jsonl.gz",
            gzip.compress(
                ("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)).encode(), mtime=0
            ),
            "application/gzip",
        ),
        ("summary.json", json.dumps(cells, indent=2).encode(), "application/json"),
        ("manifest.json", json.dumps(manifest, indent=2).encode(), "application/json"),
    ):
        downloads += f'<a download="{name}" href="data:{mime};base64,{base64.b64encode(data).decode()}">{name}</a> '
    template = Path(__file__).with_name("serving_report.html").read_text()
    replacements = {
        "TITLE": esc(title),
        "COUNT": f"{len(rows):,}",
        "DOWNLOADS": downloads,
        "OPTIONS": "".join(options),
        "SECTIONS": "".join(sections),
    }
    # One pass: user labels must never be interpreted as template directives.

    return re.sub(
        r"@@(TITLE|COUNT|DOWNLOADS|OPTIONS|SECTIONS)@@", lambda m: replacements[m[1]], template
    )


def build_report(
    paths: list[Path], output: Path, *, title="Serving comparison", force=False
) -> Path:
    if output.resolve() in {p.resolve() for p in paths}:
        raise ValueError("output must not overwrite an input")
    rows, sources = read_results(paths)
    cells = aggregate(rows)
    manifest = {
        "schema_version": 1,
        "included_rows": len(rows),
        "sources": sources,
        "grouping": list((*CONDITION, *ARM, "cache_mode")),
        "method": "All supplied records included. Identically named shapes across files are pooled within matching conditions. Model IDs and marker settings remain separate provider arms. No task quality measured.",
    }
    document = render(rows, cells, manifest, title)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if force else "x", encoding="utf-8") as f:
        f.write(document)
    return output
