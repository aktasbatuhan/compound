"""Render a provider-sweep summary to a self-contained, theme-aware HTML page.

Three panels, all keyed on the decision a switch turns on:

* per-host profile radars (quality, reliability, speed, determinism, cost, TPS
  -- whichever of those the run measured), so a host's shape reads at a glance;
* success vs context window, one mark per task per host, coloured by how many
  trials passed, so a long-context weakness (shared or host-specific) is visible
  at a glance; and
* cost vs accuracy per host, so the value frontier reads directly.

The output inlines all CSS and SVG (no network), and styles both colour schemes.
Consumed by :mod:`compound.bench_report`; kept separate so the numbers and the
picture evolve independently.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

_CSS = """
:root{--bg:#f6f7f9;--surface:#fff;--ink:#191e27;--muted:#5c6675;--faint:#8a93a1;
--hair:#e6eaef;--grid:#eef1f4;--lane:#dfe4ea;
--p3:#15803d;--p2:#7cb342;--p1:#e8a33d;--p0:#c2410c;--dot:#3b6ea5;--dw:#c2410c}
@media(prefers-color-scheme:dark){:root{--bg:#0e1218;--surface:#151b23;--ink:#e6eaf0;
--muted:#9aa4b2;--faint:#6b7482;--hair:#232d36;--grid:#1b232b;--lane:#2a343d;
--p3:#4ec27f;--p2:#8bc34a;--p1:#f0b24a;--p0:#f0803f;--dot:#6fa8dc;--dw:#f0803f}}
:root[data-theme=light]{--bg:#f6f7f9;--surface:#fff;--ink:#191e27;--grid:#eef1f4}
:root[data-theme=dark]{--bg:#0e1218;--surface:#151b23;--ink:#e6eaf0;--grid:#1b232b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.wrap{max-width:980px;margin:0 auto;padding:2.2rem 1.3rem 4rem}
h1{font-size:1.35rem;margin:0 0 .3rem;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:.9rem;margin:0 0 1.3rem;max-width:56rem}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:10px;
padding:1rem 1.1rem;margin:0 0 1.2rem;overflow-x:auto}
svg.chart{width:100%;height:auto;min-width:640px;display:block}
.ttl{fill:var(--ink);font-size:14px;font-weight:700}.sub{fill:var(--faint);font-size:11px}
.tick{fill:var(--faint);font-size:10px}.axn{fill:var(--faint);font-size:10.5px;text-anchor:middle}
.grid{stroke:var(--grid);stroke-width:1}.lane{stroke:var(--lane);stroke-width:1}
.lab{fill:var(--muted);font-size:11px;font-family:ui-monospace,Menlo,monospace}
.lab.dw{fill:var(--dw);font-weight:700}
.acc{fill:var(--muted);font-size:12px;font-weight:700;font-family:ui-monospace,Menlo,monospace}
.pt{fill:var(--ink);font-size:11px;font-weight:600;font-family:ui-monospace,Menlo,monospace}
.legend{display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.8rem;color:var(--muted);
margin:.2rem 0 1.3rem;padding:.55rem 0;border-top:1px solid var(--hair);
border-bottom:1px solid var(--hair)}
.legend b{display:inline-block;width:11px;height:11px;border-radius:50%;
vertical-align:-1px;margin-right:.3rem}
code{background:var(--hair);padding:.1em .4em;border-radius:4px;font-size:.85em}
"""


def _colour(rate: float) -> str:
    if rate >= 0.999:
        return "var(--p3)"
    if rate >= 0.66:
        return "var(--p2)"
    if rate >= 0.33:
        return "var(--p1)"
    return "var(--p0)"


def _is_dw(host: str) -> bool:
    return "doubleword" in host


def _context_chart(rows: list[dict], hosts: list[str], accuracy: dict[str, float | None]) -> str:
    ctxs = [int(r["ctx_tokens"]) for r in rows if int(r["ctx_tokens"]) > 0]
    if not ctxs:
        return '<text x="0" y="20" class="sub">no context data</text>'
    cmin, cmax = min(ctxs) - 300, max(ctxs) + 300
    W, L, R, LANE, TOP, BOT = 900, 170, 120, 30, 54, 26
    pw = W - L - R
    H = TOP + LANE * len(hosts) + BOT

    def xof(c: float) -> float:
        return L + (c - cmin) / (cmax - cmin) * pw

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    s.append('<text x="0" y="20" class="ttl">Success vs context window</text>')
    s.append('<text x="0" y="38" class="sub">Each mark = one task. Colour = trials passed. '
             'Left to right = longer conversation (prompt tokens).</text>')
    tk = ((int(cmin) // 2000) + 1) * 2000
    while tk <= cmax:
        x = xof(tk)
        s.append(f'<line x1="{x:.0f}" y1="{TOP - 6}" x2="{x:.0f}" y2="{H - BOT}" class="grid"/>')
        s.append(f'<text x="{x:.0f}" y="{TOP - 10}" class="tick" '
                 f'text-anchor="middle">{tk // 1000}k</text>')
        tk += 2000
    s.append(f'<text x="{L + pw / 2:.0f}" y="{H - 6}" class="axn">'
             'task context (prompt tokens) →</text>')
    for i, host in enumerate(hosts):
        y = TOP + LANE * i + LANE / 2
        cls = "lab dw" if _is_dw(host) else "lab"
        s.append(f'<text x="{L - 10}" y="{y + 4:.0f}" class="{cls}" '
                 f'text-anchor="end">{html.escape(host)}</text>')
        s.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W - R}" y2="{y:.0f}" class="lane"/>')
        for r in [r for r in rows if r["host"] == host]:
            c = int(r["ctx_tokens"])
            if c <= 0:
                continue
            s.append(f'<circle cx="{xof(c):.0f}" cy="{y:.0f}" r="5.2" '
                     f'fill="{_colour(float(r["success_rate"]))}" '
                     'stroke="rgba(0,0,0,.15)" stroke-width=".5"/>')
        acc = accuracy.get(host)
        if acc is not None:
            s.append(f'<text x="{W - R + 8}" y="{y + 4:.0f}" class="acc">{acc * 100:.0f}%</text>')
    s.append("</svg>")
    return "\n".join(s)


def _cost_chart(hosts: list[str], summary: dict[str, Any]) -> str:
    pts = [(h, summary["hosts"][h]) for h in hosts
           if summary["hosts"][h]["accuracy"] is not None
           and summary["hosts"][h]["cost_per_task_usd"]]
    if len(pts) < 2:
        return '<text x="0" y="20" class="sub">insufficient cost data for the value chart</text>'
    W, H, L, R, T, B = 900, 360, 70, 130, 54, 44
    pw, ph = W - L - R, H - T - B
    costs = [s["cost_per_task_usd"] for _, s in pts]
    accs = [s["accuracy"] for _, s in pts]
    cmn, cmx = min(costs) * 0.9, max(costs) * 1.08
    amn, amx = min(accs) - 0.03, max(accs) + 0.03

    def cx(v: float) -> float:
        return L + (v - cmn) / (cmx - cmn) * pw

    def cy(v: float) -> float:
        return T + ph - (v - amn) / (amx - amn) * ph

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    s.append('<text x="0" y="20" class="ttl">Cost vs quality per host</text>')
    s.append('<text x="0" y="38" class="sub">Up = more accurate. Left = cheaper. '
             'Top-left is best value.</text>')
    for gy in [amn + (amx - amn) * f for f in (0.2, 0.4, 0.6, 0.8)]:
        s.append(f'<line x1="{L}" y1="{cy(gy):.0f}" x2="{W - R}" y2="{cy(gy):.0f}" class="grid"/>')
        s.append(f'<text x="{L - 8}" y="{cy(gy) + 3:.0f}" class="tick" '
                 f'text-anchor="end">{gy * 100:.0f}%</text>')
    s.append(f'<text x="{L + pw / 2:.0f}" y="{H - 8}" class="axn">cost per task (USD)</text>')
    for host, st in pts:
        x, y = cx(st["cost_per_task_usd"]), cy(st["accuracy"])
        fill = "var(--dw)" if _is_dw(host) else "var(--dot)"
        s.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" fill="{fill}"/>')
        s.append(f'<text x="{x:.0f}" y="{y - 12:.0f}" class="pt" '
                 f'text-anchor="middle">{html.escape(host)}</text>')
        s.append(f'<text x="{x:.0f}" y="{H - B + 16:.0f}" class="tick" '
                 f'text-anchor="middle">${st["cost_per_task_usd"]:.4f}</text>')
    s.append("</svg>")
    return "\n".join(s)


def radar_axes(summary: dict[str, Any], rows: list[dict]) -> dict[str, dict[str, float]]:
    """Raw per-host axis values for the profile radars, from whatever the run measured.

    Axes where no host has data (e.g. cost without ``--prices``) are dropped;
    determinism needs multi-trial data. Higher is always better here, so
    latency and cost enter inverted.
    """
    hosts = list(summary["hosts"])
    axes: dict[str, dict[str, float]] = {h: {} for h in hosts}
    for h in hosts:
        s = summary["hosts"][h]
        episodes = s.get("episodes") or 0
        clean = episodes - (s.get("infra_errors") or 0)
        if s.get("accuracy") is not None:
            axes[h]["quality"] = s["accuracy"]
        if episodes:
            axes[h]["reliability"] = clean / episodes
        if s.get("median_latency_s") and s.get("accuracy"):
            # a host with zero successes has no meaningful serving-speed
            # signal: failing instantly is not "fast"
            axes[h]["speed"] = 1 / s["median_latency_s"]
        if s.get("median_tps"):
            axes[h]["TPS"] = s["median_tps"]
        if s.get("cost_per_task_usd"):
            axes[h]["cost"] = 1 / s["cost_per_task_usd"]
        per = [r for r in rows if r["host"] == h and int(r["trials"]) > 1]
        if per:
            flips = sum(1 for r in per if 0 < int(r["solved"]) < int(r["trials"]))
            # a host that never solves anything is only "deterministic" at failing
            solved_any = any(int(r["solved"]) for r in per)
            axes[h]["determinism"] = (1 - flips / len(per)) if solved_any else 0.0
    # keep only axes at least two hosts can be compared on
    names = [a for a in ("quality", "reliability", "speed", "determinism", "cost", "TPS")
             if sum(1 for h in hosts if a in axes[h]) >= 2]
    return {h: {a: axes[h].get(a, 0.0) for a in names} for h in hosts}


def _radar_grid(summary: dict[str, Any], rows: list[dict], hosts: list[str]) -> str:
    """Small-multiple spider charts, one per host, min-max normalized per axis."""
    import math

    raw = radar_axes(summary, rows)
    names = list(next(iter(raw.values()), {}).keys())
    if len(names) < 3:
        return '<text x="0" y="20" class="sub">not enough measured axes for profiles</text>'
    norm: dict[str, dict[str, float]] = {}
    for a in names:
        vals = [raw[h][a] for h in hosts]
        lo, hi = min(vals), max(vals)
        for h in hosts:
            v = (raw[h][a] - lo) / (hi - lo) if hi > lo else 1.0
            norm.setdefault(h, {})[a] = 0.08 + 0.92 * v
    n_ax = len(names)
    cell_w, cell_h, r0 = 300, 252, 76
    cols = min(3, len(hosts))
    rows_n = (len(hosts) + cols - 1) // cols
    W, H = cell_w * cols, 44 + cell_h * rows_n
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    s.append('<text x="0" y="20" class="ttl">Provider profiles</text>')
    s.append('<text x="0" y="38" class="sub">Outer edge = best host on that axis '
             '(min-max normalized). Axes reflect what this run measured.</text>')
    spoke = lambda cx, cy, i, frac: (  # noqa: E731
        cx + r0 * frac * math.sin(2 * math.pi * i / n_ax),
        cy - r0 * frac * math.cos(2 * math.pi * i / n_ax),
    )
    for idx, host in enumerate(hosts):
        cx = (idx % cols) * cell_w + cell_w / 2
        cy = 44 + (idx // cols) * cell_h + 106
        for frac in (0.5, 1.0):
            d = " ".join(f"{x:.0f},{y:.0f}" for x, y in
                         (spoke(cx, cy, i, frac) for i in range(n_ax)))
            s.append(f'<polygon points="{d}" fill="none" class="grid"/>')
        for i, a in enumerate(names):
            x, y = spoke(cx, cy, i, 1.0)
            s.append(f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{x:.0f}" y2="{y:.0f}" class="grid"/>')
            ly = y + (11 if y > cy else -5)
            anchor = "middle" if abs(x - cx) < 12 else ("start" if x > cx else "end")
            s.append(f'<text x="{x:.0f}" y="{ly:.0f}" class="tick" '
                     f'text-anchor="{anchor}">{html.escape(a)}</text>')
        fill = "var(--dw)" if _is_dw(host) else "var(--dot)"
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                     (spoke(cx, cy, i, norm[host][a]) for i, a in enumerate(names)))
        s.append(f'<polygon points="{d}" fill="{fill}" fill-opacity=".28" '
                 f'stroke="{fill}" stroke-width="1.6"/>')
        cls = "lab dw" if _is_dw(host) else "lab"
        s.append(f'<text x="{cx:.0f}" y="{cy + r0 + 32:.0f}" class="{cls}" '
                 f'text-anchor="middle">{html.escape(host)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def render_charts(summary: dict[str, Any], report_dir: Path) -> Path:
    """Write ``report_dir/charts.html`` from the summary + per_task.csv."""
    rows = list(csv.DictReader((report_dir / "per_task.csv").open()))
    hosts = sorted(summary["hosts"], key=lambda h: -(summary["hosts"][h]["accuracy"] or 0))
    accuracy = {h: summary["hosts"][h]["accuracy"] for h in hosts}
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Provider sweep — charts</title>"
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">"
        "<h1>Same model, many hosts — provider sweep</h1>"
        '<p class="lede">Identical weights served by different hosts on the same task corpus. '
        "Doubleword rows are highlighted. Generated with Compound "
        "(github.com/aktasbatuhan/compound).</p>"
        '<div class="legend">'
        '<span><b style="background:var(--p3)"></b>all trials passed</span>'
        '<span><b style="background:var(--p2)"></b>most</span>'
        '<span><b style="background:var(--p1)"></b>some</span>'
        '<span><b style="background:var(--p0)"></b>none</span>'
        '<span><b style="background:var(--dw)"></b>Doubleword</span></div>'
        f'<div class="card">{_radar_grid(summary, rows, hosts)}</div>'
        f'<div class="card">{_context_chart(rows, hosts, accuracy)}</div>'
        f'<div class="card">{_cost_chart(hosts, summary)}</div>'
        "</div></body></html>"
    )
    path = report_dir / "charts.html"
    path.write_text(doc)
    return path
