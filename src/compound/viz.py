"""General-purpose route-comparison reports: cost vs quality vs speed.

This is the repo's data-viz library. It knows nothing about tau-bench or any
other evaluation source; it renders ONE self-contained HTML report from a list
of route rows, where a route is "a model served somewhere" and the axes are
quality, cost, and latency. Anything that can emit the row contract gets the
report: benchmark adapters (see `compound.tau_report`), the TS engine's
telemetry, or your own script.

Row contract (JSON object per route):

    {
      "model":   "glm-5.2",            # display key; also the filter identity
      "host":    "baseten/fp8",        # serving route; provider = text before "/"
      "quality": 0.846,                # 0..1, the y-axis
      "quality_num": 11, "quality_den": 13,   # optional "solved n/N" display
      "cost":    0.0854,               # per-unit cost, the x-axis (log)
      "lat_p50": 4.4, "lat_p95": 14.5, # seconds, optional
      "tps":     36.3,                 # optional
      "quant":   "fp8",                # optional
      "served":  ["BaseTen"],          # optional: verified serving hosts
      "flagged": false                 # optional: mark provisional rows
    }

CLI:  python -m compound.viz --rows rows.json --output report.html \
          --title "..." --note "..." --cost-label "cost per episode (USD)"

The report: an efficient-frontier scatter (Pareto line), a speed-vs-quality
scatter, per-model filter pills, and the full route table. Provider logos are
fetched as favicons at render time and embedded as data URIs; providers with
no reachable logo fall back to initials, so rendering works offline and no
third-party binaries enter the repo. Light and dark themes are both first-class.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

#: Known provider -> homepage domain, used only to fetch a favicon.
PROVIDER_DOMAINS = {
    "baseten": "baseten.co", "chutes": "chutes.ai", "cloudflare": "cloudflare.com",
    "decart": "decart.ai", "deepinfra": "deepinfra.com", "deepseek": "deepseek.com",
    "digitalocean": "digitalocean.com", "doubleword": "doubleword.ai",
    "fireworks": "fireworks.ai", "gmicloud": "gmicloud.ai", "groq": "groq.com",
    "modal": "modal.com", "moonshotai": "moonshot.ai", "morph": "morphllm.com",
    "novita": "novita.ai", "openai": "openai.com", "parasail": "parasail.io",
    "together": "together.ai", "wafer": "wafer.ai", "z-ai": "z.ai",
    "cerebras": "cerebras.ai", "anthropic": "anthropic.com",
    "ionstream": "ionstream.ai",
}
MODEL_PALETTE = [("#3358d4", "#7d9bff"), ("#177a63", "#4fbfa2"), ("#c2571f", "#e8895a"),
                 ("#8e44ad", "#c39bd3"), ("#b23b2e", "#e8735f"), ("#1c6e8c", "#5ab6d8")]


@dataclass(slots=True)
class Route:
    model: str
    host: str
    quality: float
    cost: float
    quality_num: int | None = None
    quality_den: int | None = None
    lat_p50: float | None = None
    lat_p95: float | None = None
    tps: float | None = None
    quant: str = "unknown"
    served: list[str] = field(default_factory=list)
    flagged: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Route":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def provider(self) -> str:
        return self.host.split("/")[0]

    @property
    def quality_label(self) -> str:
        if self.quality_num is not None and self.quality_den is not None:
            return f"{self.quality_num}/{self.quality_den}"
        return f"{self.quality:.0%}"


def fetch_logos(providers: list[str], cache_dir: Path) -> dict[str, str]:
    """Favicon per provider as a data URI; silently skips the unreachable."""
    logos: dict[str, str] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in providers:
        cached = cache_dir / f"{p}.png"
        if not cached.exists():
            domain = PROVIDER_DOMAINS.get(p)
            if domain is None:
                continue
            try:
                url = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
                cached.write_bytes(urllib.request.urlopen(url, timeout=8).read())
            except Exception:
                continue
        raw = cached.read_bytes()
        if len(raw) < 150:  # the generic-globe fallback is ~100B; initials look better
            continue
        mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
        logos[p] = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    return logos


def render(
    rows: list[Route],
    logos: dict[str, str],
    *,
    title: str,
    note: str,
    cost_label: str = "cost per unit (USD, log scale)",
    latency_label: str = "median latency (seconds, log scale)",
    quality_axis: str | None = None,
    subtitle: str | None = None,
) -> str:
    models = sorted({r.model for r in rows})
    providers = sorted({r.provider for r in rows})
    mcolor = {m: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, m in enumerate(models)}
    dens = {r.quality_den for r in rows if r.quality_den}
    y_axis = quality_axis or (f"quality (of {dens.pop()})" if len(dens) == 1 else "quality")

    def mvar(model: str) -> str:
        return f"var(--m-{models.index(model)})"

    def header_svg(x0: float) -> str:
        """In-SVG model legend + optional subtitle, so a screenshot of a single
        chart is self-explanatory: which ring color is which model, on what."""
        parts = ['<g class="hdr">']
        lx = x0
        for m in models:
            parts.append(f'<circle cx="{lx + 6:.0f}" cy="22" r="6" fill="none" '
                         f'stroke="{mvar(m)}" stroke-width="2.5"/>')
            parts.append(f'<text x="{lx + 18:.0f}" y="26" class="lgd" '
                         f'fill="var(--ink)">{html.escape(m)}</text>')
            lx += 18 + 7.2 * len(m) + 28
        if subtitle:
            parts.append(f'<text x="{x0}" y="50" class="subtitle" '
                         f'fill="var(--ink-soft)">{html.escape(subtitle)}</text>')
        parts.append("</g>")
        return "".join(parts)

    def marker(r: Route, cx: float, cy: float, tip: str) -> str:
        t = f"<title>{html.escape(tip)}</title>"
        if r.provider in logos:
            img = f'<image href="{logos[r.provider]}" x="{cx-8:.1f}" y="{cy-8:.1f}" width="16" height="16"/>'
        else:
            img = (f'<text x="{cx:.1f}" y="{cy+3.5:.1f}" text-anchor="middle" '
                   f'class="pt-label" fill="var(--ink)">{html.escape(r.provider[:2])}</text>')
        return (f'<g class="pt" data-m="{r.model}">{t}<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" '
                f'fill="var(--panel2)" stroke="{mvar(r.model)}" stroke-width="2.5"/>{img}</g>')

    cost_lo = min(r.cost for r in rows) * 0.7
    cost_hi = max(r.cost for r in rows) * 1.4
    q_lo = max(0.0, min(r.quality for r in rows) - 0.06)
    q_hi = min(1.035, max(r.quality for r in rows) + 0.05)

    def sy(q, y0, y1):
        return y1 - (q - q_lo) / (q_hi - q_lo) * (y1 - y0)

    def logfn(lo, hi):
        def f(v, x0, x1):
            v = min(max(v, lo), hi)
            return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
        return f

    lat_vals = [r.lat_p50 for r in rows if r.lat_p50]
    lat_lo = min(lat_vals) * 0.8 if lat_vals else 1.0
    lat_hi = max(lat_vals) * 1.25 if lat_vals else 10.0

    def ticks_log(lo, hi, fmt):
        raw = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 100]
        return [(v, fmt(v)) for v in raw if lo <= v <= hi]

    def scatter(sel, W, H, xfn, xticks, xlabel, xval, frontier=False, chart_id=""):
        x0, x1, y0, y1 = 70, W - 24, 88, H - 64
        out = [header_svg(x0)]
        for i in range(6):
            q = q_lo + i * (q_hi - q_lo) / 5
            y = sy(q, y0, y1)
            out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
            out.append(f'<text x="{x0-10}" y="{y+4:.1f}" class="tick" text-anchor="end">{q:.0%}</text>')
        for v, lab in xticks:
            x = xfn(v, x0, x1)
            out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>')
            out.append(f'<text x="{x:.1f}" y="{y1+22}" class="tick" text-anchor="middle">{lab}</text>')
        out.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-12}" class="axis" text-anchor="middle">{xlabel}</text>')
        out.append(f'<text x="20" y="{(y0+y1)/2:.0f}" class="axis" text-anchor="middle" '
                   f'transform="rotate(-90 20 {(y0+y1)/2:.0f})">{html.escape(y_axis)}</text>')
        if frontier:
            best = -1.0
            fr = []
            for r in sorted(sel, key=lambda r: r.cost):
                if r.quality > best:
                    best = r.quality
                    fr.append(r)
            d = " ".join(f'{"M" if i == 0 else "L"} {xfn(r.cost, x0, x1):.1f} {sy(r.quality, y0, y1):.1f}'
                         for i, r in enumerate(fr))
            out.append(f'<path d="{d}" class="frontier"/>')
        pts = sorted(sel, key=lambda r: (-r.quality, r.cost))
        coords = [(xfn(xval(r), x0, x1), sy(r.quality, y0, y1)) for r in pts]
        labels: list[tuple[float, float, float]] = []

        def collides(lx_, rx_, ly_):
            return any(lx_ < bx1 and rx_ > bx0 and abs(ly_ - by) < 12 for bx0, bx1, by in labels) or any(
                lx_ < mx + 12 and rx_ > mx - 12 and abs(ly_ - my) < 14 for mx, my in coords)

        body = []
        for r, (cx, cy) in zip(pts, coords):
            tip = (f'{r.model} @ {r.host}: {r.quality_label}, ${r.cost:.4f}, '
                   f'p50 {r.lat_p50}s, {r.tps} tps, quant {r.quant}')
            body.append(marker(r, cx, cy, tip))
            host = r.host.replace("doubleword/", "dw/")
            w = 6.4 * len(host)
            pick = None
            for tx, ty, anc in ((cx + 16, cy + 4, "start"), (cx - 16, cy + 4, "end"),
                                (cx, cy - 17, "middle"), (cx, cy + 24, "middle")):
                left = tx - (w if anc == "end" else w / 2 if anc == "middle" else 0)
                if left < x0 or left + w > x1 or collides(left, left + w, ty):
                    continue
                pick = (tx, ty, anc, left)
                break
            if pick is None:
                tx, anc = (cx + 16, "start") if cx + 16 + w <= x1 else (cx - 16, "end")
                left = tx - (w if anc == "end" else 0)
                ty = cy + 4
                while collides(left, left + w, ty):
                    ty += 12
                pick = (tx, ty, anc, left)
            tx, ty, anc, left = pick
            labels.append((left, left + w, ty))
            body.append(f'<text x="{tx:.1f}" y="{ty:.1f}" class="pt-label" data-m="{r.model}" '
                        f'text-anchor="{anc}" fill="{mvar(r.model)}">{html.escape(host)}</text>')
        return (f'<svg id="{chart_id}" viewBox="0 0 {W} {H}" role="img">'
                + "".join(out) + "".join(body) + "</svg>")

    cost_chart = scatter(rows, 980, 658, logfn(cost_lo, cost_hi),
                         ticks_log(cost_lo, cost_hi, lambda v: f"${v:g}"),
                         cost_label, lambda r: r.cost, frontier=True, chart_id="svg-cost")
    lat_rows = [r for r in rows if r.lat_p50]
    lat_chart = scatter(lat_rows, 980, 658, logfn(lat_lo, lat_hi),
                        ticks_log(lat_lo, lat_hi, lambda v: f"{v:g}s"),
                        latency_label, lambda r: r.lat_p50, chart_id="svg-speed") if lat_rows else ""

    def logo_img(p, size=15):
        return (f'<img src="{logos[p]}" width="{size}" height="{size}" alt="" '
                f'style="vertical-align:-3px;border-radius:3px;margin-right:7px">') if p in logos else ""

    def ring(m):
        return (f'<svg width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" '
                f'fill="none" stroke="{mvar(m)}" stroke-width="2.5"/></svg>')

    table_rows = "".join(
        f"<tr><td><span class='sh'>{ring(r.model)}</span>{r.model}</td>"
        f"<td>{logo_img(r.provider)}{r.host}</td><td>{r.quant}</td>"
        f"<td class='r'><b>{r.quality_label}</b></td><td class='r'>${r.cost:.4f}</td>"
        f"<td class='r'>{r.lat_p50}s</td><td class='r'>{r.lat_p95}s</td><td class='r'>{r.tps}</td>"
        f"<td>{'&check;' if r.served else ('mode' if r.host.startswith('doubleword') else '?')}</td></tr>"
        for r in sorted(rows, key=lambda r: (-r.quality, r.cost)))

    mvars_light = " ".join(f"--m-{i}:{mcolor[m][0]};" for i, m in enumerate(models))
    mvars_dark = " ".join(f"--m-{i}:{mcolor[m][1]};" for i, m in enumerate(models))
    model_legend = "".join(f'<span class="lg">{ring(m)}{m}</span>' for m in models)
    prov_legend = "".join(f'<span class="lg">{logo_img(p, 14)}{p}</span>' for p in providers)
    pills = '<button class="fbtn" data-f="all" aria-pressed="true">All models</button>' + "".join(
        f'<button class="fbtn" data-f="{m}" aria-pressed="false">{m}</button>' for m in models)
    dim_css = ",\n".join(f'body[data-filter="{m}"] [data-m]:not([data-m="{m}"])' for m in models)

    return f"""<meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
:root {{ --paper:#f6f4ee; --panel:#efede5; --panel2:#fffdf7; --ink:#1c1e22; --ink-soft:#565b64;
  --ink-faint:#8f939b; --line:#dcd8cc; --accent:#1c1e22; {mvars_light}
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }}
@media (prefers-color-scheme: dark) {{ :root {{ --paper:#14161a; --panel:#1b1e24; --panel2:#f2f0e8;
  --ink:#e8e6de; --ink-soft:#a6aab2; --ink-faint:#70747c; --line:#2a2e36; --accent:#e8e6de; {mvars_dark} }} }}
:root[data-theme="light"] {{ --paper:#f6f4ee; --panel:#efede5; --panel2:#fffdf7; --ink:#1c1e22;
  --ink-soft:#565b64; --ink-faint:#8f939b; --line:#dcd8cc; --accent:#1c1e22; {mvars_light} }}
:root[data-theme="dark"] {{ --paper:#14161a; --panel:#1b1e24; --panel2:#f2f0e8; --ink:#e8e6de;
  --ink-soft:#a6aab2; --ink-faint:#70747c; --line:#2a2e36; --accent:#e8e6de; {mvars_dark} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:44px 28px 72px; }}
h1 {{ font-size:clamp(24px,3.6vw,34px); letter-spacing:-0.02em; line-height:1.12; margin:0 0 12px; }}
h2 {{ font-size:20px; letter-spacing:-0.01em; margin:46px 0 6px; }}
.method {{ color:var(--ink-soft); max-width:74ch; margin:0 0 10px; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0 4px; font-family:var(--mono); font-size:13px; align-items:center; }}
.legend.prov {{ font-size:11.5px; gap:13px; margin-top:8px; }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }}
.sh {{ display:inline-flex; margin-right:7px; vertical-align:-3px; }}
.filterbar {{ display:flex; gap:8px; margin:24px 0 14px; font-family:var(--mono); }}
.fbtn {{ font:12.5px var(--mono); background:var(--panel); color:var(--ink-soft);
  border:1px solid var(--line); border-radius:99px; padding:6px 14px; cursor:pointer; }}
.fbtn[aria-pressed="true"] {{ background:var(--ink); color:var(--paper); border-color:var(--ink); }}
{dim_css} {{ opacity:0.07; }}
body:not([data-filter="all"]) .frontier {{ opacity:0.12; }}
[data-m] {{ transition:opacity .18s; }}
.chart {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:18px 10px 6px; overflow-x:auto; }}
.chart svg {{ width:100%; height:auto; display:block; min-width:640px; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.tick {{ font:11px var(--mono); fill:var(--ink-faint); }}
.axis {{ font:12px var(--mono); fill:var(--ink-soft); letter-spacing:0.06em; }}
.pt-label {{ font:10.5px var(--mono); }}
.lgd {{ font:13px var(--mono); }}
.subtitle {{ font:12px var(--mono); letter-spacing:0.02em; }}
.frontier {{ fill:none; stroke:var(--accent); stroke-width:1.2; stroke-dasharray:5 4; opacity:0.5; }}
.tbl {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-family:var(--mono); font-size:12.5px;
  font-variant-numeric:tabular-nums; }}
th {{ text-align:left; font-size:10.5px; letter-spacing:0.12em; text-transform:uppercase;
  color:var(--ink-faint); font-weight:600; }}
th,td {{ padding:8px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }}
tr:last-child td {{ border-bottom:none; }}
td.r, th.r {{ text-align:right; }}
</style>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <p class="method">{html.escape(note)}</p>
  <div class="legend">{model_legend}</div>
  <div class="legend prov">{prov_legend}</div>
  <div class="filterbar" role="group" aria-label="Model filter">{pills}</div>
  <h2 id="chart-cost">Cost vs quality: the efficient frontier</h2>
  <p class="method">Dashed line traces the Pareto frontier. Logo marks the serving provider, ring color marks the model.</p>
  <div class="chart">{cost_chart}</div>
  {f'<h2 id="chart-speed">Speed vs quality</h2><p class="method">{html.escape(latency_label)}.</p><div class="chart">{lat_chart}</div>' if lat_chart else ''}
  <h2>All routes</h2>
  <div class="tbl"><table><tr><th>model</th><th>host</th><th>quant</th><th class='r'>quality</th>
  <th class='r'>cost</th><th class='r'>p50</th><th class='r'>p95</th><th class='r'>TPS</th>
  <th>pin&nbsp;check</th></tr>{table_rows}</table></div>
</div>
<script>
document.body.dataset.filter = "all";
var btns = document.querySelectorAll(".fbtn");
btns.forEach(function (b) {{
  b.addEventListener("click", function () {{
    document.body.dataset.filter = b.dataset.f;
    btns.forEach(function (x) {{ x.setAttribute("aria-pressed", String(x === b)); }});
  }});
}});
</script>
"""


def write_report(rows: list[Route], output: Path, *, title: str, note: str,
                 logo_cache: Path | None = None, **labels) -> Path:
    """Library entry point: rows in, self-contained report out."""
    cache = logo_cache or output.parent / "logos"
    logos = fetch_logos(sorted({r.provider for r in rows}), cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(rows, logos, title=title, note=note, **labels))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="JSON array of route rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Route comparison")
    parser.add_argument("--note", default="")
    parser.add_argument("--subtitle", default=None,
                        help="one line drawn inside each chart (models legend + this)")
    parser.add_argument("--cost-label", default="cost per unit (USD, log scale)")
    parser.add_argument("--logo-cache", type=Path, default=None)
    args = parser.parse_args()
    rows = [Route.from_dict(d) for d in json.loads(args.rows.read_text())]
    if not rows:
        print("no rows")
        return 1
    write_report(rows, args.output, title=args.title, note=args.note,
                 logo_cache=args.logo_cache, cost_label=args.cost_label,
                 subtitle=args.subtitle)
    print(f"{len(rows)} routes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
