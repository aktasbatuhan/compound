#!/usr/bin/env python3
"""Chart one grid's per-host metrics, one panel per metric, models side by side.

The question a reader has is not "how did this host do" but "does the host I
would pick stay the host I would pick when the model changes". So every panel
puts the two models next to each other on the same host row: a bar pair that
disagrees is a host-model pair, not a host property.

    python3 scripts/plot_grid.py artifacts/fswe-<stamp> --out chart.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from analyze_arms import summarize_arm  # noqa: E402
from analyze_grid import collect  # noqa: E402

INK = "#191817"
INK2 = "#57534a"
LINE = "#e6e2d8"
PAPER = "#faf9f5"
COLORS = {0: "#1740e6", 1: "#c92a2a"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", default="grid-chart.png")
    parser.add_argument("--title", default="Same tasks, same client, eight serving hosts")
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()

    data = collect(args.root)
    if not data:
        print(f"no ledgers under {args.root}")
        return 1
    models = sorted({m for m, _ in data})
    routes = sorted({r for _, r in data})

    stats: dict[tuple[str, str], dict] = {}
    for (model, route), tasks in data.items():
        pooled = [r for rows in tasks.values() for r in rows]
        stats[(model, route)] = summarize_arm(route, pooled)

    panels = [
        (
            "$ per 1M prompt tokens",
            lambda a: (a["cost_per_1k_prompt"] or 0) * 1000 if a["priced_calls"] else 0,
            "measured per call; blank where the host reports none",
        ),
        (
            "cache %",
            lambda a: (a["cache_ratio"] or 0) * 100,
            "share of prompt tokens served from cache",
        ),
        ("incomplete %", lambda a: a["hang_rate"] * 100, "calls that never returned within 300s"),
        ("p50 latency (s)", lambda a: a["p50_s"] or 0, "median call latency"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 6.4), dpi=200)
    fig.patch.set_facecolor(PAPER)

    height = 0.36
    ypos = list(range(len(routes)))[::-1]
    for ax, (label, getter, caption) in zip(axes, panels, strict=True):
        ax.set_facecolor(PAPER)
        for i, model in enumerate(models):
            values = [getter(stats[(model, r)]) if (model, r) in stats else 0 for r in routes]
            offset = (i - (len(models) - 1) / 2) * height
            bars = ax.barh(
                [y + offset for y in ypos], values, height=height,
                color=COLORS.get(i, "#98938a"), label=model,
            )
            for bar, value in zip(bars, values, strict=True):
                if value > 0:
                    # Precision follows the panel's scale: dollars per million
                    # tokens are sub-unit and would all read "0", while a
                    # percentage does not want three decimals.
                    top = max(values)
                    digits = 3 if top < 1 else (1 if top < 10 else 0)
                    text = f"{value:.{digits}f}"
                    ax.text(
                        bar.get_width() + max(values) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        text, va="center", fontsize=8, color=INK,
                    )
        ax.set_yticks(ypos)
        ax.set_yticklabels(routes, fontsize=10, color=INK)
        ax.set_title(label, fontsize=12, fontweight="bold", color=INK, loc="left", pad=16)
        ax.text(0, 1.02, caption, transform=ax.transAxes, fontsize=8.5, color=INK2)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#d5d0c2")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9, colors=INK2)
        ax.grid(axis="x", color=LINE, lw=0.8)
        ax.set_axisbelow(True)

    # Legend at figure level: inside a panel it lands on top of a bar whenever a
    # host scores low on that metric.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", frameon=False, fontsize=10,
        bbox_to_anchor=(0.98, 0.98), ncols=len(labels),
    )
    fig.suptitle(args.title, x=0.02, ha="left", fontsize=17, fontweight="bold", color=INK)
    if args.subtitle:
        fig.text(0.02, 0.925, args.subtitle, fontsize=10, color=INK2, va="top")
    fig.text(
        0.02, 0.015,
        "Every bar is pooled over the same tasks per host. Cache is each host's own reported "
        "cached/prompt ratio. Doubleword reports no per-call cost, so it is absent from cost "
        "comparisons; its billed total comes from its own meters.",
        fontsize=8, color=INK2, va="bottom",
    )
    plt.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.12, wspace=0.62)
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
