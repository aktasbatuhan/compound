"""Render the saved fixed-budget trial.

Run: uv run --with matplotlib python scripts/plot_flex_budget.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1] / "artifacts/flex-budget-20260910"
NAMES = {
    "gpt-astra": "GPT-6 Astra / OpenRouter",
    "gemini-studio": "Gemini Studio / OpenRouter",
    "gemini-vertex": "Gemini Vertex / OpenRouter",
    "deepseek-dw": "DeepSeek V4 Flash / Doubleword",
    "glm-flash-dw": "GLM-5.3-Flash / Doubleword",
}


def main(root=ROOT, stage="token-informed"):
    source = root / stage
    metadata = source if (source / "plan.json").exists() else root
    plan = json.loads((metadata / "plan.json").read_text())
    task_count = len({e["task_id"] for e in plan["episodes"]})
    trials = len({e["trial"] for e in plan["episodes"]})
    rows = {
        r["episode_id"]: r
        for r in map(json.loads, (source / "outcomes.jsonl").read_text().splitlines())
    }
    metrics = json.loads((root / "metrics.json").read_text())
    routes = sorted(
        NAMES,
        key=lambda route: (
            -next(m for m in metrics if m["route"] == route and m["tier"] == "flex")[
                "success_by_deadline"
            ]["300"],
            next(m for m in metrics if m["route"] == route and m["tier"] == "flex")[
                "median_successful_attempt_s"
            ]
            or float("inf"),
        ),
    )
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, route in zip(axes.flat, routes, strict=False):
        ends = []
        for tier, color, style, label in (
            ("standard", "#343a40", "-", "Realtime"),
            ("flex", "#b34535", "--", "Flex"),
        ):
            episodes = [e for e in plan["episodes"] if e["route"] == route and e["tier"] == tier]
            times = sorted(
                rows[e["episode_id"]]["duration_s"]
                for e in episodes
                if rows.get(e["episode_id"], {}).get("success") is True
                and rows[e["episode_id"]]["duration_s"] <= 900
            )
            n = len(episodes)
            x = [0, *times, 900]
            y = [0, *[(i + 1) / n for i in range(len(times))], len(times) / n]
            ax.step(x, y, where="post", color=color, linestyle=style, linewidth=1.8)
            deadlines = [60, 300, 900]
            ax.scatter(
                deadlines,
                [sum(t <= d for t in times) / n for d in deadlines],
                s=18,
                color=color,
                zorder=3,
            )
            ends.append((f"{label} {len(times)}/{n}", y[-1], color))
        # Curves that finish at the same height would otherwise overprint.
        ends.sort(key=lambda end: end[1])
        gap = 0.14
        if ends[1][1] - ends[0][1] < gap:
            middle = (ends[0][1] + ends[1][1]) / 2
            places = [middle - gap / 2, middle + gap / 2]
        else:
            places = [ends[0][1], ends[1][1]]
        for (text, _, color), place in zip(ends, places, strict=True):
            ax.annotate(
                text,
                (900, place),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
                va="center",
                annotation_clip=False,
            )
        cap = next(m["budget_usd"] for m in metrics if m["route"] == route)
        ax.set_title(f"{NAMES[route]}\n${cap:.2f} agent allowance per attempt", loc="left", pad=14)
        ax.set(xlim=(0, 900), ylim=(-0.04, 1.07), yticks=[0, 0.5, 1])
        ax.set_xticks([0, 60, 300, 900], ["0", "1", "5", "15 min"])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(length=0, colors="#5c6268")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#e8eaec", linewidth=0.6)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        1,
        "How to read\n\n"
        f"{task_count * trials} planned attempts per tier.\n"
        "Height: official grader passes by time.\n\n"
        "Dots: 1, 5 and 15 minute deadlines.\n"
        "Errors and timeouts never raise the curve.\n\n"
        f"{task_count} retail tasks, {trials} trials per task.\n"
        "Equal allowances within each route pair.\n"
        + (
            "Task 41 has a scoring ambiguity. See report."
            if stage == "run"
            else "This small trial cannot establish parity."
        ),
        va="top",
        linespacing=1.25,
    )
    fig.suptitle(
        "Official grader passes as the deadline increases"
        if stage == "run"
        else "Correct results as the deadline increases",
        x=0.07,
        ha="left",
        fontsize=23,
    )
    fig.text(
        0.07,
        0.935,
        f"{len(rows)}/{plan['episode_count']} attempts recorded. "
        "Agent and simulator time included.",
    )
    fig.subplots_adjust(left=0.07, right=0.88, top=0.865, bottom=0.06, hspace=0.65, wspace=0.65)
    fig.savefig(root / "deadline-curves.svg")
    fig.savefig(root / "deadline-curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", default="token-informed")
    args = parser.parse_args()
    main(args.root, args.stage)
