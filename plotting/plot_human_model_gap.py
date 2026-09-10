"""
Dot plot: human vs. VLM accuracy on temporal swap tasks (human study subset).
Outputs figures/human_model_gap.{pdf,png}.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = {
    "detect": {
        "chance": 50,
        "rows": [
            {"abbr": "CL", "human": 77.8, "models": [51.4, 60.0, 45.7, 40.0, 51.4]},
            {"abbr": "CR", "human": 91.7, "models": [50.0, 60.0, 61.7, 58.3, 50.0]},
            {"abbr": "DR", "human": 75.0, "models": [50.0, 41.7, 44.4, 58.3, 58.3]},
            {"abbr": "MT", "human": 88.9, "models": [55.6, 69.4, 38.9, 52.8, 47.2]},
        ],
    },
    "localize": {
        "chance": None,
        "rows": [
            {"abbr": "CL", "human": 86.1, "models": [27.8, 27.8, 19.4, 22.2, 19.4]},
            {"abbr": "CR", "human": 88.9, "models": [17.1, 25.7, 22.9, 14.3, 22.9]},
            {"abbr": "DR", "human": 83.3, "models": [16.7, 13.9, 27.8, 5.6, 8.3]},
            {"abbr": "MT", "human": 88.9, "models": [27.8, 50.0, 36.1, 16.7, 50.0]},
        ],
    },
}

import numpy as np

MODEL_NAMES = [
    "Qwen2.5-VL-7B",
    "Qwen3-VL-8B",
    "Gemma-4-E4B",
    "InternVL3-8B",
    "InternVL3.5-8B",
]
MODEL_COLORS = plt.cm.Blues(np.linspace(0.35, 0.85, len(MODEL_NAMES)))
HUMAN_COLOR = "#c2410c"
GROUND = "white"
BAND_COLOR = "#d0d8e4"
GRID_COLOR = "#e8e8e8"
TEXT_COLOR = "#1a1714"

Y_OFFSETS = [-0.15, -0.075, 0.0, 0.075, 0.15]


def draw_panel(ax, task):
    config = DATA[task]
    rows = config["rows"]
    n = len(rows)

    for i, row in enumerate(rows):
        min_m = min(row["models"])
        max_m = max(row["models"])
        gap = row["human"] - max_m
        mid = (max_m + row["human"]) / 2

        # alternating row tint
        if i % 2 == 0:
            ax.axhspan(i - 0.48, i + 0.48, color="black", alpha=0.02, zorder=0)

        # model range band
        ax.barh(
            i,
            max_m - min_m,
            left=min_m,
            height=0.22,
            color=BAND_COLOR,
            zorder=2,
            linewidth=0,
        )

        # model dots (slight vertical spread so overlapping scores separate)
        for mi, (score, color) in enumerate(zip(row["models"], MODEL_COLORS)):
            ax.scatter(
                score,
                i + Y_OFFSETS[mi],
                c=color,
                s=26,
                zorder=4,
                linewidths=0,
                alpha=0.88,
            )

        # human dot
        ax.scatter(
            row["human"],
            i,
            c=HUMAN_COLOR,
            s=65,
            zorder=5,
            edgecolors=GROUND,
            linewidths=1.5,
        )

        # human score label to the right
        ax.text(
            row["human"] + 1.5,
            i,
            f"{row['human']}",
            va="center",
            ha="left",
            fontsize=9,
            color=HUMAN_COLOR,
            fontweight="bold",
        )

        # gap line split around the label (leave ~8 data units for text)
        half_gap = 4.0
        ax.plot(
            [max_m, mid - half_gap],
            [i, i],
            color=HUMAN_COLOR,
            lw=1.2,
            alpha=0.65,
            zorder=5,
            linestyle="--",
            dashes=(3, 2),
        )
        ax.plot(
            [mid + half_gap, row["human"]],
            [i, i],
            color=HUMAN_COLOR,
            lw=1.2,
            alpha=0.65,
            zorder=5,
            linestyle="--",
            dashes=(3, 2),
        )
        ax.text(
            mid,
            i,
            f"+{gap:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color=HUMAN_COLOR,
            fontweight="bold",
            alpha=0.85,
            zorder=6,
        )

    # chance line (detection only)
    if config["chance"]:
        ax.axvline(
            config["chance"],
            color="#c8c4c0",
            lw=0.9,
            linestyle="--",
            zorder=1,
            dashes=(3, 3),
        )
        ax.text(
            config["chance"],
            -0.55,
            "chance",
            ha="center",
            va="top",
            fontsize=8,
            color="#aaa8a5",
        )

    # light vertical grid
    for v in [25, 50, 75, 100]:
        ax.axvline(v, color=GRID_COLOR, lw=0.5, zorder=0)

    ax.set_xlim(0, 107)
    ax.set_ylim(-0.65, n - 0.35)
    ax.invert_yaxis()

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["abbr"] for r in rows], fontsize=12, color="black")

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=12, color="black")

    ax.set_xlabel("Accuracy (%)", fontsize=13, labelpad=5, color="black")
    ax.set_title(
        "Detection" if task == "detect" else "Localization",
        fontsize=12,
        pad=8,
        loc="left",
        color="black",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="both", length=4, color="black")


fig, axes = plt.subplots(2, 1, figsize=(6, 8))

draw_panel(axes[0], "detect")
draw_panel(axes[1], "localize")

# shared legend
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=MODEL_COLORS[i],
        markersize=6,
        markeredgewidth=0,
        label=MODEL_NAMES[i],
    )
    for i in range(len(MODEL_NAMES))
] + [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=HUMAN_COLOR,
        markersize=8,
        markeredgecolor=GROUND,
        markeredgewidth=1.5,
        label="Human",
    )
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.16)

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    fontsize=12,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01),
    handlelength=0.8,
    columnspacing=1.0,
    handletextpad=0.4,
)

from pathlib import Path

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.savefig(OUT / "human_model_gap.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "human_model_gap.png", dpi=150, bbox_inches="tight")
print("Saved figures/human_model_gap.{pdf,png}")
plt.show()
