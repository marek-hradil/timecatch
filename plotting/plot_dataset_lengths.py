"""Overlapping histogram of scene-length distributions across the four benchmark datasets.

Usage:
    python plotting/plot_dataset_lengths.py
    python plotting/plot_dataset_lengths.py --out figures/dataset_lengths.pdf
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from colors import DATASET as DATASET_COLORS

# Frames-per-scene counts from *_histogram_filtered files.
# Format: {frame_count: num_scenes}
DATASETS = {
    "CLEVRER": {4: 12, 5: 47, 6: 290, 7: 1920, 8: 2728},
    "CRAFT":   {4: 14, 5: 97, 6: 164, 7: 254, 8: 329, 9: 340,
                10: 277, 11: 229, 12: 114, 13: 72, 14: 43, 15: 21, 16: 5},
    "DriveLM": {4: 144, 5: 166, 6: 165, 7: 92, 8: 129},
    "MTL-AQA": {4: 163, 5: 175},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    all_lengths = sorted({l for d in DATASETS.values() for l in d})
    x = np.arange(len(all_lengths))
    n = len(DATASETS)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 4))

    for i, (name, dist) in enumerate(DATASETS.items()):
        total = sum(dist.values())
        pct = [dist.get(l, 0) / total * 100 for l in all_lengths]
        offsets = x + (i - (n - 1) / 2) * width
        color = DATASET_COLORS[name.lower().replace(" ", "-")]
        ax.bar(offsets, pct, width=width, color=color,
               label=f"{name}  (n={total:,})", zorder=2)

    # Divider after frame 8 — everything to the right is CRAFT-only
    split_idx = all_lengths.index(8)
    split_x = split_idx + 0.5
    ax.axvline(split_x, color="gray", linewidth=1, linestyle="--", zorder=4)
    ax.text(split_x + 0.1, 0.97, "CRAFT-long",
            va="top", ha="left", color="gray", fontsize=12,
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(all_lengths, fontsize=12)
    ax.set_xlabel("Frames per scene", fontsize=13)
    ax.set_ylabel("Scenes (%)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(framealpha=0.9, fontsize=12)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
