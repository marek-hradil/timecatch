"""
Render a single scene sequence as a paper-ready matplotlib figure.
Highlights swapped frames with a red border; all others have a thin grey border.

Usage:
    python3 scripts/render_scene_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent

SCENE_NAME   = "scene_17_49"
DATASET      = "mtl-aqa"
SHOWN_ORDER  = [0, 1, 3, 2]
SWAP_POS     = {2, 3}          # 0-indexed positions in shown_order

DS_PATH  = ROOT / "datasets" / "MTL-AQA" / "scenes" / SCENE_NAME

OUT = ROOT / "figures" / f"human_failure_{DATASET}_{SCENE_NAME}.pdf"


def load_frame(idx):
    return np.array(Image.open(DS_PATH / f"{idx}.jpg").convert("RGB"))


def main():
    frames = [load_frame(i) for i in SHOWN_ORDER]
    n = len(frames)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.8, 2.8))

    for pos, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])

        is_swapped = pos in SWAP_POS
        edge_color = "#D62728" if is_swapped else "#AAAAAA"
        lw = 3.5 if is_swapped else 1.0

        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(lw)


    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved → {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
