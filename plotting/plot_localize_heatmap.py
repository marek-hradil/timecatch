"""Per-model swap localization heatmaps.

For each model, plots a matrix of true swap position (y) vs predicted swap
position (x), normalized by row so each row sums to 1 (i.e. P(predicted | true)).
The diagonal = correct. Off-diagonal clusters reveal positional biases.

All datasets pooled. One subplot per model in a 1×N grid.

Usage:
  python plotting/plot_localize_heatmap.py
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("Qwen2.5-VL-7B",  "results/qwen2-5-vl-7b"),
    ("Qwen3-VL-8B",    "results/qwen3-vl-8b"),
    ("InternVL3.5-8B", "results/intern-vl-3-5"),
    ("InternVL3-8B",   "results/intern-vl-3"),
    ("Gemma-4-E4B",    "results/gemma-4-e4b"),
]
DATASETS = ["clevrer", "craft", "drive-lm", "mtl-aqa"]

_GT_RE   = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_PAIR_RE = re.compile(r"^(\d+)\s*,\s*(\d+)$")

MAX_POS = 7   # 1-based positions 1–7 cover seq_len up to 8


def parse_pair(s: str) -> tuple[int, int] | None:
    s = s.strip()
    m = _GT_RE.match(s) or _PAIR_RE.match(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_matrix(model_dir: str) -> np.ndarray:
    """Return (MAX_POS × MAX_POS) count matrix [true_pos, pred_pos] (1-based, clamped)."""
    mat = np.zeros((MAX_POS, MAX_POS), dtype=float)
    for ds in DATASETS:
        path = os.path.join(BASE, model_dir, f"swap_localize_{ds}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                gt = parse_pair(r["ground_truth"])
                ans = parse_pair(r["answer"])
                if gt is None or ans is None:
                    continue
                true_pos = gt[0] + 1                   # 0-based → 1-based
                pred_pos = min(ans[0], ans[1])          # already 1-based
                # clamp to valid range
                true_pos = max(1, min(true_pos, MAX_POS))
                pred_pos = max(1, min(pred_pos, MAX_POS))
                mat[true_pos - 1, pred_pos - 1] += 1
    # row-normalise: P(predicted | true)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums


def main() -> None:
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))

    tick_labels = [str(i) for i in range(1, MAX_POS + 1)]

    for ax, (label, mdir) in zip(axes, MODELS):
        mat = build_matrix(mdir)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="equal")

        # annotate cells with percentage
        for r in range(MAX_POS):
            for c in range(MAX_POS):
                val = mat[r, c]
                if val > 0:
                    text_color = "white" if val > 0.55 else "black"
                    ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                            fontsize=6.5, color=text_color)

        ax.set_xticks(range(MAX_POS))
        ax.set_yticks(range(MAX_POS))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Predicted position", fontsize=9)
        ax.set_title(label, fontsize=10, pad=6)

    axes[0].set_ylabel("True position", fontsize=9)


    plt.tight_layout()
    out = os.path.join(BASE, "figures", "localize_heatmap.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")

    # Qwen3-only
    qwen3_label, qwen3_dir = next((l, d) for l, d in MODELS if l == "Qwen3-VL-8B")
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    mat2 = build_matrix(qwen3_dir)
    im2 = ax2.imshow(mat2, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    for r in range(MAX_POS):
        for c in range(MAX_POS):
            val = mat2[r, c]
            if val > 0:
                ax2.text(c, r, f"{val:.0%}", ha="center", va="center",
                         fontsize=6.5, color="white" if val > 0.55 else "black")
    ax2.set_xticks(range(MAX_POS))
    ax2.set_yticks(range(MAX_POS))
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_yticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel("Predicted position", fontsize=9)
    ax2.set_ylabel("True position", fontsize=9)
    ax2.set_title(qwen3_label, fontsize=10, pad=6)
    plt.tight_layout()
    out2 = os.path.join(BASE, "figures", "localize_heatmap_qwen3.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"→ {out2}")


if __name__ == "__main__":
    main()
