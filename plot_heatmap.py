"""Plot heatmaps of ground truth label and model guess distributions.

Usage:
    python plot_heatmap.py [results.csv]
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))

def is_valid(r):
    for key in ("model_prediction_first", "model_prediction_second",
                "ground_truth_first", "ground_truth_second"):
        try:
            int(r[key])
        except (ValueError, KeyError):
            return False
    return True

valid = [r for r in rows if is_valid(r)]

# Collect all position values to determine axis range
all_positions = []
for r in valid:
    all_positions += [
        int(r["ground_truth_first"]), int(r["ground_truth_second"]),
        int(r["model_prediction_first"]), int(r["model_prediction_second"]),
    ]
lo, hi = min(all_positions), max(all_positions)
size = hi - lo + 1
labels = list(range(lo, hi + 1))

def build_heatmap(rows, key_a, key_b):
    counts = defaultdict(int)
    for r in rows:
        a, b = int(r[key_a]), int(r[key_b])
        counts[(a, b)] += 1
    matrix = np.zeros((size, size), dtype=int)
    for (a, b), n in counts.items():
        matrix[a - lo, b - lo] = n
    return matrix

gt_matrix = build_heatmap(valid, "ground_truth_first", "ground_truth_second")
pred_matrix = build_heatmap(valid, "model_prediction_first", "model_prediction_second")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, matrix, title in [
    (axes[0], gt_matrix, "Ground truth label distribution"),
    (axes[1], pred_matrix, "Model guess distribution"),
]:
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Second position")
    ax.set_ylabel("First position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()
