"""Cumulative tolerance curve for swap localization.

For each model, plots the fraction of predictions within ±k frames of the
true swap position, for k = 0, 1, 2, …  One line per model, all datasets pooled.

Ground truth is stored 0-based; model answers are 1-based (prompts say so).
Error = |min(answer) − (gt_first + 1)|  i.e. distance between predicted and
true swap boundary in 1-based frame indices.

A shaded chance band shows the expected curve for a uniform random predictor
(accounts for varying sequence lengths across the dataset).

Usage:
  python plotting/plot_localize_tolerance.py
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colors import CHANCE_COLOR, MODEL_PALETTE

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
MAX_K = 6


def parse_pair(s: str) -> tuple[int, int] | None:
    s = s.strip()
    m = _GT_RE.match(s) or _PAIR_RE.match(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_errors(csv_path: str) -> list[tuple[int, int]]:
    """Return list of (error, seq_len) for each parseable row.

    error = |min(answer_1based) − (gt_first_0based + 1)|
    Unparseable answers are assigned error = seq_len (guaranteed wrong at all k).
    """
    results = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            seq_len = int(r["seq_len"])
            gt = parse_pair(r["ground_truth"])
            ans = parse_pair(r["answer"])
            if gt is None or ans is None:
                results.append((seq_len, seq_len))  # unparseable → max error
                continue
            true_pos = gt[0] + 1          # 1-based first frame of true swap
            pred_pos = min(ans[0], ans[1]) # 1-based first frame of predicted swap
            error = abs(pred_pos - true_pos)
            results.append((error, seq_len))
    return results


def cumulative_curve(errors_and_lens: list[tuple[int, int]], max_k: int) -> np.ndarray:
    """Fraction of predictions within ±k for k = 0 … max_k."""
    errors = np.array([e for e, _ in errors_and_lens])
    n = len(errors)
    return np.array([np.sum(errors <= k) / n for k in range(max_k + 1)]) * 100


def chance_curve_for_L(L: int, max_k: int) -> list[float]:
    """P(within k) for a uniform random predictor on a sequence of length L.

    Both the true position and the predicted position are uniform on {1, …, L-1}.
    For each true position t, count how many predicted positions g satisfy |g-t| ≤ k,
    then average over all t.
    """
    n_pos = L - 1
    result = []
    for k in range(max_k + 1):
        total = 0
        for t in range(1, L):          # true positions 1 … L-1
            covered = min(t + k, L - 1) - max(t - k, 1) + 1
            total += covered
        result.append(total / (n_pos * n_pos))
    return result


def chance_band(errors_and_lens: list[tuple[int, int]], max_k: int) -> np.ndarray:
    """Mean chance curve across all examples (weighted by their sequence length)."""
    seq_lens = [sl for _, sl in errors_and_lens]
    curves = [chance_curve_for_L(L, max_k) for L in seq_lens]
    return np.mean(curves, axis=0) * 100


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    k_vals = np.arange(MAX_K + 1)

    all_errors: list[tuple[int, int]] = []

    for idx, (label, mdir) in enumerate(MODELS):
        rows: list[tuple[int, int]] = []
        for ds in DATASETS:
            path = os.path.join(BASE, mdir, f"swap_localize_{ds}.csv")
            if os.path.exists(path):
                rows.extend(load_errors(path))
        if not rows:
            continue
        all_errors.extend(rows)

        curve = cumulative_curve(rows, MAX_K)
        color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        ax.plot(k_vals, curve, marker="o", markersize=5, linewidth=2,
                color=color, label=f"{label}  (n={len(rows):,})")

    if all_errors:
        chance = chance_band(all_errors, MAX_K)
        ax.plot(k_vals, chance, color=CHANCE_COLOR, linestyle="--",
                linewidth=1.5, label="Chance (uniform random)")

    ax.set_xlabel("Tolerance  k  (frames)", fontsize=11)
    ax.set_ylabel("Predictions within ±k frames (%)", fontsize=11)
    ax.set_title("Swap localization — cumulative tolerance curve\n(all datasets pooled)",
                 fontsize=12)
    ax.set_xticks(k_vals)
    ax.set_xticklabels([f"±{k}" for k in k_vals])
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = os.path.join(BASE, "figures", "localize_tolerance.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
