"""LPIPS vs swap detection — logistic regression per model, datasets pooled.

For each model, fits a logistic regression of correct ~ LPIPS on swap examples
(ground_truth=True rows with an lpips value) and plots the fitted curve.

Usage:
  python plotting/plot_lpips_accuracy.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colors import CHANCE_COLOR, MODEL_PALETTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("Qwen2.5-VL-7B",  "results/qwen2-5-vl-7b"),
    ("Qwen3-VL-8B",    "results/qwen3-vl-8b"),
    ("InternVL3.5-8B", "results/intern-vl-3-5"),
    ("Gemma-4-E4B",    "results/gemma-4-e4b"),
    ("Molmo-7B",       "results/molmo-7b"),
]
DATASETS = ["clevrer", "craft", "drive-lm", "mtl-aqa"]


def load_swap_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["ground_truth"].strip() != "True":
                continue
            lv = r.get("lpips", "").strip()
            if not lv:
                continue
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            rows.append({"lpips": float(lv), "correct": int(ans == "yes")})
    return rows


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    all_lpips: list[float] = []

    for label, mdir in MODELS:
        rows: list[dict] = []
        for ds in DATASETS:
            path = os.path.join(BASE, mdir, f"swap_detect_{ds}.csv")
            if os.path.exists(path):
                rows.extend(load_swap_rows(path))
        if not rows:
            continue
        all_lpips.extend(r["lpips"] for r in rows)

    x_range = np.linspace(min(all_lpips), max(all_lpips), 300).reshape(-1, 1)

    for idx, (label, mdir) in enumerate(MODELS):
        rows = []
        for ds in DATASETS:
            path = os.path.join(BASE, mdir, f"swap_detect_{ds}.csv")
            if os.path.exists(path):
                rows.extend(load_swap_rows(path))
        if not rows:
            continue

        X = np.array([r["lpips"] for r in rows])
        y = np.array([r["correct"] for r in rows])
        n = len(rows)
        acc = y.mean() * 100

        def neg_log_likelihood(params):
            b0, b1 = params
            logits = b0 + b1 * X.ravel()
            p = 1 / (1 + np.exp(-logits))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(neg_log_likelihood, [0.0, 0.0], method="L-BFGS-B")
        b0, b1 = res.x
        y_pred = 100 / (1 + np.exp(-(b0 + b1 * x_range.ravel())))

        color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        ax.plot(x_range, y_pred, linewidth=2, color=color,
                label=f"{label}  (n={n:,}, TPR={acc:.0f}%)")

    ax.axhline(y=50, color=CHANCE_COLOR, linestyle="--", linewidth=1.2,
               label="chance (50%)")

    ax.set_xlabel("LPIPS distance between swapped frames", fontsize=11)
    ax.set_ylabel("P(correct detection) %", fontsize=11)
    ax.set_title("Swap detection vs. visual similarity — logistic fit, all datasets pooled",
                 fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(BASE, "figures", "lpips_vs_swap_detect.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
