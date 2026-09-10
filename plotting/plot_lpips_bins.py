"""LPIPS bucket accuracy analysis — bar plots for static and quantile bins.

Two bin schemes:
  1. Static:   0.00–0.10, 0.10–0.20, … (bins with no data are skipped)
  2. Quantile: ten equal-frequency deciles of the pooled LPIPS distribution

For each scheme, plots accuracy (% correct) per LPIPS bin as grouped bars
(one group per bin, one bar per model), pooling all four datasets.

Usage:
  python plotting/plot_lpips_bins.py
"""
import csv
import os
import sys

import matplotlib.patches as mpatches
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

MIN_BIN_SIZE = 5  # bins with fewer samples are shown as empty / hatched


def load_swap_rows(csv_path: str) -> list[dict]:
    """Load all rows; correct = gave right answer for ground truth (balanced accuracy)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            lv = r.get("lpips", "").strip()
            if not lv:
                continue
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            gt = r["ground_truth"].strip() == "True"
            correct = int((gt and ans == "yes") or (not gt and ans == "no"))
            rows.append({"lpips": float(lv), "correct": correct})
    return rows


def load_all_rows() -> dict[str, list[dict]]:
    """Load rows per model, pooling all datasets."""
    model_rows: dict[str, list[dict]] = {}
    for label, mdir in MODELS:
        rows: list[dict] = []
        for ds in DATASETS:
            path = os.path.join(BASE, mdir, f"swap_detect_{ds}.csv")
            if os.path.exists(path):
                rows.extend(load_swap_rows(path))
        if rows:
            model_rows[label] = rows
    return model_rows


def bin_accuracy(rows: list[dict], edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (accuracy_pct, count) arrays of length len(edges)-1."""
    n = len(edges) - 1
    accs = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    lpips = np.array([r["lpips"] for r in rows])
    correct = np.array([r["correct"] for r in rows])
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        mask = (lpips >= lo) & (lpips <= hi if i == n - 1 else lpips < hi)
        if mask.sum() >= MIN_BIN_SIZE:
            accs[i] = correct[mask].mean() * 100
            counts[i] = int(mask.sum())
    return accs, counts


def draw_panel(
    ax: plt.Axes,
    model_rows: dict[str, list[dict]],
    edges: np.ndarray,
    labels: list[str],
    title: str,
    color_offset: int = 0,
    tick_fontsize: int = 8,
    label_fontsize: int = 10,
) -> None:
    n_bins = len(labels)
    n_models = len(model_rows)
    bar_w = 0.8 / n_models
    x = np.arange(n_bins, dtype=float)

    for i, (label, rows) in enumerate(model_rows.items()):
        accs, counts = bin_accuracy(rows, edges)
        offset = (i - n_models / 2 + 0.5) * bar_w
        color = MODEL_PALETTE[(i + color_offset) % len(MODEL_PALETTE)]
        heights = np.where(np.isnan(accs), 0.0, accs)
        bars = ax.bar(
            x + offset, heights,
            width=bar_w * 0.88,
            color=color,
            alpha=0.85,
            label=label,
        )
        # Mark empty / sparse bins with hatching
        for bar, cnt in zip(bars, counts):
            if cnt == 0:
                bar.set_hatch("///")
                bar.set_alpha(0.25)

    ax.axhline(50, color=CHANCE_COLOR, linestyle="--", linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=tick_fontsize)
    ax.set_ylabel("Accuracy (%)", fontsize=label_fontsize)
    ax.set_ylim(0, 108)
    ax.set_title(title, fontsize=11, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    model_rows = load_all_rows()
    if not model_rows:
        print("No data found — check that results/*/swap_detect_*.csv files exist.")
        return

    # Reference LPIPS values (same pairs appear in every model, pick one)
    ref_lpips = np.array([r["lpips"] for r in next(iter(model_rows.values()))])

    # ── Static bins: 0.0, 0.1, 0.2, … ──────────────────────────────────────
    lmax = ref_lpips.max()
    raw_edges = np.arange(0.0, lmax + 0.1 + 1e-9, 0.1)
    # Keep only bins that have enough data in at least one model
    n_raw = len(raw_edges) - 1
    keep = np.zeros(n_raw, dtype=bool)
    for rows in model_rows.values():
        lpips_arr = np.array([r["lpips"] for r in rows])
        for i in range(n_raw):
            lo, hi = raw_edges[i], raw_edges[i + 1]
            cnt = ((lpips_arr >= lo) & (lpips_arr < (hi if i < n_raw - 1 else hi + 1))).sum()
            if cnt >= MIN_BIN_SIZE:
                keep[i] = True

    kept_idx = np.where(keep)[0]
    static_edges = np.concatenate([raw_edges[kept_idx], [raw_edges[kept_idx[-1] + 1]]])
    static_labels = [f"{raw_edges[i]:.1f}–{raw_edges[i+1]:.1f}" for i in kept_idx]

    # ── Quantile bins: deciles ───────────────────────────────────────────────
    q_edges = np.unique(np.percentile(ref_lpips, np.linspace(0, 100, 11)))
    if len(q_edges) < 11:
        print(f"Note: {11 - len(q_edges)} duplicate quantile edges collapsed (tied LPIPS values).")
    q_edges[0] -= 1e-9  # open left edge so all values fall in a bin
    n_q = len(q_edges) - 1
    q_labels = [
        f"D{i+1}\n[{q_edges[i]:.3f}, {q_edges[i+1]:.3f}]"
        for i in range(n_q)
    ]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    draw_panel(ax, model_rows, static_edges, static_labels, "",
               tick_fontsize=14, label_fontsize=15)

    model_handles = [
        mpatches.Patch(color=MODEL_PALETTE[i % len(MODEL_PALETTE)], label=lbl)
        for i, lbl in enumerate(model_rows.keys())
    ]
    chance_handle = plt.Line2D(
        [0], [0], color=CHANCE_COLOR, linestyle="--", linewidth=1.5, label="Chance (50%)"
    )
    fig.legend(
        handles=model_handles + [chance_handle],
        loc="lower center",
        ncol=len(model_handles) + 1,
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(BASE, "figures", f"lpips_bin_accuracy.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"→ {out}")
    plt.close(fig)

    # ── Qwen3-VL-8B only, static bins, split by real vs synthetic ───────────
    qwen3_label = "Qwen3-VL-8B"
    if qwen3_label in model_rows:
        qwen3_dir = next(mdir for lbl, mdir in MODELS if lbl == qwen3_label)
        synthetic_rows: list[dict] = []
        real_rows: list[dict] = []
        for ds in DATASETS:
            path = os.path.join(BASE, qwen3_dir, f"swap_detect_{ds}.csv")
            if not os.path.exists(path):
                continue
            if ds in ("clevrer", "craft"):
                synthetic_rows.extend(load_swap_rows(path))
            else:
                real_rows.extend(load_swap_rows(path))
        split_rows = {
            "Synthetic": synthetic_rows,
            "Real":      real_rows,
        }
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        draw_panel(
            ax2, split_rows, static_edges, static_labels,
            "",
            tick_fontsize=12,
            label_fontsize=13,
        )
        split_handles = [
            mpatches.Patch(color=MODEL_PALETTE[i], label=lbl)
            for i, lbl in enumerate(split_rows.keys())
        ]
        chance_handle2 = plt.Line2D(
            [0], [0], color=CHANCE_COLOR, linestyle="--", linewidth=1.5, label="Chance (50%)"
        )
        fig2.legend(
            handles=split_handles + [chance_handle2],
            loc="lower center",
            ncol=3,
            fontsize=12,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out2 = os.path.join(BASE, "figures", "lpips_bin_accuracy_qwen3.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"→ {out2}")
    else:
        print(f"Warning: '{qwen3_label}' not found in loaded data — skipping single-model plot.")


if __name__ == "__main__":
    main()
