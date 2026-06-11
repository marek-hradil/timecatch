"""Heatmap of ground-truth vs predicted position for corrupt_localize or swap_localize.

Ground truth is stored 0-based; prompts say 1-based, so we shift gt +1
when comparing to model answers.

For swap_localize the pair (i, j) is reduced to min(pair) for the scalar
position used on both heatmap axes (the first / earlier swapped frame).
A fourth panel shows the distribution of predicted pair gaps |i - j|
(correct answers always have gap = 1).

Usage:
    python plotting/plot_heatmap.py results/qwen3-vl-8b/corrupt_localize_*.csv
    python plotting/plot_heatmap.py results/qwen3-vl-8b/swap_localize_*.csv
"""
import csv
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from colors import ABLATION_A, ABLATION_B

INT_RE      = re.compile(r"^\d{1,2}$")
PAIR_GT_RE  = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
PAIR_ANS_RE = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def detect_task(paths: list[str]) -> str:
    for p in paths:
        base = os.path.basename(p)
        if "swap_localize" in base:
            return "swap_localize"
        if "corrupt_localize" in base:
            return "corrupt_localize"
    sys.exit("Cannot detect task from filenames — expected corrupt_localize or swap_localize")


def load_corrupt(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    gt  = int(r["ground_truth"]) + 1    # 0-based → 1-based
                    ans = r["answer"].strip()
                    if not INT_RE.match(ans):
                        continue
                    rows.append({"gt": gt, "ans": int(ans), "seq_len": int(r["seq_len"])})
                except (ValueError, KeyError):
                    continue
    return rows


def load_swap(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    m_gt  = PAIR_GT_RE.match(r["ground_truth"].strip())
                    m_ans = PAIR_ANS_RE.match(r["answer"].strip())
                    if not m_gt or not m_ans:
                        continue
                    gt_pair  = (int(m_gt.group(1))  + 1, int(m_gt.group(2))  + 1)
                    ans_pair = (int(m_ans.group(1)),      int(m_ans.group(2)))
                    exact = ans_pair == gt_pair or ans_pair == (gt_pair[1], gt_pair[0])
                    gt_pos  = min(gt_pair)
                    ans_pos = min(ans_pair)
                    gap     = abs(ans_pair[0] - ans_pair[1])
                    rows.append({
                        "gt": gt_pos, "ans": ans_pos,
                        "exact": exact,
                        "gap": gap,
                        "seq_len": int(r["seq_len"]),
                    })
                except (ValueError, KeyError):
                    continue
    return rows


def plot(rows: list[dict], task: str, n_files: int, out_path: str) -> None:
    n = len(rows)
    if n == 0:
        sys.exit("No valid rows found")

    min_pos = 1
    # Axis range determined by GT so OOB predictions don't squish the heatmap.
    max_pos = max(r["gt"] for r in rows)
    oob     = sum(1 for r in rows if r["ans"] > max_pos)
    # Clamp answers for display only — keep original for exact-match computation.
    rows    = [{**r, "ans_display": min(r["ans"], max_pos)} for r in rows]
    size    = max_pos - min_pos + 1
    labels  = list(range(min_pos, max_pos + 1))

    matrix = np.zeros((size, size), dtype=int)
    for r in rows:
        gi, ai = r["gt"] - min_pos, r["ans_display"] - min_pos
        if 0 <= gi < size and 0 <= ai < size:
            matrix[gi, ai] += 1

    errors       = [r["ans_display"] - r["gt"] for r in rows]
    error_counts = Counter(errors)
    min_e, max_e = min(errors), max(errors)
    e_range      = list(range(min_e, max_e + 1))
    e_vals       = [error_counts.get(e, 0) for e in e_range]

    if task == "swap_localize":
        exact = sum(1 for r in rows if r["exact"])
    else:
        exact = sum(1 for r in rows if r["gt"] == r["ans"])

    gt_marginal = matrix.sum(axis=1)
    mean_err    = sum(e * c for e, c in error_counts.items()) / n

    pos_label = "first swapped frame (1-based)" if task == "swap_localize" else "corrupted frame (1-based)"

    is_swap    = task == "swap_localize"
    n_cols     = 4 if is_swap else 3
    fig_width  = 18 if is_swap else 14
    fig        = plt.figure(figsize=(fig_width, 5))

    oob_note = f"  |  OOB (clamped): {oob} ({oob/n*100:.1f}%)" if oob else ""
    if is_swap:
        non_consec = sum(1 for r in rows if r.get("gap", 1) != 1)
        nc_note    = f"  |  non-consecutive pairs: {non_consec} ({non_consec/n*100:.1f}%)"
    else:
        nc_note = ""
    fig.suptitle(
        f"{task} — {n} rows across {n_files} dataset(s)"
        f"  |  exact match (1-based): {exact/n*100:.1f}%{oob_note}{nc_note}",
        fontsize=10,
    )

    # Panel 1+2: heatmap
    ax1 = fig.add_subplot(1, n_cols, (1, 2))
    im = ax1.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max(matrix.max(), 1))
    ax1.set_xticks(range(size))
    ax1.set_yticks(range(size))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel(f"Predicted {pos_label}")
    ax1.set_ylabel(f"Ground truth {pos_label}")
    ax1.set_title("GT vs predicted  (diagonal = correct)")
    for i in range(size):
        ax1.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="salmon", linewidth=1.5))
    for i in range(size):
        for j in range(size):
            v = matrix[i, j]
            if v > 0:
                ax1.text(j, i, str(v), ha="center", va="center", fontsize=7,
                         color="white" if v > matrix.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    ax1_r = ax1.inset_axes([1.02, 0, 0.12, 1])
    ax1_r.barh(range(size), gt_marginal, color=ABLATION_A, alpha=0.7)
    ax1_r.set_yticks(range(size))
    ax1_r.set_yticklabels([])
    ax1_r.set_xlabel("GT\ncount", fontsize=8)
    ax1_r.invert_xaxis()

    # Panel 3: signed error histogram
    ax2 = fig.add_subplot(1, n_cols, 3)
    bars = ax2.bar(e_range, e_vals, color=[
        ABLATION_B if e != 0 else ABLATION_A for e in e_range
    ], edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Predicted − ground truth  (0 = correct)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Signed error  (mean={mean_err:+.2f})")
    ax2.set_xticks(e_range)
    for bar, v in zip(bars, e_vals):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 1,
                     str(v), ha="center", va="bottom", fontsize=8)

    # Panel 4 (swap only): predicted pair gap distribution
    if is_swap:
        gaps       = [r["gap"] for r in rows]
        gap_counts = Counter(gaps)
        max_gap    = max(gaps)
        g_range    = list(range(0, max_gap + 1))
        g_vals     = [gap_counts.get(g, 0) for g in g_range]

        GAP_CAP = 8
        g_range_capped = list(range(0, GAP_CAP + 1))
        g_vals_capped  = [gap_counts.get(g, 0) for g in range(0, GAP_CAP)]
        g_vals_capped.append(sum(gap_counts.get(g, 0) for g in range(GAP_CAP, max_gap + 1)))
        oob_gap = sum(gap_counts.get(g, 0) for g in range(GAP_CAP + 1, max_gap + 1))

        ax3 = fig.add_subplot(1, n_cols, 4)
        bar_colors = [ABLATION_A if g == 1 else ABLATION_B for g in g_range_capped]
        gbars = ax3.bar(g_range_capped, g_vals_capped, color=bar_colors, edgecolor="white")
        ax3.axvline(1, color="black", linewidth=1.2, linestyle="--")
        ax3.set_xlabel("|predicted_i − predicted_j|  (correct = 1)")
        ax3.set_ylabel("Count")
        consec_pct = gap_counts.get(1, 0) / n * 100
        cap_note   = f"  (≥{GAP_CAP} clamped)" if oob_gap else ""
        ax3.set_title(f"Predicted pair gap  (consecutive: {consec_pct:.1f}%){cap_note}")
        tick_labels = [str(g) for g in range(0, GAP_CAP)] + [f"≥{GAP_CAP}"]
        ax3.set_xticks(g_range_capped)
        ax3.set_xticklabels(tick_labels)
        for bar, v in zip(gbars, g_vals_capped):
            if v > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2, v + 1,
                         str(v), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"→ {out_path}")
    print(f"  n={n}  exact={exact}({exact/n*100:.1f}%)  mean_error={mean_err:+.2f}", end="")
    if is_swap:
        print(f"  non-consecutive={non_consec}({non_consec/n*100:.1f}%)", end="")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_heatmap.py <results.csv> [...]")

    paths = sys.argv[1:]
    task  = detect_task(paths)

    rows  = load_swap(paths) if task == "swap_localize" else load_corrupt(paths)

    src_dir  = os.path.dirname(os.path.abspath(paths[0])) or "."
    out_name = "swap_localize_heatmap.png" if task == "swap_localize" else "corrupt_localize_heatmap.png"
    out_path = os.path.join(src_dir, out_name)

    plot(rows, task, len(paths), out_path)
