"""Compare with vs without scene description accuracy per task × model.

For each task found in both directories, plots accuracy vs sequence length
with two bar groups: 'with scene description' and 'without'. Accuracy is
pooled across all datasets (craft-long excluded). Saves to the with-desc
results directory.

Usage:
    python plotting/plot_scene_desc_comparison.py results/qwen2-5-vl-7b results/qwen2-5-vl-7b/no-scene-desc
"""
import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from colors import ABLATION_A, ABLATION_B, CHANCE_COLOR

_INT_RE     = re.compile(r"^\d+$")
_PAIR_GT_RE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_PAIR_RE    = re.compile(r"^(\d+)\s*,\s*(\d+)$")

TASKS = ("corrupt_detect", "corrupt_localize", "swap_detect", "swap_localize")


def load_csv(path: str, task: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                seq_len = int(r["seq_len"])
                gt  = r["ground_truth"].strip()
                ans = r["answer"].strip()

                if task in ("corrupt_detect", "swap_detect"):
                    ans_l = ans.lower()
                    if ans_l not in ("yes", "no"):
                        continue
                    exact = (gt == "True") == (ans_l == "yes")

                elif task == "corrupt_localize":
                    if not _INT_RE.match(ans):
                        continue
                    exact = int(ans) == int(gt) + 1

                elif task == "swap_localize":
                    m_gt  = _PAIR_GT_RE.match(gt)
                    m_ans = _PAIR_RE.match(ans)
                    if not m_gt or not m_ans:
                        continue
                    gt_pair  = (int(m_gt.group(1)) + 1, int(m_gt.group(2)) + 1)
                    ans_pair = (int(m_ans.group(1)),     int(m_ans.group(2)))
                    exact = ans_pair == gt_pair or ans_pair == (gt_pair[1], gt_pair[0])

                else:
                    continue

                rows.append({"seq_len": seq_len, "exact": exact})
            except (ValueError, KeyError):
                continue
    return rows


def load_dir(directory: str, task: str) -> list[dict]:
    rows = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        if not fname.startswith(task + "_"):
            continue
        if "craft-long" in fname:
            continue
        rows.extend(load_csv(os.path.join(directory, fname), task))
    return rows


def accs_by_len(rows: list[dict]) -> tuple[dict, dict]:
    by_len: dict[int, list] = defaultdict(list)
    for r in rows:
        by_len[r["seq_len"]].append(r["exact"])
    accs   = {l: sum(v) / len(v) * 100 for l, v in by_len.items()}
    counts = {l: len(v) for l, v in by_len.items()}
    return accs, counts


def chance_line(task: str, lens: list[int]) -> tuple[list[float], str]:
    if task in ("corrupt_detect", "swap_detect"):
        return [50.0] * len(lens), "chance (50%)"
    if task == "corrupt_localize":
        return [100.0 / l for l in lens], "chance (1/L)"
    return [100.0 / (l - 1) if l > 1 else 0.0 for l in lens], "chance (1/(L−1))"


def plot_task(task: str, with_dir: str, without_dir: str, model: str) -> None:
    with_rows    = load_dir(with_dir,    task)
    without_rows = load_dir(without_dir, task)
    if not with_rows and not without_rows:
        print(f"  skip {task} — no data in either directory")
        return

    all_lens = sorted({r["seq_len"] for r in with_rows + without_rows})
    with_accs,    with_ns    = accs_by_len(with_rows)
    without_accs, without_ns = accs_by_len(without_rows)

    x     = list(range(len(all_lens)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(all_lens)), 5.5))

    with_vals    = [with_accs.get(l, 0.0)    for l in all_lens]
    without_vals = [without_accs.get(l, 0.0) for l in all_lens]

    bars_w = ax.bar([xi - width / 2 for xi in x], with_vals,    width,
                    label="with scene description",    color=ABLATION_A, edgecolor="white")
    bars_o = ax.bar([xi + width / 2 for xi in x], without_vals, width,
                    label="without scene description", color=ABLATION_B, edgecolor="white")

    for bars, ns in ((bars_w, with_ns), (bars_o, without_ns)):
        for bar, sl in zip(bars, all_lens):
            n = ns.get(sl, 0)
            if n:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"n={n}", ha="center", va="bottom", fontsize=7)

    chance, chance_label = chance_line(task, all_lens)
    ax.plot(x, chance, color=CHANCE_COLOR, linestyle="--", linewidth=1,
            marker="o", markersize=4, label=chance_label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in all_lens])
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_title(
        f"{task} — scene description ablation\n"
        f"(model: {model}, pooled across all datasets excl. craft-long)"
    )
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(with_dir, f"{task}_scene_desc_comparison.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

    n_with    = len(with_rows)
    n_without = len(without_rows)
    avg_with    = sum(r["exact"] for r in with_rows)    / n_with    * 100 if n_with    else 0
    avg_without = sum(r["exact"] for r in without_rows) / n_without * 100 if n_without else 0
    print(f"→ {out_path}")
    print(f"  with={avg_with:.1f}%  without={avg_without:.1f}%  Δ={avg_without - avg_with:+.1f}pp")


def main(with_dir: str, without_dir: str) -> None:
    for d in (with_dir, without_dir):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: directory not found: {d}")

    model = os.path.basename(os.path.abspath(with_dir))

    present = {
        t for t in TASKS
        if any(f.startswith(t + "_") and f.endswith(".csv") and "craft-long" not in f
               for f in os.listdir(with_dir))
        and any(f.startswith(t + "_") and f.endswith(".csv") and "craft-long" not in f
                for f in os.listdir(without_dir))
    }

    for task in TASKS:
        if task not in present:
            print(f"  skip {task} — missing from one or both directories")
            continue
        plot_task(task, with_dir, without_dir, model)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python plot_scene_desc_comparison.py <with-desc-dir> <without-desc-dir>")
    main(sys.argv[1], sys.argv[2])
