"""Plot scene-description ablation results per task.

For each task, shows accuracy averaged over all datasets and models,
with individual model averages overlaid as scatter points.

Auto-discovers model result pairs (results/<model>/ + results/<model>/no-scene-desc/)
in the results/ directory.

Usage:
    python plot_scene_desc_by_task.py
    python plot_scene_desc_by_task.py --out scene_desc_ablation.png
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from colors import ABLATION_A, ABLATION_B, CHANCE_COLOR

INT_RE     = re.compile(r"^\d+$")
PAIR_GT_RE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
PAIR_RE    = re.compile(r"^(\d+)\s*,\s*(\d+)$")

TASKS = ["corrupt_detect", "corrupt_localize", "swap_detect", "swap_localize"]
TASK_LABELS = {
    "corrupt_detect":   "corrupt\ndetect",
    "corrupt_localize": "corrupt\nlocalize",
    "swap_detect":      "swap\ndetect",
    "swap_localize":    "swap\nlocalize",
}
CHANCE = {
    "corrupt_detect":   50.0,
    "swap_detect":      50.0,
    "corrupt_localize": None,   # depends on seq_len — approximate below
    "swap_localize":    None,
}


def acc_csv(path: str, task: str) -> float | None:
    correct = total = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                gt  = r["ground_truth"].strip()
                ans = r["answer"].strip()
                if task in ("corrupt_detect", "swap_detect"):
                    a = ans.lower()
                    if a not in ("yes", "no"):
                        continue
                    ok = (gt == "True") == (a == "yes")
                elif task == "corrupt_localize":
                    if not INT_RE.match(ans):
                        continue
                    ok = int(ans) == int(gt) + 1
                elif task == "swap_localize":
                    mg = PAIR_GT_RE.match(gt)
                    ma = PAIR_RE.match(ans)
                    if not mg or not ma:
                        continue
                    gp = (int(mg.group(1)) + 1, int(mg.group(2)) + 1)
                    ap = (int(ma.group(1)),      int(ma.group(2)))
                    ok = ap == gp or ap == (gp[1], gp[0])
                else:
                    continue
                correct += ok
                total   += 1
            except (ValueError, KeyError):
                continue
    return correct / total * 100 if total else None


def discover_models(root: str) -> list[tuple[str, str, str]]:
    """Return list of (model_name, with_dir, without_dir) pairs."""
    pairs = []
    for name in sorted(os.listdir(root)):
        with_dir    = os.path.join(root, name)
        without_dir = os.path.join(root, name, "no-scene-desc")
        if not os.path.isdir(with_dir) or not os.path.isdir(without_dir):
            continue
        pairs.append((name, with_dir, without_dir))
    return pairs


def collect(root: str) -> dict:
    """Returns data[task][model] = {"with": [acc, ...], "without": [acc, ...]}"""
    models = discover_models(root)
    if not models:
        raise RuntimeError(f"No results/<model>/no-scene-desc/ pairs found in {root}")

    data: dict = {t: defaultdict(lambda: {"with": [], "without": []}) for t in TASKS}

    for model, with_dir, without_dir in models:
        for task in TASKS:
            for fname in sorted(os.listdir(with_dir)):
                if not fname.startswith(task + "_") or not fname.endswith(".csv"):
                    continue
                if "craft-long" in fname:
                    continue
                # ds = fname.removeprefix(task + "_").removesuffix(".csv")
                wp = os.path.join(with_dir,    fname)
                op = os.path.join(without_dir, fname)
                if not os.path.exists(op):
                    continue
                w = acc_csv(wp, task)
                o = acc_csv(op, task)
                if w is not None:
                    data[task][model]["with"].append(w)
                if o is not None:
                    data[task][model]["without"].append(o)

    return data


def main(out_path: str) -> None:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    data = collect(root)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), sharey=False)
    fig.suptitle(
        "Scene description ablation — accuracy averaged over all datasets × models",
        fontsize=11,
    )

    for ax, task in zip(axes, TASKS):
        task_data = data[task]

        # Grand averages
        all_with    = [a for v in task_data.values() for a in v["with"]]
        all_without = [a for v in task_data.values() for a in v["without"]]
        grand_with    = np.mean(all_with)    if all_with    else 0.0
        grand_without = np.mean(all_without) if all_without else 0.0

        width = 0.35
        ax.bar(0, grand_with,    width, color=ABLATION_A, edgecolor="white",
               label="with scene desc",    zorder=2)
        ax.bar(1, grand_without, width, color=ABLATION_B, edgecolor="white",
               label="without scene desc", zorder=2)

        ax.text(0, grand_with    + 1.0, f"{grand_with:.1f}%",    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(1, grand_without + 1.0, f"{grand_without:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Chance line
        if CHANCE[task] is not None:
            ax.axhline(CHANCE[task], color=CHANCE_COLOR, linestyle="--", linewidth=1,
                       label=f"chance ({CHANCE[task]:.0f}%)", zorder=1)

        delta = grand_without - grand_with
        ax.set_title(f"{TASK_LABELS[task]}\nΔ = {delta:+.1f}pp", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["with\nscene desc", "without\nscene desc"], fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 115)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    # Single legend: bar colours + model scatter colours
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out_path}")

    # Print summary table
    print(f"\n{'Task':<20}  {'with':>7}  {'without':>7}  {'Δ':>8}")
    print("-" * 48)
    for task in TASKS:
        all_with    = [a for v in data[task].values() for a in v["with"]]
        all_without = [a for v in data[task].values() for a in v["without"]]
        gw = np.mean(all_with)    if all_with    else float("nan")
        go = np.mean(all_without) if all_without else float("nan")
        print(f"{task:<20}  {gw:7.1f}%  {go:7.1f}%  {go-gw:+8.1f}pp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="scene_desc_ablation_by_task.png",
                        help="Output PNG path (default: scene_desc_ablation_by_task.png)")
    args = parser.parse_args()
    main(args.out)
