"""Qwen3-VL model size scaling: 2B → 4B → 8B → 32B across all tasks.

Accuracy is averaged over all datasets (craft-long excluded).

Usage:
  python plotting/plot_qwen3_scaling.py
"""

import ast
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("2B", "results-qwen3-vl-2b"),
    ("4B", "results-qwen3-vl-4b"),
    ("8B", "results-qwen3-vl-8b"),
    ("32B", "results-qwen3-vl-32b"),
]
DATASETS = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
TASKS = ["corrupt_detect", "corrupt_localize", "swap_detect", "swap_localize"]
TASK_LABELS = {
    "corrupt_detect": "corrupt\ndetect",
    "corrupt_localize": "corrupt\nlocalize",
    "swap_detect": "swap\ndetect",
    "swap_localize": "swap\nlocalize",
}


def acc_binary(path: str) -> float | None:
    correct = total = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ans = row["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            gt = row["ground_truth"].strip() == "True"
            correct += int(gt == (ans == "yes"))
            total += 1
    return correct / total * 100 if total else None


def acc_localize_single(path: str) -> float | None:
    correct = total = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                gt = int(row["ground_truth"].strip())
                ans = int(row["answer"].strip()) - 1
                correct += int(gt == ans)
                total += 1
            except (ValueError, KeyError):
                pass
    return correct / total * 100 if total else None


def acc_localize_pair(path: str) -> float | None:
    correct = total = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                gt = set(ast.literal_eval(row["ground_truth"].strip()))
                ans = set(int(x) - 1 for x in row["answer"].strip().split(","))
                correct += int(gt == ans)
                total += 1
            except Exception:
                pass
    return correct / total * 100 if total else None


LOADERS = {
    "corrupt_detect": acc_binary,
    "corrupt_localize": acc_localize_single,
    "swap_detect": acc_binary,
    "swap_localize": acc_localize_pair,
}


def mean_across_datasets(task: str, mdir: str) -> float | None:
    accs = []
    for ds in DATASETS:
        path = os.path.join(BASE, mdir, f"{task}_{ds}.csv")
        if os.path.exists(path):
            a = LOADERS[task](path)
            if a is not None:
                accs.append(a)
    return sum(accs) / len(accs) if accs else None


def main() -> None:
    # accs[task][model_label] = float
    accs: dict[str, list[float]] = {t: [] for t in TASKS}
    for label, mdir in MODELS:
        for task in TASKS:
            val = mean_across_datasets(task, mdir)
            accs[task].append(val or 0.0)

    n_tasks = len(TASKS)
    n_models = len(MODELS)
    x = np.arange(n_tasks)
    bar_width = 0.8 / n_models

    blues = plt.cm.Blues(np.linspace(0.35, 0.85, n_models))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, _) in enumerate(MODELS):
        ys = [accs[task][i] for task in TASKS]
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            offsets,
            ys,
            bar_width,
            label=label,
            color=blues[i],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Model size",
        fontsize=18,
        title_fontsize=18,
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()

    out_path = os.path.join(BASE, "qwen3_vl_scaling.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
