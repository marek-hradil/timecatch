"""
Human annotation accuracy — detect vs localize, grouped by dataset.
Usage: uv run python plot_annotations.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
DATASETS = ["CRAFT", "CLEVRER", "MTL-AQA", "drive_lm"]
TASKS = ["detect", "localize"]
LABELS = {"detect": "Detect", "localize": "Localize"}
COLORS = {"detect": "#4C72B0", "localize": "#DD8452"}


def accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    correct = sum(r["ground_truth"] == r["human_answer"] for r in rows)
    return correct / len(rows) * 100


def main() -> None:
    accs = {task: [accuracy(ANNOTATIONS_DIR / f"{task}_{ds}.csv") for ds in DATASETS] for task in TASKS}

    x = list(range(len(DATASETS)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, task in enumerate(TASKS):
        offset = (i - 0.5) * width
        xs = [xi + offset for xi in x]
        ys = [a if a is not None else 0.0 for a in accs[task]]
        bars = ax.bar(xs, ys, width, label=LABELS[task], color=COLORS[task], edgecolor="white")
        for bar, a in zip(bars, accs[task]):
            if a is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{a:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(50, color="salmon", linestyle="--", linewidth=1, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Human annotation accuracy — swap detect vs localize")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = Path(__file__).parent / "human_accuracy.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
