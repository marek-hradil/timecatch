"""Plot binary-detection accuracy for ONE task on ONE dataset, across multiple models.

Pass CSVs from different `results/<model>/` directories that share the same
task and dataset. The plot puts sequence length on x-axis and groups bars by model.

Usage:
  python plotting/plot_models_detect.py results/*/corrupt_detect_craft-long.csv
  python plotting/plot_models_detect.py results/*/swap_detect_craft-long.csv
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
from colors import CHANCE_COLOR, MODEL_PALETTE


def parse_path(path: str) -> tuple[str, str, str]:
    """Return (task, dataset, model_key) for a path like 'results/<model>/<task>_<dataset>.csv'."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    model = parent
    base = os.path.basename(path).removesuffix(".csv")
    m = re.match(r"^(corrupt_detect|swap_detect)_(.+)$", base)
    if not m:
        raise ValueError(f"Can't parse task/dataset from filename: {base}")
    return m.group(1), m.group(2), model


def load_rows(csv_path: str):
    rows, invalid = [], 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                invalid += 1
                continue
            gt = r["ground_truth"].strip() == "True"
            rows.append({"seq_len": int(r["seq_len"]), "correct": gt == (ans == "yes")})
    return rows, invalid


def main(paths: list[str]) -> None:
    task: str | None = None
    dataset: str | None = None
    by_model: dict[str, list[dict]] = {}

    for p in paths:
        t, ds, model = parse_path(p)
        if task is None:
            task = t
        elif task != t:
            sys.exit(f"ERROR: mixed tasks ({task} vs {t}) — pass only one task's files")
        if dataset is None:
            dataset = ds
        elif dataset != ds:
            sys.exit(f"ERROR: mixed datasets ({dataset} vs {ds}) — pass only one dataset's files")
        rows, invalid = load_rows(p)
        if not rows:
            print(f"  skip  {p}  (no valid rows)")
            continue
        by_model[model] = rows
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n * 100 if n else 0
        print(f"  load  {model:18s} from {os.path.basename(p):40s} n={n:5d}  invalid={invalid:4d}  acc={acc:5.1f}%")

    if not by_model or task is None or dataset is None:
        sys.exit("No usable files")

    all_lens = sorted({r["seq_len"] for rs in by_model.values() for r in rs})
    models = sorted(by_model)

    accs:   dict[str, list[float | None]] = {m: [] for m in models}
    counts: dict[str, list[int]]          = {m: [] for m in models}
    for m in models:
        for sl in all_lens:
            bucket = [r["correct"] for r in by_model[m] if r["seq_len"] == sl]
            if bucket:
                accs[m].append(sum(bucket) / len(bucket) * 100)
                counts[m].append(len(bucket))
            else:
                accs[m].append(None)
                counts[m].append(0)

    width = 0.8 / max(1, len(models))
    x = list(range(len(all_lens)))
    colors = MODEL_PALETTE
    fig_w = max(10.0, 1.6 * len(all_lens) * len(models) ** 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    for i, m in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ys = [a if a is not None else 0.0 for a in accs[m]]
        bars = ax.bar([xi + offset for xi in x], ys, width,
                      label=m, color=colors[i % len(colors)], edgecolor="white")
        for bar, n, a in zip(bars, counts[m], accs[m]):
            if a is None or n == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=50, color=CHANCE_COLOR, linestyle="--", linewidth=1, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in all_lens])
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{task} on {dataset} — accuracy by sequence length × model")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", ncol=min(4, len(models) + 1))
    plt.tight_layout()

    out_name = f"{task}_{dataset}_by_model.png"
    plt.savefig(out_name, dpi=120)
    plt.close(fig)
    print(f"\n→ {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_models_detect.py <results.csv> [...]")
    main(sys.argv[1:])
