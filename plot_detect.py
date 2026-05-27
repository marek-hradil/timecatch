"""Plot binary-detection accuracy across datasets, grouped by sequence length.

Pass any number of result CSVs that share the same task (corrupt_detect OR
swap_detect). craft-long is auto-skipped — it has different sequence lengths
than the rest, so mixing it in clutters the x-axis.

Usage:
  python plot_detect.py results-qwen2-5-vl-7b/corrupt_detect_*.csv
  python plot_detect.py results-qwen2-5-vl-7b/swap_detect_*.csv
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt


def parse_filename(path: str) -> tuple[str, str]:
    base = os.path.basename(path).removesuffix(".csv")
    m = re.match(r"^(corrupt_detect|swap_detect)_(.+)$", base)
    if not m:
        raise ValueError(f"Can't parse task/dataset from filename: {base}")
    return m.group(1), m.group(2)


def load_rows(csv_path: str):
    rows, invalid = [], 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                invalid += 1
                continue
            gt = r["ground_truth"].strip() == "True"
            rows.append({"seq_len": int(r["seq_len"]),
                         "correct": gt == (ans == "yes")})
    return rows, invalid


def main(paths: list[str]) -> None:
    task: str | None = None
    by_dataset: dict[str, list[dict]] = {}
    src_dir = os.path.dirname(os.path.abspath(paths[0])) or "."

    for p in paths:
        t, ds = parse_filename(p)
        if ds == "craft-long":
            print(f"  skip  {os.path.basename(p)}  (craft-long auto-excluded)")
            continue
        if task is None:
            task = t
        elif task != t:
            sys.exit(f"ERROR: mixed tasks ({task} vs {t}) — pass only one task's files")
        rows, invalid = load_rows(p)
        by_dataset[ds] = rows
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n * 100 if n else 0
        print(f"  load  {os.path.basename(p):40s} n={n:5d}  invalid={invalid:4d}  overall acc={acc:5.1f}%")

    if not by_dataset or task is None:
        sys.exit("No usable files")

    all_lens = sorted({r["seq_len"] for rs in by_dataset.values() for r in rs})
    datasets = sorted(by_dataset)

    accs:   dict[str, list[float | None]] = {ds: [] for ds in datasets}
    counts: dict[str, list[int]]          = {ds: [] for ds in datasets}
    for ds in datasets:
        for sl in all_lens:
            bucket = [r["correct"] for r in by_dataset[ds] if r["seq_len"] == sl]
            if bucket:
                accs[ds].append(sum(bucket) / len(bucket) * 100)
                counts[ds].append(len(bucket))
            else:
                accs[ds].append(None)
                counts[ds].append(0)

    # Grouped bars: width per-bar = 0.8 / num_datasets.
    width = 0.8 / max(1, len(datasets))
    x = list(range(len(all_lens)))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig_w = max(10.0, 1.6 * len(all_lens) * len(datasets) ** 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    for i, ds in enumerate(datasets):
        offset = (i - (len(datasets) - 1) / 2) * width
        ys = [a if a is not None else 0.0 for a in accs[ds]]
        bars = ax.bar([xi + offset for xi in x], ys, width,
                      label=ds, color=colors[i % len(colors)], edgecolor="white")
        for bar, n, a in zip(bars, counts[ds], accs[ds]):
            if a is None or n == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=50, color="salmon", linestyle="--", linewidth=1, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in all_lens])
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Accuracy (%)")
    model_hint = os.path.basename(src_dir).removeprefix("results-")
    title = f"{task} — accuracy by sequence length × dataset"
    if model_hint and model_hint != src_dir:
        title += f"\n(model: {model_hint})"
    ax.set_title(title)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", ncol=min(4, len(datasets) + 1))
    plt.tight_layout()

    out_path = os.path.join(src_dir, f"{task}_by_length.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_detect.py <results.csv> [...]")
    main(sys.argv[1:])
