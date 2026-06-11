"""Plot localization exact-match accuracy for ONE task on ONE dataset, across multiple models.

Usage:
  python plot_models_localize.py results-*/corrupt_localize_craft-long.csv
  python plot_models_localize.py results-*/swap_localize_craft-long.csv
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
from colors import CHANCE_COLOR, MODEL_PALETTE

_SWAP_GT_RE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_INT_RE     = re.compile(r"^\d+$")
_PAIR_RE    = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def parse_path(path: str) -> tuple[str, str, str]:
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    model = parent.removeprefix("results-")
    base = os.path.basename(path).removesuffix(".csv")
    m = re.match(r"^(corrupt_localize|swap_localize)_(.+)$", base)
    if not m:
        raise ValueError(f"Can't parse task/dataset from filename: {base}")
    return m.group(1), m.group(2), model


def parse_corrupt(s: str):
    s = s.strip()
    return (int(s),) if _INT_RE.match(s) else None


def parse_swap(s: str):
    s = s.strip()
    m = _SWAP_GT_RE.match(s) or _PAIR_RE.match(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_rows(csv_path: str, task: str):
    parse = parse_swap if task == "swap_localize" else parse_corrupt
    rows, invalid = [], 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            gt = parse(r["ground_truth"])
            ans = parse(r["answer"])
            if gt is None or ans is None:
                invalid += 1
                continue
            shifted_gt = tuple(g + 1 for g in gt)
            if task == "swap_localize":
                exact = ans == shifted_gt or ans == (shifted_gt[1], shifted_gt[0])
            else:
                exact = ans == shifted_gt
            rows.append({"seq_len": int(r["seq_len"]), "exact": exact})
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
            sys.exit(f"ERROR: mixed tasks ({task} vs {t})")
        if dataset is None:
            dataset = ds
        elif dataset != ds:
            sys.exit(f"ERROR: mixed datasets ({dataset} vs {ds})")
        rows, invalid = load_rows(p, t)
        if not rows:
            print(f"  skip  {p}  (no valid rows)")
            continue
        by_model[model] = rows
        n = len(rows)
        acc = sum(r["exact"] for r in rows) / n * 100 if n else 0
        print(f"  load  {model:18s} from {os.path.basename(p):40s} n={n:5d}  invalid={invalid:4d}  acc={acc:5.1f}%")

    if not by_model or task is None or dataset is None:
        sys.exit("No usable files")

    all_lens = sorted({r["seq_len"] for rs in by_model.values() for r in rs})
    models = sorted(by_model)

    accs:   dict[str, list[float | None]] = {m: [] for m in models}
    counts: dict[str, list[int]]          = {m: [] for m in models}
    for m in models:
        for sl in all_lens:
            bucket = [r["exact"] for r in by_model[m] if r["seq_len"] == sl]
            if bucket:
                accs[m].append(sum(bucket) / len(bucket) * 100)
                counts[m].append(len(bucket))
            else:
                accs[m].append(None)
                counts[m].append(0)

    if task == "corrupt_localize":
        chance = [100.0 / l for l in all_lens]
    else:
        chance = [100.0 / (l - 1) if l > 1 else 0.0 for l in all_lens]

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

    ax.plot(x, chance, color=CHANCE_COLOR, linestyle="--", linewidth=1,
            marker="o", markersize=4, label="chance (per length)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in all_lens])
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Exact match accuracy (%, 1-based-shifted)")
    ax.set_title(f"{task} on {dataset} — exact match by sequence length × model")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", ncol=min(4, len(models) + 1))
    plt.tight_layout()

    out_name = f"{task}_{dataset}_by_model.png"
    plt.savefig(out_name, dpi=120)
    plt.close(fig)
    print(f"\n→ {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_models_localize.py <results.csv> [...]")
    main(sys.argv[1:])
