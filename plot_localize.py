"""Plot localization exact-match accuracy across datasets, grouped by sequence length.

Pass any number of result CSVs that share the same task (corrupt_localize OR
swap_localize). craft-long is auto-skipped (different seq_len range from the rest).

Note: the prompts say "Use 1-based indexing" but the scripts store ground truth
0-based, so a model that answers correctly in 1-based naturally looks off-by-one
on raw exact-match. This plot uses **1-based-shifted** exact match
(ground_truth + 1) so the metric reflects the model's actual behaviour.

Usage:
  python plot_localize.py results-qwen2-5-vl-7b/corrupt_localize_*.csv
  python plot_localize.py results-qwen2-5-vl-7b/swap_localize_*.csv
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt


_SWAP_GT_RE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_INT_RE     = re.compile(r"^\d+$")
_PAIR_RE    = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def parse_filename(path: str) -> tuple[str, str]:
    base = os.path.basename(path).removesuffix(".csv")
    m = re.match(r"^(corrupt_localize|swap_localize)_(.+)$", base)
    if not m:
        raise ValueError(f"Can't parse task/dataset from filename: {base}")
    return m.group(1), m.group(2)


def parse_corrupt(s: str) -> tuple[int] | None:
    s = s.strip()
    return (int(s),) if _INT_RE.match(s) else None


def parse_swap(s: str) -> tuple[int, int] | None:
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
            shifted_gt = tuple(g + 1 for g in gt)            # 1-based correction
            if task == "swap_localize":
                exact = ans == shifted_gt or ans == (shifted_gt[1], shifted_gt[0])
            else:
                exact = ans == shifted_gt
            rows.append({"seq_len": int(r["seq_len"]), "exact": exact})
    return rows, invalid


def main(paths: list[str], allow_craft_long: bool = False) -> None:
    task: str | None = None
    by_dataset: dict[str, list[dict]] = {}
    src_dir = os.path.dirname(os.path.abspath(paths[0])) or "."

    for p in paths:
        t, ds = parse_filename(p)
        if ds == "craft-long" and not allow_craft_long:
            print(f"  skip  {os.path.basename(p)}  (craft-long auto-excluded)")
            continue
        if task is None:
            task = t
        elif task != t:
            sys.exit(f"ERROR: mixed tasks ({task} vs {t}) — pass only one task's files")
        rows, invalid = load_rows(p, t)
        by_dataset[ds] = rows
        n = len(rows)
        acc = sum(r["exact"] for r in rows) / n * 100 if n else 0
        print(f"  load  {os.path.basename(p):42s} n={n:5d}  invalid={invalid:4d}  acc(1-based)={acc:5.1f}%")

    if not by_dataset or task is None:
        sys.exit("No usable files")

    all_lens = sorted({r["seq_len"] for rs in by_dataset.values() for r in rs})
    datasets = sorted(by_dataset)

    accs:   dict[str, list[float | None]] = {ds: [] for ds in datasets}
    counts: dict[str, list[int]]          = {ds: [] for ds in datasets}
    for ds in datasets:
        for sl in all_lens:
            bucket = [r["exact"] for r in by_dataset[ds] if r["seq_len"] == sl]
            if bucket:
                accs[ds].append(sum(bucket) / len(bucket) * 100)
                counts[ds].append(len(bucket))
            else:
                accs[ds].append(None)
                counts[ds].append(0)

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

    # Random-baseline curve — depends on task and seq_len.
    # corrupt_localize: 1/L (pick one frame out of L)
    # swap_localize:    1/(L-1) (pick one of L-1 possible adjacent swap positions)
    if task == "corrupt_localize":
        chance = [100.0 / l for l in all_lens]
    else:
        chance = [100.0 / (l - 1) if l > 1 else 0.0 for l in all_lens]
    ax.plot(x, chance, color="salmon", linestyle="--", linewidth=1,
            marker="o", markersize=4, label="chance (per length)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in all_lens])
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Exact match accuracy (%, 1-based-shifted)")
    model_hint = os.path.basename(src_dir).removeprefix("results-")
    title = f"{task} — exact match by sequence length × dataset"
    if model_hint and model_hint != src_dir:
        title += f"\n(model: {model_hint}, ground-truth shifted +1 to match 1-based prompt)"
    ax.set_title(title)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", ncol=min(4, len(datasets) + 1))
    plt.tight_layout()

    out_path = os.path.join(src_dir, f"{task}_by_length.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_localize.py <results.csv> [...]")
    main(sys.argv[1:])
