"""Print results table for the no-scene-description ablation.

Reads from results/<model>/no-scene-desc/ instead of the main model directory.

Usage:
  python plotting/print_no_scene_desc_table.py            # LaTeX + plain
  python plotting/print_no_scene_desc_table.py --plain    # plain only
  python plotting/print_no_scene_desc_table.py --root .   # run from repo root
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DATASETS    = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
DS_ABBREV   = {"clevrer": "CL", "craft": "CR", "drive-lm": "DR", "mtl-aqa": "MT"}
TASKS       = ["corrupt_detect", "corrupt_localize", "swap_detect", "swap_localize"]
TASK_LABELS = {
    "corrupt_detect":   "corrupt\\_detect",
    "corrupt_localize": "corrupt\\_localize",
    "swap_detect":      "swap\\_detect",
    "swap_localize":    "swap\\_localize",
}

CellKey = tuple[str, str]

MODEL_NAMES: dict[str, str] = {
    "qwen2-5-vl-7b":  "Qwen2.5-VL-7B",
    "qwen3-vl-2b":    "Qwen3-VL-2B",
    "qwen3-vl-4b":    "Qwen3-VL-4B",
    "qwen3-vl-8b":    "Qwen3-VL-8B",
    "qwen3-vl-32b":   "Qwen3-VL-32B",
    "intern-vl":      "InternVL-2",
    "intern-vl-3":    "InternVL3-8B",
    "intern-vl-3-5":  "InternVL3.5-8B",
    "molmo-7b":       "Molmo-7B",
    "gemma-4-e4b":    "Gemma-4-E4B",
}
MODEL_ORDER = [
    "qwen2-5-vl-7b", "qwen3-vl-8b", "gemma-4-e4b", "intern-vl-3",
    "intern-vl-3-5", "molmo-7b", "intern-vl",
    "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-32b",
]

# ---------------------------------------------------------------------------
# Accuracy helpers (identical to print_main_table_detailed.py)
# ---------------------------------------------------------------------------

_INT_RE  = re.compile(r"^\d+$")
_PAIR_GT = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_PAIR_AN = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def _parse_corrupt(s: str):
    s = s.strip()
    return (int(s),) if _INT_RE.match(s) else None


def _parse_swap(s: str):
    s = s.strip()
    m = _PAIR_GT.match(s) or _PAIR_AN.match(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _accuracy_detect(path: Path) -> float | None:
    correct = total = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            gt = r["ground_truth"].strip() == "True"
            correct += int(gt == (ans == "yes"))
            total += 1
    return 100.0 * correct / total if total else None


def _accuracy_localize(path: Path, task: str) -> float | None:
    parse = _parse_swap if task == "swap_localize" else _parse_corrupt
    correct = total = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gt  = parse(r["ground_truth"])
            ans = parse(r["answer"])
            if gt is None or ans is None:
                continue
            shifted = tuple(g + 1 for g in gt)
            if task == "swap_localize":
                exact = ans == shifted or ans == (shifted[1], shifted[0])
            else:
                exact = ans == shifted
            correct += int(exact)
            total += 1
    return 100.0 * correct / total if total else None


def task_accuracy(path: Path, task: str) -> float | None:
    if task in ("corrupt_detect", "swap_detect"):
        return _accuracy_detect(path)
    return _accuracy_localize(path, task)


# ---------------------------------------------------------------------------
# Chance baseline (from main results dir for seq_len distribution)
# ---------------------------------------------------------------------------

def compute_chance(root: Path) -> dict[CellKey, float]:
    ref_dir = next(
        (d for d in sorted((root / "results").iterdir()) if d.is_dir() and (d / "no-scene-desc").is_dir()),
        None,
    )
    chance: dict[CellKey, float] = {}
    for task in TASKS:
        for ds in DATASETS:
            if task in ("corrupt_detect", "swap_detect"):
                chance[(task, ds)] = 50.0
                continue
            lengths: list[int] = []
            if ref_dir is not None:
                path = ref_dir / "no-scene-desc" / f"{task}_{ds}.csv"
                if path.exists():
                    with open(path, newline="") as f:
                        for r in csv.DictReader(f):
                            try:
                                lengths.append(int(r["seq_len"]))
                            except (KeyError, ValueError):
                                pass
            if not lengths:
                chance[(task, ds)] = 0.0
                continue
            if task == "corrupt_localize":
                mean_p = sum(100.0 / n for n in lengths) / len(lengths)
            else:
                mean_p = sum(200.0 / (n * (n - 1)) for n in lengths) / len(lengths)
            chance[(task, ds)] = mean_p
    return chance


# ---------------------------------------------------------------------------
# Discovery + loading from no-scene-desc subdirectory
# ---------------------------------------------------------------------------

def discover_models(root: Path) -> list[str]:
    return [
        d.name
        for d in sorted((root / "results").iterdir())
        if d.is_dir() and (d / "no-scene-desc").is_dir()
    ]


def compute_cells(root: Path, model: str) -> dict[CellKey, float | None]:
    res_dir = root / "results" / model / "no-scene-desc"
    cells: dict[CellKey, float | None] = {}
    for task in TASKS:
        for ds in DATASETS:
            path = res_dir / f"{task}_{ds}.csv"
            cells[(task, ds)] = task_accuracy(path, task) if path.exists() else None
    return cells


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "-"
    s = f"{v:.1f}"
    return rf"\textbf{{{s}}}" if bold else s


def print_plain(rows, best, chance):
    abbrevs = [DS_ABBREV[ds] for ds in DATASETS]
    col_w   = 6
    name_w  = 18

    print(f"{'Model':<{name_w}}", end="")
    for task in TASKS:
        print(f"  {task:^{len(abbrevs) * (col_w + 2) - 1}}", end="")
    print()

    print(f"{'':>{name_w}}", end="")
    for _ in TASKS:
        for a in abbrevs:
            print(f"  {a:>{col_w}}", end="")
    print()

    sep = "-" * (name_w + len(TASKS) * len(DATASETS) * (col_w + 2))
    print(sep)

    for name, cells in rows:
        print(f"{name:<{name_w}}", end="")
        for task in TASKS:
            for ds in DATASETS:
                v = cells.get((task, ds))
                is_best = v is not None and abs(v - best.get((task, ds), -1)) < 0.05
                s = ("*" if is_best else " ") + (f"{v:.1f}" if v is not None else "-")
                print(f"  {s:>{col_w}}", end="")
        print()

    print(sep)
    print(f"{'Chance':<{name_w}}", end="")
    for task in TASKS:
        for ds in DATASETS:
            print(f"  {chance[(task, ds)]:>{col_w}.1f}", end="")
    print()


def print_latex(rows, best, chance):
    n_ds    = len(DATASETS)
    abbrevs = [DS_ABBREV[ds] for ds in DATASETS]

    col_spec = "l" + "".join(" " + "c" * n_ds for _ in TASKS)
    print(r"\begin{table*}[t!]")
    print(r"    \centering")
    print(r"    \setlength{\tabcolsep}{4pt}")
    print(r"    \small")
    print(rf"    \begin{{tabular}}{{{col_spec}}}")
    print(r"    \toprule")

    col_idx = 2
    task_header = "    "
    rules = []
    for task in TASKS:
        task_header += rf" & \multicolumn{{{n_ds}}}{{c}}{{{TASK_LABELS[task]}}}"
        rules.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + n_ds - 1}}}")
        col_idx += n_ds
    print(task_header + r" \\")
    print("    " + " ".join(rules))

    sub = "    "
    for _ in TASKS:
        for a in abbrevs:
            sub += f" & {a}"
    print(sub + r" \\")
    print(r"    \midrule")

    # Chance row first (matches the reference table style with Random at top)
    chance_line = "    Random"
    for task in TASKS:
        for ds in DATASETS:
            chance_line += f" & {chance[(task, ds)]:.1f}"
    print(chance_line + r" \\")
    print(r"    \midrule")

    for name, cells in rows:
        line = f"    {name}"
        for task in TASKS:
            for ds in DATASETS:
                v = cells.get((task, ds))
                bold = v is not None and abs(v - best.get((task, ds), -1)) < 0.05
                line += " & " + _fmt(v, bold)
        print(line + r" \\")

    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Detection and localization accuracy (\%) without scene descriptions."
          r" CL\,=\,clevrer, CR\,=\,craft, DR\,=\,drive-lm, MT\,=\,mtl-aqa."
          r" \textbf{Bold} indicates best model per column.}")
    print(r"    \label{tab:no_scene_desc_results}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--root", default="..", help="Repo root")
    args = parser.parse_args()

    root   = Path(args.root).resolve()
    models = discover_models(root)
    if not models:
        sys.exit("No model directories with no-scene-desc/ found in results/.")

    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    models.sort(key=lambda m: (order_idx.get(m, 999), m))

    data: dict[str, dict] = {}
    for model in models:
        cells = compute_cells(root, model)
        if all(v is None for v in cells.values()):
            continue
        data[model] = cells

    best: dict[CellKey, float] = {}
    for task in TASKS:
        for ds in DATASETS:
            vals = [d[(task, ds)] for d in data.values() if d.get((task, ds)) is not None]
            if vals:
                best[(task, ds)] = max(vals)

    rows   = [(MODEL_NAMES.get(m, m), data[m]) for m in models if m in data]
    chance = compute_chance(root)

    if args.plain:
        print_plain(rows, best, chance)
    else:
        print("\n=== Plain text ===\n")
        print_plain(rows, best, chance)
        print("\n=== LaTeX ===\n")
        print_latex(rows, best, chance)


if __name__ == "__main__":
    main()
