"""Print delta table: no-scene-desc results minus main results.

Positive = better without scene description, negative = worse.

Usage:
  python plotting/print_no_scene_desc_delta_table.py            # LaTeX + plain
  python plotting/print_no_scene_desc_delta_table.py --plain    # plain only
  python plotting/print_no_scene_desc_delta_table.py --root .   # run from repo root
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DATASETS    = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
DS_ABBREV   = {"clevrer": "CL", "craft": "CR", "drive-lm": "DR", "mtl-aqa": "MT"}
TASKS       = ["swap_detect", "swap_localize", "corrupt_detect", "corrupt_localize"]
TASK_LABELS = {
    "corrupt_detect":   "Frame Detect",
    "corrupt_localize": "Frame Localize",
    "swap_detect":      "Temporal Detect",
    "swap_localize":    "Temporal Localize",
}
INCLUDE = {"qwen2-5-vl-7b", "qwen3-vl-8b", "gemma-4-e4b", "intern-vl-3", "intern-vl-3-5"}

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
# Accuracy helpers
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


def compute_cells(res_dir: Path) -> dict[CellKey, float | None]:
    cells: dict[CellKey, float | None] = {}
    for task in TASKS:
        for ds in DATASETS:
            path = res_dir / f"{task}_{ds}.csv"
            cells[(task, ds)] = task_accuracy(path, task) if path.exists() else None
    return cells


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(root: Path) -> list[str]:
    return [
        d.name
        for d in sorted((root / "results").iterdir())
        if d.is_dir() and (d / "no-scene-desc").is_dir()
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "-"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}"


def _cell_color(v: float, max_abs: float, cap: float = 60.0) -> str:
    """Return a LaTeX cellcolor prefix, sqrt-scaled so small deltas stay visible."""
    if max_abs == 0 or v == 0:
        return ""
    intensity = round((abs(v) / max_abs) ** 0.5 * cap)
    color = "green" if v > 0 else "red"
    return rf"\cellcolor{{{color}!{intensity}}}"


def print_plain(rows):
    abbrevs = [DS_ABBREV[ds] for ds in DATASETS]
    col_w   = 7
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

    for name, deltas in rows:
        print(f"{name:<{name_w}}", end="")
        for task in TASKS:
            for ds in DATASETS:
                v = deltas.get((task, ds))
                print(f"  {_fmt_delta(v):>{col_w}}", end="")
        print()

    print(sep)


def print_latex(rows):
    n_ds    = len(DATASETS)
    abbrevs = [DS_ABBREV[ds] for ds in DATASETS]

    # Find max absolute delta across all cells for normalising colour intensity
    all_vals = [v for _, deltas in rows for v in deltas.values() if v is not None]
    max_abs  = max((abs(v) for v in all_vals), default=1.0)

    col_spec = "l" + "".join(" " + "c" * n_ds for _ in TASKS)
    print(r"% Requires \usepackage[table]{xcolor} in the preamble")
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

    for name, deltas in rows:
        line = f"    {name}"
        for task in TASKS:
            for ds in DATASETS:
                v = deltas.get((task, ds))
                if v is None:
                    line += " & -"
                else:
                    color  = _cell_color(v, max_abs)
                    text   = _fmt_delta(v)
                    line  += f" & {color}{text}"
        print(line + r" \\")

    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Change in accuracy (\%) after removing scene descriptions."
          r" Positive values ({\color{green!70!black}green}) indicate improved performance"
          r" without scene descriptions, while negative values ({\color{red}red}) indicate"
          r" degradation. Removing scene descriptions has the largest effect on frame-level"
          r" localization, whereas temporal anomaly performance changes only modestly.}")
    print(r"    \label{tab:no_scene_desc_delta}")
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
    models = [m for m in discover_models(root) if m in INCLUDE]
    if not models:
        sys.exit("No model directories with no-scene-desc/ found in results/.")

    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    models.sort(key=lambda m: (order_idx.get(m, 999), m))

    rows = []
    for model in models:
        main_cells = compute_cells(root / "results" / model)
        nsd_cells  = compute_cells(root / "results" / model / "no-scene-desc")

        if all(v is None for v in nsd_cells.values()):
            continue

        deltas: dict[CellKey, float | None] = {}
        for key in main_cells:
            m = main_cells[key]
            n = nsd_cells.get(key)
            deltas[key] = round(n - m, 1) if (m is not None and n is not None) else None

        rows.append((MODEL_NAMES.get(model, model), deltas))

    if not rows:
        sys.exit("No data found.")

    if args.plain:
        print_plain(rows)
    else:
        print("\n=== Plain text (no-scene-desc minus main) ===\n")
        print_plain(rows)
        print("\n=== LaTeX ===\n")
        print_latex(rows)


if __name__ == "__main__":
    main()
