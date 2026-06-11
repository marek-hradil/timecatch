"""Print the main results table (LaTeX + plain text) averaged over all datasets.

Discovers all results/<model>/ directories automatically. Averages over the
four main datasets (clevrer, craft, drive-lm, mtl-aqa) weighted by sample count.

Usage:
  python print_main_table.py               # all discovered models
  python print_main_table.py --plain       # plain-text table only (no LaTeX)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
TASKS    = ["corrupt_detect", "corrupt_localize", "swap_detect", "swap_localize"]

# Human performance (swap tasks only)
HUMAN = {"swap_detect": 91.7, "swap_localize": 97.5}

# Chance baselines (binary detect = 50%; localize depends on frame-count distribution)
CHANCE = {"corrupt_detect": 50.0, "corrupt_localize": 17.3, "swap_detect": 50.0, "swap_localize": 7.9}

# Pretty display names for known model keys (auto-generated fallback: key as-is)
MODEL_NAMES: dict[str, str] = {
    "qwen2-5-vl-7b":   "Qwen2.5-VL-7B",
    "qwen3-vl-2b":     "Qwen3-VL-2B",
    "qwen3-vl-4b":     "Qwen3-VL-4B",
    "qwen3-vl-8b":     "Qwen3-VL-8B",
    "qwen3-vl-32b":    "Qwen3-VL-32B",
    "qwen3-5-9b":      "Qwen3.5-9B",
    "intern-vl":       "InternVL-2",
    "intern-vl-3":     "InternVL3-8B",
    "intern-vl-3-5":   "InternVL3.5-8B",
    "molmo-7b":        "Molmo-7B",
    "gemma-4-e4b":     "Gemma-4-E4B",
}

# Preferred row order (models not listed here appear at the end)
MODEL_ORDER = [
    "qwen2-5-vl-7b",
    "qwen3-vl-8b",
    "intern-vl-3-5",
    "gemma-4-e4b",
    "qwen3-5-9b",
    "molmo-7b",
    "intern-vl-3",
    "intern-vl",
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3-vl-32b",
]

# Model keys to exclude from the table (ablations, variants, etc.)
EXCLUDE = {"qwen3-vl-8b-thinking", "qwen3-vl-8b-video", "no-scene-desc", "prompt-ablation", "thinking-ablation"}

# ---------------------------------------------------------------------------
# Accuracy helpers (mirrors plot_models_detect.py / plot_models_localize.py)
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


def _accuracy_detect(path: Path) -> tuple[float, int] | None:
    correct = total = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            gt = r["ground_truth"].strip() == "True"
            correct += int(gt == (ans == "yes"))
            total += 1
    return (100.0 * correct / total, total) if total else None


def _accuracy_localize(path: Path, task: str) -> tuple[float, int] | None:
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
    return (100.0 * correct / total, total) if total else None


def task_accuracy(path: Path, task: str) -> tuple[float, int] | None:
    if task in ("corrupt_detect", "swap_detect"):
        return _accuracy_detect(path)
    return _accuracy_localize(path, task)


# ---------------------------------------------------------------------------
# Discovery + aggregation
# ---------------------------------------------------------------------------

def discover_models(root: Path) -> list[str]:
    models = []
    for d in sorted((root / "results").iterdir()):
        if d.is_dir():
            models.append(d.name)
    return models


def compute_averages(root: Path, model: str) -> dict[str, float | None]:
    res_dir = root / "results" / model
    avgs: dict[str, float | None] = {}
    for task in TASKS:
        total_correct = total_n = 0
        found = 0
        for ds in DATASETS:
            csv_path = res_dir / f"{task}_{ds}.csv"
            if not csv_path.exists():
                continue
            result = task_accuracy(csv_path, task)
            if result is None:
                continue
            acc, n = result
            total_correct += acc * n
            total_n += n
            found += 1
        avgs[task] = (total_correct / total_n) if total_n > 0 else None
    return avgs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "-"
    s = f"{v:.1f}"
    return rf"\textbf{{{s}}}" if bold else s


def print_plain(rows: list[tuple[str, dict]], best: dict[str, float]) -> None:
    col_w = 18
    header = f"{'Model':<{col_w}}" + "".join(f"{t:>18}" for t in TASKS)
    print(header)
    print("-" * len(header))
    for name, avgs in rows:
        vals = "".join(
            f"{'*' + f'{avgs[t]:.1f}' if avgs[t] == best.get(t) else f'{avgs[t]:.1f}' if avgs[t] is not None else '-':>18}"
            for t in TASKS
        )
        print(f"{name:<{col_w}}{vals}")
    print("-" * len(header))
    human_vals = "".join(f"{HUMAN[t]:>18.1f}" if t in HUMAN else f"{'—':>18}" for t in TASKS)
    chance_vals = "".join(f"{CHANCE[t]:>18.1f}" for t in TASKS)
    print(f"{'Human†':<{col_w}}{human_vals}")
    print(f"{'Chance':<{col_w}}{chance_vals}")


def print_latex(rows: list[tuple[str, dict]], best: dict[str, float]) -> None:
    task_labels = {
        "corrupt_detect":   r"corrupt\_detect",
        "corrupt_localize": r"corrupt\_localize",
        "swap_detect":      r"swap\_detect",
        "swap_localize":    r"swap\_localize",
    }
    print(r"\begin{table*}[ht!]")
    print(r"    \centering")
    print(r"    \captionsetup{width=0.85\linewidth}")
    print(r"    \begin{tabular}{lcccc}")
    print(r"    \toprule")
    header = "    Model & " + " & ".join(task_labels[t] for t in TASKS) + r" \\"
    print(header)
    print(r"    \midrule")
    for name, avgs in rows:
        cells = [f"\\textbf{{{avgs[t]:.1f}}}" if avgs[t] is not None and round(avgs[t], 1) == round(best.get(t, -1), 1)
                 else f"{avgs[t]:.1f}" if avgs[t] is not None else "-"
                 for t in TASKS]
        print(f"    {name:<30} & " + " & ".join(f"{c:>20}" for c in cells) + r" \\")
    print(r"    \midrule")
    human_cells = [f"{HUMAN[t]:.1f}" if t in HUMAN else "-" for t in TASKS]
    print(r"    Human$^\dagger$" + " & " + " & ".join(f"{c:>20}" for c in human_cells) + r" \\")
    chance_cells = [f"{CHANCE[t]:.1f}" for t in TASKS]
    print(r"    Chance" + " & " + " & ".join(f"{c:>20}" for c in chance_cells) + r" \\")
    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Detection and localization accuracy (\%) across all four tasks, averaged over all datasets."
          r" $^\dagger$Human performance is evaluated on swap tasks only."
          r" \textbf{Bold} indicates best model performance.}")
    print(r"    \label{tab:main_results}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", action="store_true", help="Print plain text only, no LaTeX")
    parser.add_argument("--root", default="..", help="Repo root (default: parent of this script)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    models = [m for m in discover_models(root) if m not in EXCLUDE]
    if not models:
        sys.exit("No model directories found in results/.")

    # Sort by preferred order, then alphabetically for unknowns
    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    models.sort(key=lambda m: (order_idx.get(m, 999), m))

    # Compute per-model averages
    data: dict[str, dict] = {}
    for model in models:
        avgs = compute_averages(root, model)
        if all(v is None for v in avgs.values()):
            continue
        data[model] = avgs

    # Find best per task (excluding human/chance)
    best: dict[str, float] = {}
    for task in TASKS:
        vals = [v[task] for v in data.values() if v[task] is not None]
        if vals:
            best[task] = max(vals)

    rows = [(MODEL_NAMES.get(m, m), data[m]) for m in models if m in data]

    if args.plain:
        print_plain(rows, best)
    else:
        print("\n=== Plain text ===\n")
        print_plain(rows, best)
        print("\n=== LaTeX ===\n")
        print_latex(rows, best)


if __name__ == "__main__":
    main()
