"""Print a paired human-vs-model accuracy table on the human-study subset.

For each (dataset, task) pair, models are evaluated on the identical stimuli
that approved human participants saw. Accuracy is computed per-annotation:
each human annotation is one scored item — if two participants saw the same
stimulus, it counts twice on both the human and model sides.

Usage:
  python plotting/print_human_comparison.py               # LaTeX + plain
  python plotting/print_human_comparison.py --plain       # plain only
  python plotting/print_human_comparison.py --root .
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS    = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
DS_ABBREV   = {"clevrer": "CL", "craft": "CR", "drive-lm": "DR", "mtl-aqa": "MT"}
TASKS       = ["swap_detect", "swap_localize"]

MODEL_NAMES: dict[str, str] = {
    "qwen2-5-vl-7b":  "Qwen2.5-VL-7B",
    "qwen3-vl-8b":    "Qwen3-VL-8B",
    "gemma-4-e4b":    "Gemma-4-E4B",
    "intern-vl-3":    "InternVL3-8B",
    "intern-vl-3-5":  "InternVL3.5-8B",
}
MODEL_ORDER = ["qwen2-5-vl-7b", "qwen3-vl-8b", "gemma-4-e4b", "intern-vl-3", "intern-vl-3-5"]

CellKey = tuple[str, str]   # (task, dataset_stem)

# ---------------------------------------------------------------------------
# Answer parsing — reused from print_main_table_detailed.py
# ---------------------------------------------------------------------------

_PAIR_GT = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_PAIR_AN = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def _parse_swap(s: str) -> tuple[int, int] | None:
    s = s.strip()
    m = _PAIR_GT.match(s) or _PAIR_AN.match(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _acc_detect(rows: list[dict]) -> float | None:
    correct = total = 0
    for r in rows:
        ans = r["answer"].strip().lower()
        if ans not in ("yes", "no"):
            continue
        gt = str(r["ground_truth"]).strip() == "True"
        correct += int(gt == (ans == "yes"))
        total += 1
    return 100.0 * correct / total if total else None


def _acc_localize(rows: list[dict]) -> float | None:
    correct = total = 0
    for r in rows:
        gt  = _parse_swap(str(r["ground_truth"]))
        ans = _parse_swap(str(r["answer"]))
        if gt is None or ans is None:
            continue
        # ground_truth stored 0-based; model answers 1-based → shift gt
        shifted = (gt[0] + 1, gt[1] + 1)
        correct += int(ans == shifted or ans == (shifted[1], shifted[0]))
        total += 1
    return 100.0 * correct / total if total else None

# ---------------------------------------------------------------------------
# Load human subset manifests
# ---------------------------------------------------------------------------

def load_manifests(root: Path) -> dict[CellKey, dict]:
    """Returns {(task, ds_stem): manifest_dict}"""
    subset_dir = root / "human_subset"
    manifests: dict[CellKey, dict] = {}
    for task in ("detect", "localize"):
        for ds_stem in DATASETS:
            path = subset_dir / f"swap_{task}_{ds_stem}.json"
            if path.exists():
                with open(path) as f:
                    manifests[(f"swap_{task}", ds_stem)] = json.load(f)
    return manifests


# ---------------------------------------------------------------------------
# Human accuracy from manifests
# ---------------------------------------------------------------------------

def human_accuracy(manifests: dict[CellKey, dict]) -> dict[CellKey, float | None]:
    acc: dict[CellKey, float | None] = {}
    for (task, ds_stem), manifest in manifests.items():
        correct = total = 0
        for stimulus in manifest["stimuli"]:
            gt_raw = stimulus["ground_truth"]
            for ann in stimulus["annotations"]:
                ha = ann["human_answer"]
                if task == "swap_detect":
                    gt  = bool(gt_raw)
                    ans = bool(ha) if isinstance(ha, bool) else str(ha).lower() == "true"
                    correct += int(gt == ans)
                    total += 1
                else:  # swap_localize
                    if not isinstance(gt_raw, list) or not isinstance(ha, list):
                        continue
                    gt_t  = (int(gt_raw[0]), int(gt_raw[1]))
                    ans_t = (int(ha[0]), int(ha[1]))
                    correct += int(gt_t == ans_t or gt_t == (ans_t[1], ans_t[0]))
                    total += 1
        acc[(task, ds_stem)] = 100.0 * correct / total if total else None
    return acc


# ---------------------------------------------------------------------------
# Model accuracy from results CSVs
# ---------------------------------------------------------------------------

def load_model_results(root: Path) -> dict[str, dict[CellKey, float | None]]:
    """Returns {model_key: {(task, ds_stem): accuracy}}"""
    results: dict[str, dict[CellKey, float | None]] = {}
    for model_key in MODEL_ORDER:
        model_dir = root / "results" / model_key
        if not model_dir.exists():
            continue
        cells: dict[CellKey, float | None] = {}
        for task in ("swap_detect", "swap_localize"):
            for ds_stem in DATASETS:
                csv_path = model_dir / f"human_{task}_{ds_stem}.csv"
                if not csv_path.exists():
                    cells[(task, ds_stem)] = None
                    continue
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
                if task == "swap_detect":
                    cells[(task, ds_stem)] = _acc_detect(rows)
                else:
                    cells[(task, ds_stem)] = _acc_localize(rows)
        if any(v is not None for v in cells.values()):
            results[model_key] = cells
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:.1f}"
    return rf"\textbf{{{s}}}" if bold else s


def print_plain(
    human: dict[CellKey, float | None],
    model_results: dict[str, dict[CellKey, float | None]],
    manifests: dict[CellKey, dict],  # noqa: ARG001 — used for stimulus counts
) -> None:
    abbrevs = [DS_ABBREV[ds] for ds in DATASETS]
    col_w  = 7
    name_w = 18

    print(f"{'Model':<{name_w}}", end="")
    for task in TASKS:
        label = task.replace("_", " ")
        print(f"  {label:^{len(abbrevs) * (col_w + 2) - 1}}", end="")
    print()

    print(f"{'':>{name_w}}", end="")
    for _ in TASKS:
        for a in abbrevs:
            print(f"  {a:>{col_w}}", end="")
    print()

    # stimulus counts
    print(f"{'n (stimuli)':<{name_w}}", end="")
    for task in TASKS:
        for ds_stem in DATASETS:
            m = manifests.get((task, ds_stem))
            n = len(m["stimuli"]) if m else 0
            print(f"  {n:>{col_w}}", end="")
    print()

    sep = "-" * (name_w + len(TASKS) * len(DATASETS) * (col_w + 2))
    print(sep)

    # human row
    print(f"{'Human':<{name_w}}", end="")
    for task in TASKS:
        for ds_stem in DATASETS:
            v = human.get((task, ds_stem))
            print(f"  {f'{v:.1f}' if v is not None else '—':>{col_w}}", end="")
    print()
    print(sep)

    for model_key in MODEL_ORDER:
        cells = model_results.get(model_key)
        if cells is None:
            continue
        name = MODEL_NAMES.get(model_key, model_key)
        print(f"{name:<{name_w}}", end="")
        for task in TASKS:
            for ds_stem in DATASETS:
                v = cells.get((task, ds_stem))
                print(f"  {f'{v:.1f}' if v is not None else '—':>{col_w}}", end="")
        print()


def print_latex(
    human: dict[CellKey, float | None],
    model_results: dict[str, dict[CellKey, float | None]],
) -> None:
    abbrevs = [DS_ABBREV[ds] for ds in DATASETS]
    task_labels = {"swap_detect": "Detection", "swap_localize": "Localization"}

    print(r"\begin{table}[t]")
    print(r"    \centering")
    print(r"    \small")
    print(r"    \begin{tabular}{lcccc}")
    print(r"    \toprule")
    print(r"    Task & " + " & ".join(abbrevs) + r" \\")
    print(r"    \midrule")

    for t_idx, task in enumerate(TASKS):
        if t_idx > 0:
            print(r"    \midrule")
        print(f"    {task_labels[task]} & & & & \\\\")
        for model_key in MODEL_ORDER:
            cells = model_results.get(model_key)
            if cells is None:
                continue
            name = MODEL_NAMES.get(model_key, model_key)
            line = f"    {name}"
            for ds_stem in DATASETS:
                v = cells.get((task, ds_stem))
                line += " & " + _fmt(v)
            print(line + r" \\")
        # Human row last
        human_line = r"    Human"
        for ds_stem in DATASETS:
            v = human.get((task, ds_stem))
            human_line += " & " + (f"{v:.1f}" if v is not None else "—")
        print(human_line + r" \\")

    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Temporal anomaly detection and localization accuracy (\%) on the"
          r" human study subset. Humans consistently outperform all evaluated VLMs across"
          r" datasets, highlighting a substantial gap between human and model temporal"
          r" reasoning. CL\,=\,clevrer, CR\,=\,craft, DR\,=\,drive-lm, MT\,=\,mtl-aqa.}")
    print(r"    \label{tab:human_results}")
    print(r"\end{table}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--root", default="..", help="Repo root")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    manifests = load_manifests(root)
    if not manifests:
        sys.exit("No human_subset/*.json manifests found. Run export_human_subset.py first.")

    human = human_accuracy(manifests)
    model_results = load_model_results(root)

    if args.plain:
        print_plain(human, model_results, manifests)
    else:
        print("\n=== Plain text ===\n")
        print_plain(human, model_results, manifests)
        print("\n=== LaTeX ===\n")
        print_latex(human, model_results)


if __name__ == "__main__":
    main()
