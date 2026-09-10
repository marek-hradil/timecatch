"""Main results table (tab:detailed_results) with Wilson score confidence intervals.

Reuses the exact scoring logic from print_main_table_detailed.py, but keeps the
raw (correct, total) counts instead of collapsing straight to a percentage, so a
binomial interval can be put on each cell.

Every cell is a proportion of independent Bernoulli trials: each sequence is
scored once, correct or not (see dataset.py -- sample_binary flips a coin per
scene to decide swap/no-swap, sample_position always swaps). So n equals the
number of sequences in the dataset, and Wilson applies directly.

Wilson is preferred over the textbook Wald interval because several cells sit
near the 0/1 boundary (99.6%, 9.8%), where Wald produces intervals that are
symmetric, too narrow, and can leave [0, 1] entirely.

Usage:
  python plotting/print_main_table_ci.py                 # LaTeX + plain
  python plotting/print_main_table_ci.py --plain         # plain only
  python plotting/print_main_table_ci.py --style stacked # CI under point (default)
  python plotting/print_main_table_ci.py --style pm      # inline +-halfwidth
  python plotting/print_main_table_ci.py --style bracket # inline [lo, hi]
  python plotting/print_main_table_ci.py --transpose     # models as columns
  python plotting/print_main_table_ci.py --conf 0.99
"""

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from print_main_table_detailed import (  # noqa: E402
    DATASETS,
    DS_ABBREV,
    INCLUDE,
    MODEL_NAMES,
    MODEL_ORDER,
    TASKS,
    TASK_LABELS,
    _parse_corrupt,
    _parse_swap,
    compute_chance,
)

# ---------------------------------------------------------------------------
# Counts (same scoring as print_main_table_detailed, but returning n)
# ---------------------------------------------------------------------------


def _counts_detect(path: Path) -> tuple[int, int]:
    correct = total = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            gt = r["ground_truth"].strip() == "True"
            correct += int(gt == (ans == "yes"))
            total += 1
    return correct, total


def _counts_localize(path: Path, task: str) -> tuple[int, int]:
    parse = _parse_swap if task == "swap_localize" else _parse_corrupt
    correct = total = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gt = parse(r["ground_truth"])
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
    return correct, total


def task_counts(path: Path, task: str) -> tuple[int, int]:
    if task in ("corrupt_detect", "swap_detect"):
        return _counts_detect(path)
    return _counts_localize(path, task)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

# Inverse normal CDF via Acklam's rational approximation -- avoids a scipy
# dependency in the plotting path, which is otherwise matplotlib-only.
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def _norm_ppf(p: float) -> float:
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


def wilson(correct: int, total: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (point, lo, hi) as percentages, using the Wilson score interval."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    z = _norm_ppf(1 - (1 - conf) / 2)
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return 100 * p, 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def collect(root: Path, conf: float):
    """-> {model: {(task, ds): (point, lo, hi, correct, total)}}"""
    out = {}
    for model in MODEL_ORDER:
        if model not in INCLUDE:
            continue
        cells = {}
        for task in TASKS:
            for ds in DATASETS:
                path = root / "results" / model / f"{task}_{ds}.csv"
                if not path.exists():
                    cells[(task, ds)] = None
                    continue
                c, n = task_counts(path, task)
                pt, lo, hi = wilson(c, n, conf)
                cells[(task, ds)] = (pt, lo, hi, c, n)
        out[model] = cells
    return out


def fmt_cell(cell, style: str, bold: bool) -> str:
    if cell is None:
        return "--"
    pt, lo, hi, _, _ = cell
    point = f"{pt:.1f}"
    if bold:
        point = rf"\textbf{{{point}}}"
    if style == "stacked":
        return rf"\cell{{{point}}}{{$\pm${(hi - lo) / 2:.1f}}}"
    if style == "bracket":
        return rf"{point}\,\ci{{[{lo:.1f},\,{hi:.1f}]}}"
    return rf"{point}\,\ci{{$\pm${(hi - lo) / 2:.1f}}}"


# Measured against acl.sty: \textwidth (the table* limit) is 455.24pt.
#   original table, no CIs, tabcolsep 4pt ...... 451.5pt
#   inline "+-x.x", \small, tabcolsep 2.4pt .... 615.8pt  (overfull by 160pt)
#   inline "+-x.x", \tiny,  tabcolsep 2pt ...... 489.0pt  (still overfull)
#   stacked cells, \small,  tabcolsep 3pt ...... 417.5pt  (fits, 38pt spare)
# Split into two table* blocks (9 columns each), inline "+-x.x", \small:
#   CI at 5.4pt ................................ 325.8pt
#   CI at 7pt .................................. 348.5pt  (chosen)
#   CI at 8pt .................................. 361.4pt
# Transposed (models as columns, task x dataset as rows), inline, CI at 7pt:
#   tabcolsep 1.5pt ............................ 384.2pt  (chosen)
#   tabcolsep 4pt .............................. 419.2pt
# Transposing is the one layout that keeps a single table AND a single caption
# AND inline CIs at readable size: 7 columns instead of 17. It is the default.
# Stacked wins because the CI line is narrower than the point estimate, so
# column widths are unchanged from the CI-free table -- the cost is row height,
# not width.
#
# Width is pinned by tabular*{\textwidth} + \extracolsep{\fill}, which spreads
# the slack between columns; a plain tabular is only as wide as its content and
# \centering then shows the remainder as gutters. Verified in the rendered PDF:
# the top rule and the caption block share the same left/right pixel bounds.
# (Measuring this with \sbox reads 2.25pt high -- \extracolsep{\fill} does not
#  resolve the same way outside a real \hsize. Trust the render, not the box.)
TABCOLSEP = {"stacked": "2pt", "pm": "1.5pt", "bracket": "1.5pt"}


TASK_GROUPS = [
    ("Temporal", ["swap_detect", "swap_localize"]),
    ("Frame-level", ["corrupt_detect", "corrupt_localize"]),
]


def print_macros(style: str) -> None:
    if style == "stacked":
        print(r"\newcommand{\cell}[2]{\shortstack{#1\\[-0.35ex]%")
        print(r"  {\fontsize{5.6}{6}\selectfont\color{black!55}#2}}}")
    else:
        # 7pt grey against 10pt black: readable, clearly subordinate.
        print(r"\newcommand{\ci}[1]{{\fontsize{7}{8}\selectfont\color{black!55}#1}}")


def _best_per_cell(data, tasks):
    """(task, ds) -> model with the highest point estimate."""
    best = {}
    for task in tasks:
        for ds in DATASETS:
            vals = [(d[(task, ds)][0], m) for m, d in data.items() if d[(task, ds)]]
            if vals:
                best[(task, ds)] = max(vals)[1]
    return best


def emit_wide(data, chance, style, tasks, best) -> None:
    """Models as rows, (task x dataset) as columns -- the original orientation."""
    print(r"    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l "
          + " ".join(["cccc"] * len(tasks)) + "@{}}")
    print(r"    \toprule")
    print("     & " + " & ".join(
        rf"\multicolumn{{4}}{{c}}{{{TASK_LABELS[t]}}}" for t in tasks) + r" \\")
    print("    " + " ".join(
        rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(tasks))))
    print("     & " + " & ".join(
        DS_ABBREV[d] for _ in tasks for d in DATASETS) + r" \\")
    print(r"    \midrule")
    rnd = " & ".join(f"{chance[(t, d)]:.1f}" for t in tasks for d in DATASETS)
    print(rf"     Random & {rnd} \\")
    print(r"     \midrule")
    for model in MODEL_ORDER:
        if model not in data:
            continue
        cells = " & ".join(
            fmt_cell(data[model][(t, d)], style, best.get((t, d)) == model)
            for t in tasks for d in DATASETS)
        print(rf"    {MODEL_NAMES[model]} & {cells} \\")
    print(r"    \bottomrule")
    print(r"    \end{tabular*}")


DS_FULL = {"clevrer": "CLEVRER", "craft": "CRAFT",
           "drive-lm": "DriveLM", "mtl-aqa": "MTL-AQA"}


def emit_transposed(data, chance, style, tasks, best) -> None:
    """(task x dataset) as rows, models as columns.

    Only 2 + len(models) columns wide instead of 17, so inline CIs fit at full
    size in a single table. Taller, but an appendix table has the vertical room.
    """
    models = [m for m in MODEL_ORDER if m in data]
    print(r"    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lc"
          + "c" * len(models) + "@{}}")
    print(r"    \toprule")
    print("     & Random & " + " & ".join(MODEL_NAMES[m] for m in models) + r" \\")
    ncols = 2 + len(models)
    for task in tasks:
        # Group heading is a banded row: full-width \midrule above and below, so
        # it reads as an ordinary row rather than a floating sub-caption.
        print(r"    \midrule")
        print(rf"    \multicolumn{{{ncols}}}{{@{{}}l}}"
              rf"{{\textit{{{TASK_LABELS[task]}}}}} \\")
        print(r"    \midrule")
        for ds in DATASETS:
            cells = " & ".join(
                fmt_cell(data[m][(task, ds)], style, best.get((task, ds)) == m)
                for m in models)
            print(rf"    \quad {DS_FULL[ds]} & {chance[(task, ds)]:.1f} & {cells} \\")
    print(r"    \bottomrule")
    print(r"    \end{tabular*}")


def print_latex(data, chance, conf: float, style: str, tasks=None,
                group_label: str = "", emit_macro: bool = True,
                transpose: bool = False) -> None:
    tasks = tasks or TASKS
    ncol = len(tasks) * len(DATASETS)
    best = _best_per_cell(data, tasks)

    if emit_macro:
        print(r"% Preamble: booktabs + xcolor (both already loaded in acl_latex.tex).")
        print_macros(style)
    print(r"\begin{table*}[t!]")
    print(r"    \centering")
    print(rf"    \setlength{{\tabcolsep}}{{{TABCOLSEP[style]}}}")
    print(r"    \small")
    (emit_transposed if transpose else emit_wide)(data, chance, style, tasks, best)

    pct = int(round(conf * 100))
    what = f"{group_label} anomaly detection and localization" if group_label \
        else "Detection and localization"
    suffix = "_" + group_label.split("-")[0].lower() if group_label else ""
    bold_note = ("bold marks the highest point estimate per row"
                 if transpose else
                 "bold marks the highest point estimate per column")
    print(rf"""    \caption{{{what} accuracy (\%) across datasets
    (the same runs as Table~\ref{{tab:detailed_results}}), with {pct}\% Wilson score
    intervals. Each cell is a proportion over independent sequences, one trial per
    sequence: $n$ = 4{{,}}997 / 858 / 696 / 338 for CLEVRER / CRAFT / DriveLM /
    MTL-AQA. Intervals are marginal and not corrected for multiple comparisons;
    {bold_note}, which is not itself a significance claim.}}""")
    print(rf"    \label{{tab:detailed_results_ci{suffix}}}")
    print(r"\end{table*}")
    print()
    print(rf"% ncol={ncol}; intervals are marginal, not corrected for multiple comparisons.")


def print_plain(data, chance, conf: float) -> None:
    pct = int(round(conf * 100))
    for task in TASKS:
        print(f"\n=== {TASK_LABELS[task]}  ({pct}% Wilson) ===")
        print(f"{'model':<16}" + "".join(f"{DS_ABBREV[d]:>26}" for d in DATASETS))
        print(f"{'Random':<16}" + "".join(
            f"{chance[(task, d)]:>26.1f}" for d in DATASETS))
        for model in MODEL_ORDER:
            if model not in data:
                continue
            row = ""
            for ds in DATASETS:
                cell = data[model][(task, ds)]
                if cell is None:
                    row += f"{'--':>26}"
                else:
                    pt, lo, hi, c, n = cell
                    row += f"{f'{pt:.1f} [{lo:.1f},{hi:.1f}] {c}/{n}':>26}"
            print(f"{MODEL_NAMES[model]:<16}{row}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--conf", default=0.95, type=float)
    ap.add_argument("--style", default="pm",
                    choices=["pm", "stacked", "bracket"])
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--wide", action="store_true",
                    help="original orientation (models as rows); needs --style "
                         "stacked or --split to fit")
    ap.add_argument("--transpose", action="store_true", default=True,
                    help="models as columns, (task x dataset) as rows: narrow "
                         "enough for inline CIs in one table")
    ap.add_argument("--split", action="store_true",
                    help="emit two table* blocks (Temporal / Frame-level) so inline "
                         "CIs fit at full size")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    data = collect(root, args.conf)
    chance = compute_chance(root)

    if not args.latex:
        print_plain(data, chance, args.conf)
    if not args.plain:
        print()
        if args.split:
            for i, (label, tasks) in enumerate(TASK_GROUPS):
                print_latex(data, chance, args.conf, args.style,
                            tasks=tasks, group_label=label, emit_macro=(i == 0))
                print()
        else:
            print_latex(data, chance, args.conf, args.style,
                        transpose=not args.wide)


if __name__ == "__main__":
    main()
