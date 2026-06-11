"""Print a LaTeX table of point-biserial correlations between LPIPS and model
answer (yes=1 / no=0) for each model × dataset combination.

Usage:
  python plotting/print_lpips_correlation_table.py
  python plotting/print_lpips_correlation_table.py --root .
"""
import argparse
import csv
from pathlib import Path

from scipy import stats

MODELS = [
    ("Qwen2.5-VL-7B",  "results-qwen2-5-vl-7b"),
    ("Qwen3-VL-8B",    "results-qwen3-vl-8b"),
    ("InternVL3.5-8B", "results-intern-vl-3-5"),
    ("Gemma-4-E4B",    "results-gemma-4-e4b"),
    ("InternVL3-8B",   "results-intern-vl-3"),
    ("Molmo-7B",       "results-molmo-7b"),
]
DATASETS    = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
DS_LABELS   = {"clevrer": "CL", "craft": "CR",
               "drive-lm": "DR", "mtl-aqa": "MT"}


def load(path: str) -> tuple[list[float], list[int]]:
    import ast
    lpips_vals, correct = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            lv = r.get("lpips", "").strip()
            if not lv:
                continue
            try:
                gt  = set(ast.literal_eval(r["ground_truth"].strip()))
                ans = set(int(x) - 1 for x in r["answer"].strip().split(","))
                lpips_vals.append(float(lv))
                correct.append(int(gt == ans))
            except Exception:
                pass
    return lpips_vals, correct


def sig_stars(p: float) -> str:
    if p < 0.001:
        return r"^{***}"
    if p < 0.01:
        return r"^{**}"
    if p < 0.05:
        return r"^{*}"
    return ""


def fmt_cell(r: float, p: float) -> str:
    import math
    if math.isnan(r):
        return r"---"
    sign = "+" if r >= 0 else "-"
    val  = f"{abs(r):.2f}"
    if p < 0.05:
        return rf"${sign}\textbf{{{val}}}$"
    return rf"${sign}{val}$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="..", help="Repo root")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    col_spec  = "l" + "c" * len(DATASETS)
    ds_header = " & ".join(DS_LABELS[ds] for ds in DATASETS)

    print(r"\begin{table}[h]")
    print(r"    \centering")
    print(r"    \small")
    print(rf"    \begin{{tabular}}{{{col_spec}}}")
    print(r"    \toprule")
    print(rf"    Model & {ds_header} \\")
    print(r"    \midrule")

    for label, mdir in MODELS:
        cells = []
        for ds in DATASETS:
            path = root / mdir / f"swap_localize_{ds}.csv"
            if not path.exists():
                cells.append("---")
                continue
            lp, correct = load(str(path))
            r, p = stats.pointbiserialr(correct, lp)
            cells.append(fmt_cell(r, p))

        print(f"    {label} & " + " & ".join(cells) + r" \\")

    # mean LPIPS per dataset as context row
    print(r"    \midrule")
    mean_cells = []
    for ds in DATASETS:
        path = root / "results-qwen3-vl-8b" / f"swap_localize_{ds}.csv"
        vals = []
        if path.exists():
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    lv = r.get("lpips", "").strip()
                    if lv:
                        vals.append(float(lv))
        mean_cells.append(f"${sum(vals)/len(vals):.2f}$" if vals else "---")
    print(r"    \textit{Mean LPIPS} & " + " & ".join(mean_cells) + r" \\")

    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Point-biserial correlation between LPIPS perceptual distance"
          r" of the swapped frame pair and swap localization correctness"
          r" (correct\,=\,1, wrong\,=\,0)."
          r" CL\,=\,CLEVRER, CR\,=\,CRAFT, DR\,=\,DriveLM, MT\,=\,MTL-AQA."
          r" \textbf{Bold} indicates $p < 0.05$."
          r" `---' indicates undefined correlation (constant model output).}")
    print(r"    \label{tab:lpips_correlation}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
