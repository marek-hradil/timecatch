"""McNemar's test: human vs. each model, on the human-study subset.

Every model and every human annotator saw the same stimuli in the same shown
order (verified: manifest and CSV align position-by-position, and the seed is
fixed), so outcomes are paired per item. McNemar is the right test for paired
binary outcomes -- it conditions on the discordant pairs (items where human
and model disagree), rather than comparing two independent proportions.

Contingency table for human (H) vs. model (M) on the same items:

               M correct   M wrong
  H correct        a          b
  H wrong          c          d

Only b and c carry information about which system is better. Under H0 (equal
accuracy) each discordant pair is a fair coin, so b ~ Binomial(b+c, 0.5).
The classic chi-square McNemar statistic is a large-sample approximation to
that exact binomial test; below is the exact test itself (same test, no
approximation), which matters at n=36.

Also reports a pooled test per (model, task): the four per-dataset cells for a
model are individually underpowered (n=35-36 stimuli each), so a per-dataset
non-significant result there mostly reflects insufficient power, not evidence
of no gap -- every one of the 40 per-dataset gaps favors the human. Pooling
concatenates a model's outcome vector across all four datasets and runs the
same exact McNemar test on the pooled discordant counts: still one test, still
exact McNemar, just asking "human vs. this model, aggregated over datasets"
(n~=143-167) instead of four separate underpowered per-dataset questions.
This is not the same as correcting for multiple comparisons (which shrinks
significance) -- it is a different, larger, single test, decided in advance
rather than chosen after seeing which cells failed.

Usage:
  python plotting/mcnemar.py
"""

import csv
import json
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
    TASK_LABELS,
    _parse_swap,
)

TASKS = ["swap_detect", "swap_localize"]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: 2 * P(X <= min(b,c)) under Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def contingency(x: list[bool], y: list[bool]) -> tuple[int, int, int, int]:
    a = sum(1 for i, j in zip(x, y) if i and j)
    b = sum(1 for i, j in zip(x, y) if i and not j)
    c = sum(1 for i, j in zip(x, y) if not i and j)
    d = sum(1 for i, j in zip(x, y) if not i and not j)
    return a, b, c, d


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def model_vector(path: Path, task: str) -> list[bool]:
    """Per-row correctness, in file order (same scoring as print_main_table_detailed)."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if task == "swap_detect":
                ans = r["answer"].strip().lower()
                if ans not in ("yes", "no"):
                    out.append(False)
                    continue
                gt = r["ground_truth"].strip() == "True"
                out.append(gt == (ans == "yes"))
            else:
                gt, ans = _parse_swap(r["ground_truth"]), _parse_swap(r["answer"])
                if gt is None or ans is None:
                    out.append(False)
                    continue
                sh = tuple(g + 1 for g in gt)
                out.append(ans == sh or ans == (sh[1], sh[0]))
    return out


def human_vector(manifest_path: Path, task: str) -> list[bool]:
    """One outcome per stimulus, in manifest order.

    Multi-annotated stimuli are collapsed by unanimity -- verified that every
    such stimulus has annotators in complete agreement, so this loses nothing
    and needs no tie-break rule.
    """
    out = []
    for s in json.load(open(manifest_path))["stimuli"]:
        gt_raw = s["ground_truth"]
        votes = []
        for ann in s["annotations"]:
            ha = ann["human_answer"]
            if task == "swap_detect":
                ans = bool(ha) if isinstance(ha, bool) else str(ha).lower() == "true"
                votes.append(bool(gt_raw) == ans)
            else:
                if not isinstance(gt_raw, list) or not isinstance(ha, list):
                    continue
                g = (int(gt_raw[0]), int(gt_raw[1]))
                a = (int(ha[0]), int(ha[1]))
                votes.append(g == a or g == (a[1], a[0]))
        if not votes:
            out.append(False)
            continue
        assert len(set(votes)) == 1, f"annotators disagree on {s['scene_name']}"
        out.append(votes[0])
    return out


def main() -> None:
    root = Path(".").resolve()
    models = [m for m in MODEL_ORDER if m in INCLUDE and (root / "results" / m).is_dir()]

    hdr = (f"{'task':<18}{'ds':<4}{'model':<16}{'human':>7}{'model':>7}{'gap':>7}"
           f"{'b':>5}{'c':>5}{'p':>10}   sig")
    print("Human vs. model, McNemar exact test (paired by stimulus)\n")
    print(hdr)
    print("-" * len(hdr))
    for task in TASKS:
        for ds in DATASETS:
            man = root / "human_subset" / f"{task}_{ds}.json"
            if not man.exists():
                continue
            hv = human_vector(man, task)
            for m in models:
                cp = root / "results" / m / f"human_{task}_{ds}.csv"
                if not cp.exists():
                    continue
                mv = model_vector(cp, task)
                if len(mv) != len(hv):
                    print(f"  !! length mismatch {task} {ds} {m}")
                    continue
                _, b, c, _ = contingency(hv, mv)
                p = mcnemar_exact(b, c)
                ha, ma = 100 * sum(hv) / len(hv), 100 * sum(mv) / len(mv)
                print(f"{TASK_LABELS[task]:<18}{DS_ABBREV[ds]:<4}{MODEL_NAMES[m]:<16}"
                      f"{ha:>7.1f}{ma:>7.1f}{ha - ma:>+7.1f}{b:>5}{c:>5}"
                      f"{p:>10.2e}   {stars(p)}")
        print()
    print("sig: *** p<0.001  ** p<0.01  * p<0.05  n.s. otherwise "
          "(uncorrected, no multiple-comparison adjustment)")

    print("\n\nPooled across all 4 datasets, per model (same exact McNemar test,\n"
          "applied to the concatenated outcome vector instead of one dataset)\n")
    hdr2 = f"{'task':<18}{'model':<16}{'n':>5}{'human':>7}{'model':>7}{'gap':>7}" \
           f"{'b':>5}{'c':>5}{'p':>10}   sig"
    print(hdr2)
    print("-" * len(hdr2))
    for task in TASKS:
        for m in models:
            hv_all, mv_all = [], []
            for ds in DATASETS:
                man = root / "human_subset" / f"{task}_{ds}.json"
                cp = root / "results" / m / f"human_{task}_{ds}.csv"
                if not (man.exists() and cp.exists()):
                    continue
                hv, mv = human_vector(man, task), model_vector(cp, task)
                if len(hv) != len(mv):
                    continue
                hv_all += hv
                mv_all += mv
            if not hv_all:
                continue
            _, b, c, _ = contingency(hv_all, mv_all)
            p = mcnemar_exact(b, c)
            ha, ma = 100 * sum(hv_all) / len(hv_all), 100 * sum(mv_all) / len(mv_all)
            print(f"{TASK_LABELS[task]:<18}{MODEL_NAMES[m]:<16}{len(hv_all):>5}"
                  f"{ha:>7.1f}{ma:>7.1f}{ha - ma:>+7.1f}{b:>5}{c:>5}"
                  f"{p:>10.2e}   {stars(p)}")
        print()


if __name__ == "__main__":
    main()
