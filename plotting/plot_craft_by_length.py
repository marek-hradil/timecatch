"""4-panel line plot: CRAFT + CRAFT-long, all tasks, all available models.

Merges craft (seq_len 4–8) and craft-long (8–16) onto one x-axis.
A dotted vertical line marks the craft / craft-long boundary at seq_len 8.5.

Usage:
  python plotting/plot_craft_by_length.py                        # auto-discover results/*
  python plotting/plot_craft_by_length.py results/qwen3-vl-8b results/intern-vl-3-5
"""

import csv
import os
import re
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colors import CHANCE_COLOR, MODEL_PALETTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = ["corrupt_detect", "corrupt_localize", "swap_detect", "swap_localize"]
TASK_LABELS = {
    "corrupt_detect": "Frame Detect",
    "corrupt_localize": "Frame Localize",
    "swap_detect": "Temporal Detect",
    "swap_localize": "Temporal Localize",
}

MODEL_LABELS = {
    "qwen2-5-vl-7b": "Qwen2.5-VL-7B",
    "qwen3-vl-8b": "Qwen3-VL-8B",
    "intern-vl-3-5": "InternVL3.5",
    "intern-vl-3": "InternVL3",
    "intern-vl": "InternVL",
    "qwen3-vl-2b": "Qwen3-VL-2B",
    "qwen3-vl-4b": "Qwen3-VL-4B",
    "qwen3-vl-32b": "Qwen3-VL-32B",
    "gemma-4-e4b": "Gemma4-E4B",
    "molmo-7b": "Molmo-7B",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

_INT_RE = re.compile(r"^\d+$")
_SWAP_GT_RE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_PAIR_RE = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def load_binary(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            gt = r["ground_truth"].strip() == "True"
            rows.append({"seq_len": int(r["seq_len"]), "correct": gt == (ans == "yes")})
    return rows


def load_corrupt_localize(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g = _INT_RE.match(r["ground_truth"].strip())
            a = _INT_RE.match(r["answer"].strip())
            if not g or not a:
                continue
            rows.append(
                {
                    "seq_len": int(r["seq_len"]),
                    "correct": int(a.group()) == int(g.group()) + 1,
                }
            )
    return rows


def load_swap_localize(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gm = _SWAP_GT_RE.match(r["ground_truth"].strip()) or _PAIR_RE.match(
                r["ground_truth"].strip()
            )
            am = _PAIR_RE.match(r["answer"].strip())
            if not gm or not am:
                continue
            gt = (int(gm.group(1)) + 1, int(gm.group(2)) + 1)
            ans = (int(am.group(1)), int(am.group(2)))
            rows.append(
                {
                    "seq_len": int(r["seq_len"]),
                    "correct": ans == gt or ans == (gt[1], gt[0]),
                }
            )
    return rows


LOADERS = {
    "corrupt_detect": load_binary,
    "swap_detect": load_binary,
    "corrupt_localize": load_corrupt_localize,
    "swap_localize": load_swap_localize,
}


def load_model_task(model_dir: str, task: str) -> list[dict]:
    rows = []
    for ds in ("craft", "craft-long"):
        path = os.path.join(model_dir, f"{task}_{ds}.csv")
        if os.path.exists(path):
            rows.extend(LOADERS[task](path))
    return rows


def main(result_dirs: list[str]) -> None:
    # data[task][model_label] = {seq_len: [correct, ...]}
    buckets: dict[str, dict[str, dict[int, list[bool]]]] = {t: {} for t in TASKS}
    all_lens_set: set[int] = set()

    for d in result_dirs:
        model_key = os.path.basename(d)
        label = MODEL_LABELS.get(model_key, model_key)
        has_data = False
        for task in TASKS:
            rows = load_model_task(d, task)
            if not rows:
                continue
            has_data = True
            if label not in buckets[task]:
                buckets[task][label] = {}
            for r in rows:
                sl = r["seq_len"]
                all_lens_set.add(sl)
                buckets[task][label].setdefault(sl, []).append(r["correct"])
        if not has_data:
            print(f"  skip  {os.path.basename(d)}  (no craft/craft-long data)")

    if not all_lens_set:
        sys.exit("No data found.")

    all_lens = sorted(all_lens_set)
    all_models = sorted({m for t in TASKS for m in buckets[t]})
    print(f"Models: {all_models}")
    print(f"Seq lens: {all_lens}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

    for ax, task in zip(axes, TASKS):
        for i, model in enumerate(all_models):
            model_data = buckets[task].get(model, {})
            xs, ys = [], []
            for sl in all_lens:
                bucket = model_data.get(sl, [])
                if bucket:
                    xs.append(sl)
                    ys.append(sum(bucket) / len(bucket) * 100)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                label=model,
                color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                marker=MARKERS[i % len(MARKERS)],
                linewidth=1.8,
                markersize=5,
            )

        if task in ("corrupt_detect", "swap_detect"):
            ax.axhline(
                50,
                color=CHANCE_COLOR,
                linestyle="--",
                linewidth=1,
                label="random chance",
            )
        else:
            cxs = all_lens
            if task == "corrupt_localize":
                cys = [100.0 / l for l in cxs]
            else:
                cys = [100.0 / (l - 1) if l > 1 else 0.0 for l in cxs]
            ax.plot(
                cxs,
                cys,
                color=CHANCE_COLOR,
                linestyle="--",
                linewidth=1,
                label="random chance",
            )

        ax.set_title(TASK_LABELS[task], fontsize=13, fontweight="bold")
        ax.set_xlabel("Sequence length", fontsize=13)
        ax.set_xticks(all_lens)
        ax.set_xticklabels([str(l) for l in all_lens], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy (%)", fontsize=13)

    # Shared legend below the figure (exclude per-task chance duplicate labels)
    handles, labels = axes[0].get_legend_handles_labels()
    ncols = min(len(all_models) + 1, 6)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncols,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    out = os.path.join(BASE, "figures", "craft_combined_by_length.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out}")


DEFAULT_MODELS = {
    "gemma-4-e4b",
    "intern-vl-3",
    "intern-vl-3-5",
    "qwen3-vl-8b",
    "qwen2-5-vl-7b",
}


def _auto_discover() -> list[str]:
    results_dir = os.path.join(BASE, "results")
    return sorted(
        os.path.join(results_dir, e)
        for e in os.listdir(results_dir)
        if e in DEFAULT_MODELS and os.path.isdir(os.path.join(results_dir, e))
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirs = [a if os.path.isabs(a) else os.path.join(BASE, a) for a in sys.argv[1:]]
    else:
        dirs = _auto_discover()
    main(dirs)
