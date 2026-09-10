"""Print a LaTeX table of dataset statistics for TimeCatch.

Usage:
  python scripts/print_dataset_stats_table.py
"""
import json
import os
import statistics

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DATASETS = [
    ("CLEVRER",    "datasets/CLEVRER/scenes_filtered.json",  "Synthetic",   None),
    ("CRAFT",      "datasets/CRAFT/scenes_filtered.json",    "Synthetic",   lambda s: len(s["scene_paths"]) <= 8),
    ("DriveLM",    "datasets/drive_lm/scenes_filtered.json", "Real-world",  None),
    ("MTL-AQA",    "datasets/MTL-AQA/scenes_filtered.json",  "Real-world",  None),
    ("CRAFT-Long", "datasets/CRAFT/scenes_filtered.json",    "Synthetic",   lambda s: len(s["scene_paths"]) > 8),
]


def load(path, predicate):
    with open(os.path.join(BASE, path)) as f:
        scenes = json.load(f)
    if predicate:
        scenes = [s for s in scenes if predicate(s)]
    return scenes


rows = []
for name, path, domain, pred in DATASETS:
    scenes = load(path, pred)
    lengths = [len(s["scene_paths"]) for s in scenes]
    rows.append((name, domain, len(scenes), statistics.mean(lengths)))

# LaTeX output
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\small")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"Dataset & Domain & Sequences & Avg.\ Length \\")
print(r"\midrule")
for i, (name, domain, n_seq, avg_len) in enumerate(rows):
    if i == 2:   # blank line before real-world datasets
        print()
    if i == 4:   # blank line before CRAFT-Long
        print()
    print(rf"{name} & {domain} & {n_seq:,} & {avg_len:.1f} \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Statistics of datasets included in TimeCatch.}")
print(r"\label{tab:dataset_stats}")
print(r"\end{table}")
