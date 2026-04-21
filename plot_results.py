import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))

# Filter error rows (both predictions 0)
valid = [
    r for r in rows
    if not (r["model_prediction_first"] == "0" and r["model_prediction_second"] == "0")
]

correct_by_mode_len: dict[str, dict[int, list[bool]]] = {
    "series": defaultdict(list),
    "composite": defaultdict(list),
}

for r in valid:
    mode = r.get("input_mode", "series")
    if mode not in correct_by_mode_len:
        continue
    length = int(r["prediction_series_length"])
    match = (
        r["model_prediction_first"] == r["ground_truth_first"]
        and r["model_prediction_second"] == r["ground_truth_second"]
    )
    correct_by_mode_len[mode][length].append(match)

all_lengths = sorted(
    set(correct_by_mode_len["series"]) | set(correct_by_mode_len["composite"])
)

def acc(mode, l):
    bucket = correct_by_mode_len[mode].get(l, [])
    return sum(bucket) / len(bucket) * 100 if bucket else 0

def count(mode, l):
    return len(correct_by_mode_len[mode].get(l, []))

series_acc    = [acc("series", l)    for l in all_lengths]
composite_acc = [acc("composite", l) for l in all_lengths]
random_acc    = [1 / (l - 1) * 100  for l in all_lengths]
series_counts    = [count("series", l)    for l in all_lengths]
composite_counts = [count("composite", l) for l in all_lengths]

x = range(len(all_lengths))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

bars_s = ax.bar([i - width for i in x], series_acc,    width, label="Series",    color="steelblue",  edgecolor="white")
bars_c = ax.bar([i          for i in x], composite_acc, width, label="Composite", color="mediumseagreen", edgecolor="white")
bars_r = ax.bar([i + width  for i in x], random_acc,    width, label="Random",    color="salmon",     edgecolor="white")

for bar, n in zip(bars_s, series_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"n={n}", ha="center", va="bottom", fontsize=8)
for bar, n in zip(bars_c, composite_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"n={n}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels([str(l) for l in all_lengths])
ax.set_xlabel("Sequence length")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Swap detection accuracy by sequence length and input mode")
ax.set_ylim(0, 115)
ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8)
ax.legend()

plt.tight_layout()
plt.show()
