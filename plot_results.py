import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
with open(path, newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["outcome"] != "INVALID"]

correct_by_mode_len: dict[str, dict[int, list[bool]]] = {
    "series": defaultdict(list),
    "composite": defaultdict(list),
}

for r in rows:
    mode = r["input_mode"]
    if mode not in correct_by_mode_len:
        continue
    length = int(r["sequence_length"])
    correct_by_mode_len[mode][length].append(r["outcome"] in ("TP", "TN"))

all_lengths = sorted(
    set(correct_by_mode_len["series"]) | set(correct_by_mode_len["composite"])
)


def acc(mode, l):
    bucket = correct_by_mode_len[mode].get(l, [])
    return sum(bucket) / len(bucket) * 100 if bucket else 0


def count(mode, l):
    return len(correct_by_mode_len[mode].get(l, []))


series_acc = [acc("series", l) for l in all_lengths]
composite_acc = [acc("composite", l) for l in all_lengths]
series_counts = [count("series", l) for l in all_lengths]
composite_counts = [count("composite", l) for l in all_lengths]

x = range(len(all_lengths))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

bars_s = ax.bar([i - width / 2 for i in x], series_acc, width, label="Series", color="steelblue", edgecolor="white")
bars_c = ax.bar([i + width / 2 for i in x], composite_acc, width, label="Composite", color="mediumseagreen", edgecolor="white")

for bar, n in zip(bars_s, series_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"n={n}", ha="center", va="bottom", fontsize=8)
for bar, n in zip(bars_c, composite_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"n={n}", ha="center", va="bottom", fontsize=8)

ax.axhline(y=50, color="salmon", linestyle="--", linewidth=1, label="chance (50%)")
ax.set_xticks(list(x))
ax.set_xticklabels([str(l) for l in all_lengths])
ax.set_xlabel("Sequence length")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Swap detection accuracy by sequence length and input mode")
ax.set_ylim(0, 115)
ax.legend()

plt.tight_layout()
plt.savefig("results_accuracy.png", dpi=100)
plt.show()
