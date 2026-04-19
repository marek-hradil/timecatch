import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))

# Filter out error rows (model_prediction_first == 0 and second == 0)
valid = [
    r for r in rows
    if not (r["model_prediction_first"] == "0" and r["model_prediction_second"] == "0")
]

correct_by_len: dict[int, list[bool]] = defaultdict(list)
for r in valid:
    length = int(r["prediction_series_length"])
    match = (
        r["model_prediction_first"] == r["ground_truth_first"]
        and r["model_prediction_second"] == r["ground_truth_second"]
    )
    correct_by_len[length].append(match)

lengths = sorted(correct_by_len)
accuracies = [sum(correct_by_len[l]) / len(correct_by_len[l]) * 100 for l in lengths]
counts = [len(correct_by_len[l]) for l in lengths]
# Random baseline: 1 correct pair out of (length - 1) consecutive pairs
random_accuracies = [1 / (l - 1) * 100 for l in lengths]

x = range(len(lengths))
width = 0.4

fig, ax = plt.subplots(figsize=(8, 5))
model_bars = ax.bar([i - width / 2 for i in x], accuracies, width, label="Model", color="steelblue", edgecolor="white")
random_bars = ax.bar([i + width / 2 for i in x], random_accuracies, width, label="Random", color="salmon", edgecolor="white")

for bar, count in zip(model_bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"n={count}",
        ha="center", va="bottom", fontsize=9,
    )

ax.set_xticks(list(x))
ax.set_xticklabels([str(l) for l in lengths])
ax.set_xlabel("Sequence length")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Swap detection accuracy by sequence length")
ax.set_ylim(0, 110)
ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8)
ax.legend()

plt.tight_layout()
plt.show()
