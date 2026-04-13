import csv
import matplotlib.pyplot as plt
from collections import defaultdict

with open("results.csv", newline="") as f:
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

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([str(l) for l in lengths], accuracies, color="steelblue", edgecolor="white")

for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"n={count}",
        ha="center", va="bottom", fontsize=9,
    )

ax.set_xlabel("Sequence length")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Swap detection accuracy by sequence length")
ax.set_ylim(0, 110)
ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.show()
