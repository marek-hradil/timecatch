import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
DATASET_LABELS = ["CLEVRER", "CRAFT", "Drive-LM", "MTL-AQA"]


def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["scenario"]] = row
    return rows


def accuracy(rows, scenarios=None):
    ok = tot = 0
    for scen, row in rows.items():
        if scenarios is not None and scen not in scenarios:
            continue
        gt = row["ground_truth"].strip().lower() == "true"
        ans = row["answer"].strip().lower()
        if ans in ("yes", "true", "1"):
            pred = True
        elif ans in ("no", "false", "0"):
            pred = False
        else:
            continue
        if pred == gt:
            ok += 1
        tot += 1
    return ok / tot if tot else 0.0


def load_baseline_scenarios(ds):
    path = os.path.join(BASE, "results", "prompt-ablation",
                        f"swap_detect_{ds}_qwen3-vl-8b_prompt-a.csv")
    rows = load_csv(path)
    return set(rows.keys()), rows


PROMPT_PATHS = {
    name: {
        ds: os.path.join(BASE, "results", "prompt-ablation",
                         f"swap_detect_{ds}_qwen3-vl-8b_prompt-{ch}.csv")
        for ds in DATASETS
    }
    for name, ch in [("Prompt B", "b"), ("Prompt C", "c"),
                     ("Prompt D", "d"), ("Prompt E", "e")]
}

SIGNAL_CONDITIONS = {
    "No scene desc": {
        "color": "#e07b39", "marker": "D", "size": 80, "zorder": 5,
        "paths": {ds: os.path.join(BASE, "results", "qwen3-vl-8b", "no-scene-desc",
                                   f"swap_detect_{ds}.csv") for ds in DATASETS},
    },
    "Thinking": {
        "color": "#3a9e64", "marker": "s", "size": 80, "zorder": 5,
        "paths": {ds: os.path.join(BASE, "results", "thinking-ablation",
                                   f"swap_detect_{ds}_qwen3-vl-8b-thinking_thinking.csv")
                  for ds in DATASETS},
    },
}

# Compute deltas relative to Prompt A (same subset scenarios)
prompt_deltas = {name: [] for name in PROMPT_PATHS}
signal_deltas = {name: [] for name in SIGNAL_CONDITIONS}

for ds in DATASETS:
    scenarios, baseline_rows = load_baseline_scenarios(ds)
    base_acc = accuracy(baseline_rows, scenarios)
    for name, paths in PROMPT_PATHS.items():
        rows = load_csv(paths[ds])
        prompt_deltas[name].append((accuracy(rows, scenarios) - base_acc) * 100)
    for name, cfg in SIGNAL_CONDITIONS.items():
        rows = load_csv(cfg["paths"][ds])
        signal_deltas[name].append((accuracy(rows, scenarios) - base_acc) * 100)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))

x = np.arange(len(DATASETS))
jitter = np.linspace(-0.15, 0.15, len(PROMPT_PATHS))

# Prompt dots — small, grey, semi-transparent
for j, (name, vals) in enumerate(prompt_deltas.items()):
    ax.scatter(x + jitter[j], vals, color="#999999", s=28, alpha=0.6,
               zorder=3, clip_on=False,
               label="Prompt variant" if j == 0 else "_nolegend_")

# Signal dots — large, coloured, on top
for name, cfg in SIGNAL_CONDITIONS.items():
    ax.scatter(x, signal_deltas[name], color=cfg["color"], marker=cfg["marker"],
               s=cfg["size"], zorder=cfg["zorder"], label=name, clip_on=False)

ax.axhline(0, color="black", linewidth=0.8, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(DATASET_LABELS, fontsize=12)
ax.set_ylabel("Δ Accuracy (pp vs. Prompt A)", fontsize=13)
ax.tick_params(axis="y", labelsize=12)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=12, frameon=False, loc="lower left")

plt.tight_layout()
out = os.path.join(BASE, "figures", "ablation_deltas.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, bbox_inches="tight")
out_png = out.replace(".pdf", ".png")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved {out}")
print(f"Saved {out_png}")

# Print numbers for reference
print("\nΔ accuracies (pp vs Prompt A):")
print(f"{'':20s}", "  ".join(f"{d:>10s}" for d in DATASET_LABELS))
for name, vals in {**prompt_deltas, **signal_deltas}.items():
    row = "  ".join(f"{v:>+10.1f}" for v in vals)
    print(f"{name:20s}  {row}")
