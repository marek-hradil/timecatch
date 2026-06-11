"""Shared color definitions for all plotting scripts."""

# Stable per-dataset colors (seaborn "deep" palette)
DATASET = {
    "clevrer":    "#4c72b0",
    "craft":      "#dd8452",
    "craft-long": "#dd8452",
    "drive-lm":   "#55a868",
    "drivelm":    "#55a868",
    "mtl-aqa":    "#c44e52",
}

# Chance / random-baseline line
CHANCE_COLOR = "#999999"

# Ablation pair: positive/correct/with  vs  negative/incorrect/without
ABLATION_A = "steelblue"
ABLATION_B = "salmon"

# Input-mode comparison (plot_results.py)
MODE_A = "steelblue"       # Series
MODE_B = "mediumseagreen"  # Composite

# Fallback ordered palette for dynamic model lists
MODEL_PALETTE = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52",
    "#8172b2", "#937860", "#da8bc3", "#8c8c8c",
    "#ccb974", "#64b5cd",
]
