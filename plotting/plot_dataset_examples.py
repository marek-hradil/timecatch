import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "datasets")

DATASETS = [
    ("CLEVRER", "clevrer"),
    ("CRAFT",   "craft"),
    ("DriveLM", "drive_lm"),
    ("MTL-AQA", "mtl-aqa"),
]

N_FRAMES = 4

def load_frames(dataset_name):
    manifest_path = os.path.join(BASE, dataset_name, "scenes_filtered.json")
    with open(manifest_path) as f:
        scenes = json.load(f)
    seq = next((s for s in scenes if len(s["scene_paths"]) >= N_FRAMES), scenes[0])
    return seq["scene_paths"][:N_FRAMES], os.path.join(BASE, dataset_name)

def read_image(path):
    return np.array(Image.open(path))

def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    target_w = int(round(w * target_h / h))
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.LANCZOS))

# Layout parameters (all in inches)
FIG_W       = 16.0
FRAME_GAP   = 0.05   # between frames within a dataset
DATASET_GAP = 0.35   # between the two datasets in one row
ROW_GAP     = 0.45   # between the two row groups
TITLE_H     = 0.30   # space for dataset name above each row group
T_MARGIN    = 0.10   # top margin
B_MARGIN    = 0.10   # bottom margin
L_MARGIN    = 0.02 * FIG_W
R_MARGIN    = 0.02 * FIG_W

GROUPS = [(DATASETS[0], DATASETS[1]), (DATASETS[2], DATASETS[3])]

# For each row, compute the frame height that fills the content width
def compute_frame_height(ar_a, ar_b):
    usable = FIG_W - L_MARGIN - R_MARGIN - DATASET_GAP - 2 * (N_FRAMES - 1) * FRAME_GAP
    return usable / (N_FRAMES * (ar_a + ar_b))

# Load one image per dataset to get native AR
def native_ar(dataset_name):
    paths, root = load_frames(dataset_name)
    img = read_image(os.path.join(root, paths[0]))
    return img.shape[1] / img.shape[0]

group_ars = [(native_ar(ds_a), native_ar(ds_b)) for (_, ds_a), (_, ds_b) in GROUPS]
row_heights = [compute_frame_height(ar_a, ar_b) for ar_a, ar_b in group_ars]

fig_h = T_MARGIN + TITLE_H + row_heights[0] + ROW_GAP + TITLE_H + row_heights[1] + B_MARGIN
fig = plt.figure(figsize=(FIG_W, fig_h))

# Build rows from top to bottom
y_cursor = fig_h - T_MARGIN  # current y in inches, counting down

for group_idx, ((label_a, ds_a), (label_b, ds_b)) in enumerate(GROUPS):
    ar_a, ar_b = group_ars[group_idx]
    H = row_heights[group_idx]

    frame_bottom = y_cursor - TITLE_H - H  # bottom edge of frames in inches

    x_cursor = L_MARGIN

    for group_col, (label, ds_name, ar) in enumerate(
        [(label_a, ds_a, ar_a), (label_b, ds_b, ar_b)]
    ):
        frame_paths, dataset_root = load_frames(ds_name)
        dataset_x_start = x_cursor

        for frame_idx, rel_path in enumerate(frame_paths):
            frame_w = H * ar  # width of this frame in inches

            ax = fig.add_axes([
                x_cursor / FIG_W,
                frame_bottom / fig_h,
                frame_w / FIG_W,
                H / fig_h,
            ])
            img = read_image(os.path.join(dataset_root, rel_path))
            ax.imshow(img, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            x_cursor += frame_w + FRAME_GAP

        # Dataset title centred above its frames
        dataset_center_x = (dataset_x_start + x_cursor - FRAME_GAP) / 2
        title_y = (frame_bottom + H + 0.05) / fig_h
        fig.text(dataset_center_x / FIG_W, title_y, label,
                 ha='center', va='bottom', fontsize=13, fontweight='bold')

        if group_col == 0:
            x_cursor += DATASET_GAP - FRAME_GAP  # replace last frame gap with dataset gap

    y_cursor = frame_bottom - ROW_GAP

out_path = os.path.join(os.path.dirname(__file__), "..", "figures", "dataset_examples.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
