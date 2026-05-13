"""
Filter a dataset's scenes.json by LPIPS quality.

Two-pass filter:
  1. Greedy frame removal: scan each scene left to right; whenever a consecutive
     pair has LPIPS < MIN_LPIPS, drop the right frame and continue from the left.
  2. Scene removal: drop any scene with fewer than MIN_FRAMES frames left.

Writes datasets/<DATASET>/scenes_filtered.json — the original scenes.json is untouched.

Usage:
    python3 scripts/filter_scenes.py [clevrer | craft | drivelm | mtl-aqa]
"""

import json
import os
import sys

import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

DATASETS = {
    "clevrer":  "CLEVRER",
    "craft":    "CRAFT",
    "drivelm":  "drive_lm",
    "mtl-aqa":  "MTL-AQA",
}

MIN_LPIPS = 0.05
MIN_FRAMES = 4

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def dataset_dir(name: str) -> str:
    return os.path.join(ROOT, "datasets", DATASETS[name])


def load_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0)


def lpips_pair(loss_fn, abs_a: str, abs_b: str) -> float:
    with torch.no_grad():
        return loss_fn(load_tensor(abs_a), load_tensor(abs_b)).item()


def filter_scene_frames(paths: list[str], dset_dir: str, loss_fn) -> list[str]:
    """Greedy left-to-right: drop the right frame of any pair below MIN_LPIPS."""
    kept = list(paths)
    i = 0
    while i < len(kept) - 1:
        abs_a = os.path.join(dset_dir, kept[i])
        abs_b = os.path.join(dset_dir, kept[i + 1])
        if lpips_pair(loss_fn, abs_a, abs_b) < MIN_LPIPS:
            kept.pop(i + 1)
        else:
            i += 1
    return kept


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        print(f"Usage: filter_scenes.py [{' | '.join(DATASETS)}]")
        sys.exit(1)

    name = sys.argv[1]
    dset_dir = dataset_dir(name)
    input_json = os.path.join(dset_dir, "scenes.json")
    output_json = os.path.join(dset_dir, "scenes_filtered.json")

    with open(input_json) as f:
        scenes = json.load(f)

    loss_fn = lpips.LPIPS(net="alex", verbose=False)

    frames_before = sum(len(s["scene_paths"]) for s in scenes)
    filtered_scenes = []
    scenes_dropped = 0
    total_frames_dropped = 0

    for scene in tqdm(scenes, desc=f"Filtering {name}"):
        original_paths = scene["scene_paths"]
        kept_paths = filter_scene_frames(original_paths, dset_dir, loss_fn)

        total_frames_dropped += len(original_paths) - len(kept_paths)

        if len(kept_paths) < MIN_FRAMES:
            scenes_dropped += 1
            continue

        filtered_scenes.append({**scene, "scene_paths": kept_paths})

    frames_after = sum(len(s["scene_paths"]) for s in filtered_scenes)

    print(f"\nScenes:  {len(scenes)} -> {len(filtered_scenes)} ({scenes_dropped} dropped, <{MIN_FRAMES} frames)")
    print(f"Frames:  {frames_before} -> {frames_after} ({total_frames_dropped} removed by LPIPS filter)")
    print(f"Pairs:   {frames_before - len(scenes)} -> {frames_after - len(filtered_scenes)}")

    with open(output_json, "w") as f:
        json.dump(filtered_scenes, f, indent=2)

    print(f"\nWritten to {output_json}")


if __name__ == "__main__":
    main()
