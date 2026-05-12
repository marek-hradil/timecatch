import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

DATASETS = {
    "clevrer":  "CLEVRER",
    "craft":    "CRAFT",
    "drivelm":  "drive_lm",
    "mtl-aqa":  "MTL-AQA",
}

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def dataset_dir(name: str) -> str:
    return os.path.join(ROOT, "datasets", DATASETS[name])


def load_scenes(dataset_dir: str) -> list:
    with open(os.path.join(dataset_dir, "scenes.json")) as f:
        return json.load(f)


def print_overview(scenes: list, name: str):
    total_images = sum(len(s["scene_paths"]) for s in scenes)
    print(f"Dataset: {name}")
    print(f"Total scenes: {len(scenes)}")
    print(f"Total images: {total_images}")


def print_frame_histogram(scenes: list):
    frame_counts = Counter(len(s["scene_paths"]) for s in scenes)
    print("\nFrames per scene:")
    for n in sorted(frame_counts):
        bar = "#" * (frame_counts[n] // 10)
        print(f"  {n}: {bar} ({frame_counts[n]})")


def consecutive_pairs(scenes: list, dset_dir: str) -> list[tuple[str, str, str]]:
    pairs = []
    for scene in scenes:
        scene_id = str(scene["video_index"])
        paths = scene["scene_paths"]
        for a, b in zip(paths, paths[1:]):
            path_a = os.path.join(dset_dir, a)
            path_b = os.path.join(dset_dir, b)
            if os.path.exists(path_a) and os.path.exists(path_b):
                pairs.append((path_a, path_b, scene_id))
    return pairs


def similarity_pair(path_a: str, path_b: str) -> float:
    a = np.array(Image.open(path_a).convert("L"))
    b = np.array(Image.open(path_b).convert("L"))
    return structural_similarity(a, b, data_range=255)


def compute_similarity_scores(pairs: list[tuple[str, str, str]]) -> list[float]:
    return [similarity_pair(a, b) for a, b, _ in tqdm(pairs, desc="Computing SSIM")]


def print_similarity_histogram(scores: list[float], bins: int = 20):
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    print(f"\nSSIM of consecutive frames ({len(scores)} pairs):")
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * (count // 10)
        print(f"  {lo:.2f}-{hi:.2f}: {bar} ({count})")
    print(f"\n  mean={np.mean(scores):.3f}  median={np.median(scores):.3f}  min={np.min(scores):.3f}  max={np.max(scores):.3f}")


def print_pairs_in_range(pairs: list[tuple[str, str, str]], scores: list[float], lo: float, hi: float):
    matches = [(s, a, b, sid) for (a, b, sid), s in zip(pairs, scores) if lo <= s < hi]
    print(f"\nPairs with similarity {lo:.2f}-{hi:.2f} ({len(matches)}):")
    for score, a, b, scene_id in sorted(matches, reverse=True):
        print(f"  {score:.3f}  {scene_id}  {os.path.basename(a)}  ->  {os.path.basename(b)}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        print(f"Usage: describe_dataset.py [{' | '.join(DATASETS)}]")
        sys.exit(1)

    name = sys.argv[1]
    dset_dir = dataset_dir(name)
    scenes = load_scenes(dset_dir)

    print_overview(scenes, name)
    print_frame_histogram(scenes)

    pairs = consecutive_pairs(scenes, dset_dir)
    if not pairs:
        print("\nNo image pairs found.")
        return

    scores = compute_similarity_scores(pairs)
    print_similarity_histogram(scores)
    print_pairs_in_range(pairs, scores, lo=0.98, hi=1.0)


if __name__ == "__main__":
    main()
