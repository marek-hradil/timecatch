import json
import os
from collections import Counter

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

DATASET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "datasets", "drive_lm"))
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.json")


def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


def print_overview(data: dict):
    total_images = sum(len(scene["frames"]) for scene in data.values())
    print(f"Total scenes: {len(data)}")
    print(f"Total images: {total_images}")


def print_frame_histogram(data: dict):
    frame_counts = Counter(len(scene["frames"]) for scene in data.values())
    print("\nFrames per scene:")
    for n in sorted(frame_counts):
        bar = "#" * (frame_counts[n] // 10)
        print(f"  {n}: {bar} ({frame_counts[n]})")


def consecutive_pairs(data: dict) -> list[tuple[str, str, str]]:
    """Returns (path_a, path_b, scene_id) for every consecutive frame pair."""
    pairs = []
    for scene_id, scene in data.items():
        frames = scene["frames"]
        for a, b in zip(frames, frames[1:]):
            path_a = os.path.join(DATASET_DIR, a)
            path_b = os.path.join(DATASET_DIR, b)
            if os.path.exists(path_a) and os.path.exists(path_b):
                pairs.append((path_a, path_b, scene_id))
    return pairs


def ssim_pair(path_a: str, path_b: str) -> float:
    a = np.array(Image.open(path_a).convert("L"))
    b = np.array(Image.open(path_b).convert("L"))
    return structural_similarity(a, b, data_range=255)


def compute_ssim_scores(pairs: list[tuple[str, str, str]]) -> list[float]:
    return [ssim_pair(a, b) for a, b, _ in tqdm(pairs, desc="Computing SSIM")]


def print_ssim_histogram(scores: list[float], bins: int = 10):
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    print(f"\nSSIM of consecutive frames ({len(scores)} pairs):")
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * (count // 10)
        print(f"  {lo:.1f}-{hi:.1f}: {bar} ({count})")
    print(f"\n  mean={np.mean(scores):.3f}  median={np.median(scores):.3f}  min={np.min(scores):.3f}  max={np.max(scores):.3f}")


def print_pairs_in_range(pairs: list[tuple[str, str, str]], scores: list[float], lo: float, hi: float):
    matches = [(s, a, b, sid) for (a, b, sid), s in zip(pairs, scores) if lo <= s < hi]
    print(f"\nPairs with SSIM {lo:.1f}-{hi:.1f} ({len(matches)}):")
    for score, a, b, scene_id in sorted(matches, reverse=True):
        print(f"  {score:.3f}  {scene_id}  {os.path.basename(a)}  ->  {os.path.basename(b)}")


def main():
    data = load_metadata()

    print_overview(data)
    print_frame_histogram(data)

    pairs = consecutive_pairs(data)
    if not pairs:
        print("\nNo image pairs found — check that samples/ images are present.")
        return

    scores = compute_ssim_scores(pairs)
    print_ssim_histogram(scores)
    print_pairs_in_range(pairs, scores, lo=0.0, hi=0.3)
    print_pairs_in_range(pairs, scores, lo=0.9, hi=1.0)


if __name__ == "__main__":
    main()
