"""
DriveLM has heterogeneous real-world scenes with a moving camera, so a single
background frame is not meaningful. Raw DINOv2 similarity is used instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from describe_utils import (
    load_model, load_scenes, consecutive_pairs,
    compute_scores, print_overview, print_frame_histogram,
    print_histogram, print_pairs_in_range,
)

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "drive_lm"


def main():
    model, device = load_model()

    scenes = load_scenes(str(DATASET_DIR))
    print_overview(scenes, "DriveLM")
    print_frame_histogram(scenes)

    pairs = consecutive_pairs(scenes, str(DATASET_DIR))

    scores = compute_scores(pairs, model, device)
    print_histogram(scores, "DINOv2 cosine similarity")
    print_pairs_in_range(pairs, scores, lo=0.98, hi=1.0)


if __name__ == "__main__":
    main()
