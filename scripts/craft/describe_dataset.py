import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from describe_utils import (
    load_model, embed, load_scenes, consecutive_pairs,
    compute_scores, print_overview, print_frame_histogram,
    print_histogram, print_pairs_in_range,
)

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "CRAFT"
BG_PATH = DATASET_DIR / "background.jpg"


def main():
    model, device = load_model()

    bg_emb = embed(str(BG_PATH), model, device)

    scenes = load_scenes(str(DATASET_DIR))
    print_overview(scenes, "CRAFT")
    print_frame_histogram(scenes)

    pairs = consecutive_pairs(scenes, str(DATASET_DIR))

    print("\n--- Raw DINOv2 ---")
    raw_scores = compute_scores(pairs, model, device)
    print_histogram(raw_scores, "DINOv2 cosine similarity")

    print("\n--- Background-subtracted DINOv2 ---")
    fg_scores = compute_scores(pairs, model, device, bg_emb=bg_emb)
    print_histogram(fg_scores, "DINOv2 foreground similarity")
    print_pairs_in_range(pairs, fg_scores, lo=0.98, hi=1.0)


if __name__ == "__main__":
    main()
