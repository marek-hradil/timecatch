"""
Restructure CLEVRER into the unified dataset format.

Reads:  datasets/CLEVRER/dataset.json
Writes: datasets/CLEVRER/scenes.json

Frame files are already in scenes/scene_<id>/0.jpg — no copying needed.
Only field names are normalised.
"""

import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "CLEVRER"


def main():
    src = json.load(open(DATASET_DIR / "dataset.json"))

    scenes = [
        {
            "video_index": e["video_index"],
            "scene_paths": e["frame_paths"],
            "scene_description": e["scene_text"],
        }
        for e in src
    ]

    out_path = DATASET_DIR / "scenes.json"
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"Written {len(scenes)} scenes to {out_path}")


if __name__ == "__main__":
    main()
