"""
Restructure the FULL CRAFT dataset (train+validation+test, 9,917 videos)
into the unified dataset format, alongside (not overwriting) the original
1,983-video test-split-only scenes.json used for the paper's main results.

Reads:  datasets/CRAFT/scenes_full_raw.json (written by prepare_frames_full.py)
Writes: datasets/CRAFT/scenes_full.json

Collapses multiple QA entries per video into one scene entry. Mirrors
restructure_data.py exactly otherwise.
"""

import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "CRAFT"


def main():
    src = json.load(open(DATASET_DIR / "scenes_full_raw.json"))

    seen: set[int] = set()
    scenes = []
    for e in src:
        vid = e["video_index"]
        if vid in seen:
            continue
        seen.add(vid)
        scenes.append({
            "video_index": vid,
            "scene_paths": e["frame_paths"],
            "scene_description": e["scene_text"],
        })

    out_path = DATASET_DIR / "scenes_full.json"
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"Written {len(scenes)} scenes to {out_path}")


if __name__ == "__main__":
    main()
