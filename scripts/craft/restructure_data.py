"""
Restructure CRAFT into the unified dataset format.

Reads:  datasets/CRAFT/scenes.json
Writes: datasets/CRAFT/scenes.json  (overwritten with simplified schema)

Collapses multiple QA entries per video into one scene entry.
Only the first encountered frame_paths and scene_text per video_index are kept.
"""

import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "CRAFT"


def main():
    src = json.load(open(DATASET_DIR / "scenes.json"))

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

    out_path = DATASET_DIR / "scenes.json"
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"Written {len(scenes)} scenes to {out_path}")


if __name__ == "__main__":
    main()
