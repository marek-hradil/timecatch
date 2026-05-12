"""
Restructure DriveLM into the unified dataset format.

Reads:  datasets/drive_lm/metadata.json
Writes: datasets/drive_lm/scenes.json
        datasets/drive_lm/scenes/scene_<idx>/0.jpg, 1.jpg, ...

Frame files are copied from samples/ into per-scene subdirectories.
Originals in samples/ are left untouched.

Timestamps are extracted from filenames and used to sort frames
chronologically. Scenes where the metadata order does not match
chronological order are flagged with "ordering_flag": true.
"""

import json
import re
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "drive_lm"
SAMPLES_DIR = DATASET_DIR / "samples"
SCENES_DIR = DATASET_DIR / "scenes"

_TS_RE = re.compile(r"__(\d+)\.jpg$")


def extract_timestamp(path: str) -> int:
    m = _TS_RE.search(path)
    return int(m.group(1)) if m else 0


def main():
    metadata = json.load(open(DATASET_DIR / "metadata.json"))

    # Stable ordering: sort hex scene IDs so indices are deterministic
    scene_ids = sorted(metadata.keys())

    scenes = []
    for video_index, scene_id in enumerate(scene_ids):
        scene = metadata[scene_id]
        frames = [f for f in scene["frames"] if f is not None]

        # Sort by embedded timestamp
        frames_sorted = sorted(frames, key=extract_timestamp)
        ordering_ok = frames_sorted == frames

        # Copy into scenes/scene_<idx>/
        scene_dir = SCENES_DIR / f"scene_{video_index}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        scene_paths = []
        for i, src_rel in enumerate(frames_sorted):
            src = DATASET_DIR / src_rel
            dst = scene_dir / f"{i}.jpg"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            scene_paths.append(str(dst.relative_to(DATASET_DIR)))

        entry = {
            "video_index": video_index,
            "scene_paths": scene_paths,
            "scene_description": scene["scene_description"],
        }
        if not ordering_ok:
            entry["ordering_flag"] = True

        scenes.append(entry)

    out_path = DATASET_DIR / "scenes.json"
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)

    flagged = sum(1 for s in scenes if s.get("ordering_flag"))
    print(f"Written {len(scenes)} scenes to {out_path}")
    if flagged:
        print(f"  {flagged} scenes flagged with ordering issues")


if __name__ == "__main__":
    main()
