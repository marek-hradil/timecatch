"""
Restructure MTL-AQA into the unified dataset format.

Reads:  datasets/MTL-AQA/{VV}_{CC}/*.jpg  (prepared by prepare_frames.py)
        datasets/MTL-AQA/test_split.pkl
        datasets/MTL-AQA/annotations.pkl
Writes: datasets/MTL-AQA/scenes/scene_{VV}_{CC}/0.jpg, 1.jpg, ...
        datasets/MTL-AQA/scenes.json
"""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np  # noqa: F401 — required for annotations.pkl unpickling

MTL_ROOT = Path(__file__).parent.parent.parent / "datasets" / "MTL-AQA"
SCENES_ROOT = MTL_ROOT / "scenes"

_ROTATION = {0: "forward", 1: "backward", 2: "reverse", 3: "inward"}
_POSITION = {0: "tuck", 1: "pike", 2: "straight"}


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_scene_description(ann: dict) -> str:
    rotation = _ROTATION[ann["rotation_type"]]
    position = _POSITION[ann["position"]]
    ss = ann["ss_no"] / 2
    tw = ann["tw_no"] / 2

    if ann["armstand"]:
        prefix = "An armstand"
    elif rotation[0] in "aeiou":
        prefix = "An"
    else:
        prefix = "A"

    movement = f"{rotation} dive"
    if ss > 0:
        movement += f" with {ss:g} somersault{'s' if ss != 1 else ''}"
    if tw > 0:
        movement += f" and {tw:g} twist{'s' if tw != 1 else ''}"
    movement += f" in {position} position"

    return f"{prefix} {movement}."


def copy_frames(video_id: int, clip_id: int) -> list[str]:
    src_dir = MTL_ROOT / f"{video_id:02d}_{clip_id:02d}"
    scene_id = f"{video_id:02d}_{clip_id:02d}"
    out_dir = SCENES_ROOT / f"scene_{scene_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(src_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    paths = []
    for idx, src in enumerate(frames):
        dst = out_dir / f"{idx}.jpg"
        shutil.copy2(src, dst)
        paths.append(str(dst.relative_to(MTL_ROOT)))

    return paths


def main():
    test_split = load_pickle(MTL_ROOT / "test_split.pkl")
    annotations = load_pickle(MTL_ROOT / "annotations.pkl")

    total = len(test_split)
    scenes = []

    for idx, (video_id, clip_id) in enumerate(test_split, 1):
        ann = annotations[(video_id, clip_id)]
        scene_id = f"{video_id:02d}_{clip_id:02d}"

        scene_paths = copy_frames(video_id, clip_id)
        scene_description = build_scene_description(ann)

        scenes.append({
            "video_index": scene_id,
            "scene_paths": scene_paths,
            "scene_description": scene_description,
        })

        print(f"  [{idx}/{total}] {scene_id}  {len(scene_paths)} frames  {scene_description}")

    out_path = MTL_ROOT / "scenes.json"
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"\nWritten {len(scenes)} scenes to {out_path}")


if __name__ == "__main__":
    main()
