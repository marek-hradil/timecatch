"""
Prepares the DriveLM dataset for front-camera-only temporal analysis.

Transforms datasets/drive_lm in-place:
  - Flattens samples/CAM_FRONT/* -> samples/*
  - Removes all other camera subdirectories
  - Rewrites metadata.json to keep only scene_description and an ordered
    list of CAM_FRONT frame paths
"""

import json
import os
import shutil

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "drive_lm")
DATASET_DIR = os.path.normpath(DATASET_DIR)
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.json")
SAMPLES_DIR = os.path.join(DATASET_DIR, "samples")
CAM_FRONT_DIR = os.path.join(SAMPLES_DIR, "CAM_FRONT")


def main():
    with open(METADATA_PATH) as f:
        data = json.load(f)

    # Move CAM_FRONT images up to samples/
    moved = 0
    missing = []
    for filename in os.listdir(CAM_FRONT_DIR):
        src = os.path.join(CAM_FRONT_DIR, filename)
        dst = os.path.join(SAMPLES_DIR, filename)
        shutil.move(src, dst)
        moved += 1
    print(f"Moved {moved} images to {SAMPLES_DIR}")

    # Remove all camera subdirectories
    for entry in os.listdir(SAMPLES_DIR):
        entry_path = os.path.join(SAMPLES_DIR, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            print(f"Removed directory: {entry_path}")

    # Build new metadata
    new_data = {}
    for scene_id, scene in data.items():
        frames = []
        for frame in scene["key_frames"].values():
            cam_front_path = frame["image_paths"].get("CAM_FRONT", "")
            filename = os.path.basename(cam_front_path)
            local_path = os.path.join(SAMPLES_DIR, filename)
            if not os.path.exists(local_path):
                missing.append(filename)
                frames.append(None)
            else:
                frames.append(os.path.join("samples", filename))

        new_data[scene_id] = {
            "scene_description": scene["scene_description"],
            "frames": frames,
        }

    with open(METADATA_PATH, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"Wrote new metadata.json ({len(new_data)} scenes)")

    if missing:
        print(f"WARNING: {len(missing)} frames had no local image: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    else:
        print("All frame images resolved successfully.")


if __name__ == "__main__":
    main()
