"""
Clean up MTL-AQA frames in-place:
  1. Delete clip folders not in the test split.
  2. Delete frames outside [start_frame, end_frame] within each test clip.
  3. Subsample to 1 frame per 1 s (every VIDEO_FPS * 1.0 frames by index).
"""

import pickle
import shutil
from pathlib import Path

MTL_ROOT = Path(__file__).parent.parent.parent / "datasets" / "MTL-AQA"
VIDEO_FPS = 25.0
SAMPLE_INTERVAL = round(VIDEO_FPS * 1.0)  # 1 frame per 1 s


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    test_split = load_pickle(MTL_ROOT / "test_split.pkl")
    annotations = load_pickle(MTL_ROOT / "annotations.pkl")

    test_keys = {(v, c) for v, c in test_split}

    all_clip_dirs = [
        d for d in MTL_ROOT.iterdir()
        if d.is_dir() and "_" in d.name and d.name[0].isdigit()
    ]

    # 1. Remove non-test clip folders
    removed_dirs = 0
    for d in all_clip_dirs:
        parts = d.name.split("_")
        key = (int(parts[0]), int(parts[1]))
        if key not in test_keys:
            shutil.rmtree(d)
            removed_dirs += 1
    print(f"Removed {removed_dirs} non-test clip folders.")

    # 2. Trim frames outside [start_frame, end_frame] in each test clip
    removed_frames = 0
    for video_id, clip_id in sorted(test_keys):
        ann = annotations[(video_id, clip_id)]
        start, end = ann["start_frame"], ann["end_frame"]
        clip_dir = MTL_ROOT / f"{video_id:02d}_{clip_id:02d}"
        for frame in clip_dir.glob("*.jpg"):
            n = int(frame.stem)
            if n < start or n > end:
                frame.unlink()
                removed_frames += 1

    print(f"Removed {removed_frames} out-of-range frames across {len(test_keys)} test clips.")

    # 3. Subsample: keep every SAMPLE_INTERVAL-th frame by sorted index
    removed_sampled = 0
    for video_id, clip_id in sorted(test_keys):
        clip_dir = MTL_ROOT / f"{video_id:02d}_{clip_id:02d}"
        frames = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        for idx, frame in enumerate(frames):
            if idx % SAMPLE_INTERVAL != 0:
                frame.unlink()
                removed_sampled += 1

    print(f"Removed {removed_sampled} frames during subsampling (kept every {SAMPLE_INTERVAL}th).")


if __name__ == "__main__":
    main()
