"""Clean up the freshly-extracted full MTL-AQA frame archive in place, for
ALL 1,412 clips (train_split_0.pkl + test_split.pkl combined), not just the
353-clip test split scripts/mtl-aqa/prepare_frames.py originally kept.

Mirrors prepare_frames.py's trim + subsample steps exactly, but:
  - operates on datasets/MTL-AQA_raw_full/ (a fresh extraction of the same
    MUSDL-hosted archive, kept separate from datasets/MTL-AQA/ so the
    original 353-clip processed data used for the paper's baseline is never
    touched)
  - does NOT delete any clip folders — the archive already contains exactly
    the 1,412 clips referenced by train_split.pkl + test_split.pkl, nothing
    extra to prune.

  1. Trim frames outside [start_frame, end_frame] within each clip.
  2. Subsample to 1 frame per 1 s (every VIDEO_FPS * 1.0 frames by index).
"""

import pickle
from pathlib import Path

MTL_ROOT = Path(__file__).parent.parent.parent / "datasets" / "MTL-AQA"
RAW_ROOT = Path(__file__).parent.parent.parent / "datasets" / "MTL-AQA_raw_full"
VIDEO_FPS = 25.0
SAMPLE_INTERVAL = round(VIDEO_FPS * 1.0)  # 1 frame per 1 s


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    train_split = load_pickle(MTL_ROOT / "train_split.pkl")
    test_split = load_pickle(MTL_ROOT / "test_split.pkl")
    annotations = load_pickle(MTL_ROOT / "annotations.pkl")

    all_keys = {(v, c) for v, c in train_split} | {(v, c) for v, c in test_split}
    print(f"{len(train_split)} train + {len(test_split)} test = {len(all_keys)} total clips")

    # 1. Trim frames outside [start_frame, end_frame]
    removed_frames = 0
    for video_id, clip_id in sorted(all_keys):
        ann = annotations[(video_id, clip_id)]
        start, end = ann["start_frame"], ann["end_frame"]
        clip_dir = RAW_ROOT / f"{video_id:02d}_{clip_id:02d}"
        if not clip_dir.is_dir():
            print(f"  WARNING: missing clip dir {clip_dir.name}, skipping")
            continue
        for frame in clip_dir.glob("*.jpg"):
            n = int(frame.stem)
            if n < start or n > end:
                frame.unlink()
                removed_frames += 1

    print(f"Removed {removed_frames} out-of-range frames across {len(all_keys)} clips.")

    # 2. Subsample: keep every SAMPLE_INTERVAL-th frame by sorted index
    removed_sampled = 0
    for video_id, clip_id in sorted(all_keys):
        clip_dir = RAW_ROOT / f"{video_id:02d}_{clip_id:02d}"
        if not clip_dir.is_dir():
            continue
        frames = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        for idx, frame in enumerate(frames):
            if idx % SAMPLE_INTERVAL != 0:
                frame.unlink()
                removed_sampled += 1

    print(f"Removed {removed_sampled} frames during subsampling (kept every {SAMPLE_INTERVAL}th).")


if __name__ == "__main__":
    main()
