"""Extract 8 uniformly-spaced frames per TempCompass video (410 videos,
403MB of .mp4 -> a much smaller set of .jpg frames, so only the frames --
not the videos -- need to be rsynced to the cluster).

Reads:  datasets/TempCompass/videos/<video_id>.mp4
Writes: datasets/TempCompass/frames/<video_id>/0.jpg .. 7.jpg

Usage:
    python3 scripts/tempcompass_prepare_frames.py
"""

from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent / "datasets" / "TempCompass"
VIDEOS_DIR = ROOT / "videos"
FRAMES_DIR = ROOT / "frames"
N_FRAMES = 8


def extract(video_path: Path, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    indices = [round(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)] if total > 1 else [0]

    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(str(out_dir / f"{i}.jpg"), frame)
        saved += 1
    cap.release()
    return saved


def main():
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    print(f"Found {len(videos)} videos")
    for i, video_path in enumerate(videos, 1):
        video_id = video_path.stem
        out_dir = FRAMES_DIR / video_id
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == N_FRAMES:
            continue
        n = extract(video_path, out_dir)
        if i % 50 == 0 or n < N_FRAMES:
            print(f"  [{i}/{len(videos)}] {video_id}: {n} frames")

    print("Done.")


if __name__ == "__main__":
    main()
