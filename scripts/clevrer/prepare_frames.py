"""
Sample ~1.5 frames/second from each CLEVRER .mp4 clip and write frames into the CLEVRER dataset.

Output layout:
  datasets/CLEVRER/
    scenes/
      scene_13005/
        0.jpg
        1.jpg
        ...
    dataset.json   -- one entry per video with frame_paths and scene_text
"""

import json
from pathlib import Path

import cv2

CLEVRER_ROOT = Path(__file__).parent.parent.parent / "datasets" / "CLEVRER"
OUT_ROOT = Path(__file__).parent.parent.parent / "datasets" / "CLEVRER"
FRAMES_ROOT = OUT_ROOT / "scenes"
ANNOTATIONS_ROOT = CLEVRER_ROOT / "annotations" / "annotation_validation"
SAMPLES_ROOT = CLEVRER_ROOT / "samples"
VIDEO_FPS = 25.0


# --- I/O helpers -------------------------------------------------------------


def load_annotation(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_dataset(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


# --- Path helpers ------------------------------------------------------------


def find_all_annotations() -> list[Path]:
    return sorted(ANNOTATIONS_ROOT.glob("annotation_*/*.json"))


def video_path(video_filename: str) -> Path | None:
    vid_id = Path(video_filename).stem  # "video_13005"
    num = int(vid_id.split("_")[1])
    bucket_start = (num // 1000) * 1000
    bucket = f"video_{bucket_start}-{bucket_start + 1000}"
    p = SAMPLES_ROOT / bucket / video_filename
    return p if p.exists() else None


def frame_out_dir(video_filename: str) -> Path:
    vid_id = Path(video_filename).stem  # "video_13005"
    scene_id = vid_id.replace("video_", "scene_")
    return FRAMES_ROOT / scene_id


def cached_frames(out_dir: Path) -> list[str] | None:
    if out_dir.exists() and any(out_dir.iterdir()):
        return sorted(str(p.relative_to(OUT_ROOT)) for p in out_dir.glob("*.jpg"))
    return None


# --- Sampling ----------------------------------------------------------------


def open_video(path: Path) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(str(path))
    return cap if cap.isOpened() else None


def extract_frames(cap: cv2.VideoCapture, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    step = max(1, round(fps / 1.5))

    saved, frame_idx, sample_idx = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = out_dir / f"{sample_idx}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(str(out_path.relative_to(OUT_ROOT)))
            sample_idx += 1
        frame_idx += 1

    return saved


def sample_video(vid_path: Path, out_dir: Path) -> list[str]:
    cap = open_video(vid_path)
    if cap is None:
        print(f"  WARNING: could not open {vid_path}")
        return []
    frames = extract_frames(cap, out_dir)
    cap.release()
    return frames


def process_video(video_filename: str, idx: int, total: int) -> list[str]:
    out_dir = frame_out_dir(video_filename)

    frames = cached_frames(out_dir)
    if frames is not None:
        print(
            f"  [{idx}/{total}] skip (cached) {video_filename}  ({len(frames)} frames)"
        )
        return frames

    vid_path = video_path(video_filename)
    if vid_path is None:
        print(f"  [{idx}/{total}] MISSING {video_filename}")
        return []

    frames = sample_video(vid_path, out_dir)
    print(f"  [{idx}/{total}] {video_filename}  -> {len(frames)} frames")
    return frames


# --- Scene text --------------------------------------------------------------


_SHAPE_MAP = {"sphere": "ball"}


def build_scene_text(annotation: dict) -> str:
    props = {o["object_id"]: o for o in annotation["object_property"]}

    def fmt(obj_id: int) -> str | None:
        o = props.get(obj_id, {})
        color = o.get("color", "")
        shape = _SHAPE_MAP.get(o.get("shape", ""), o.get("shape", ""))
        desc = f"{color} {shape}".strip()
        return desc or None

    seen: set[str] = set()
    parts: list[str] = []
    for c in sorted(annotation["collision"], key=lambda c: c["frame_id"]):
        a = fmt(c["object_ids"][0])
        b = fmt(c["object_ids"][1])
        if not a or not b:
            continue
        sentence = f"{a} will collide with {b}"
        if sentence not in seen:
            seen.add(sentence)
            parts.append(sentence)

    if not parts:
        names = [fmt(oid) for oid in sorted(props)]
        names = [n for n in names if n]
        return "In the scene, " + ", ".join(names) + " are present."
    if len(parts) == 1:
        return f"In the scene, {parts[0]}."
    return "In the scene, " + ", and ".join([", ".join(parts[:-1]), parts[-1]]) + "."


# --- Entry point -------------------------------------------------------------


def main():
    ann_paths = find_all_annotations()
    total = len(ann_paths)
    print(f"Found {total} annotation files.")

    entries = []
    for idx, ann_path in enumerate(ann_paths, 1):
        ann = load_annotation(ann_path)
        video_filename = ann["video_filename"]

        frame_paths = process_video(video_filename, idx, total)
        scene_text = build_scene_text(ann)

        entries.append(
            {
                "video_index": ann["scene_index"],
                "video_filename": video_filename,
                "frame_paths": frame_paths,
                "scene_text": scene_text,
            }
        )

    save_dataset(entries, OUT_ROOT / "dataset.json")
    print(f"\nDone. {len(entries)} entries written to {OUT_ROOT / 'dataset.json'}")


if __name__ == "__main__":
    main()
