"""
Sample 1 frame/second from each CRAFT .mpg clip and write a CRAFT_frames dataset.

Output layout:
  datasets/CRAFT_frames/
    frames/
      sid_1/
        000000/
          frame_000.jpg
          frame_001.jpg
          ...
    dataset.json   -- original QA entries + "frame_paths" list
"""

import json
from pathlib import Path

import cv2

CRAFT_ROOT = Path(__file__).parent.parent / "datasets" / "CRAFT"
OUT_ROOT = Path(__file__).parent.parent / "datasets" / "CRAFT_frames"
FRAMES_ROOT = OUT_ROOT / "frames"
MAX_SCENES_PER_SID = 20


# --- I/O helpers -------------------------------------------------------------


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_dataset(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


# --- Path helpers ------------------------------------------------------------


def video_abs_path(craft_root: Path, rel_path: str) -> Path:
    return craft_root / rel_path.lstrip("./")


def frame_out_dir(frames_root: Path, rel_path: str) -> Path:
    """Map './videos/sid_1/000000.mpg' -> frames_root/sid_1/000000/"""
    parts = Path(rel_path).parts
    sid = parts[-2]
    clip = Path(parts[-1]).stem
    return frames_root / sid / clip


def cached_frames(out_dir: Path) -> list[str] | None:
    """Return sorted relative frame paths if already sampled, else None."""
    if out_dir.exists() and any(out_dir.iterdir()):
        return sorted(str(p.relative_to(OUT_ROOT)) for p in out_dir.glob("*.jpg"))
    return None


# --- Sampling ----------------------------------------------------------------


def open_video(path: Path) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(str(path))
    return cap if cap.isOpened() else None


def extract_frames(cap: cv2.VideoCapture, out_dir: Path) -> list[str]:
    """Write 1 frame/second to out_dir, return relative paths from OUT_ROOT."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 75.0
    step = max(1, round(fps))

    saved, frame_idx, sample_idx = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = out_dir / f"frame_{sample_idx:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(str(out_path.relative_to(OUT_ROOT)))
            sample_idx += 1
        frame_idx += 1

    return saved


def sample_video(video_path: Path, out_dir: Path) -> list[str]:
    cap = open_video(video_path)
    if cap is None:
        print(f"  WARNING: could not open {video_path}")
        return []
    frames = extract_frames(cap, out_dir)
    cap.release()
    return frames


# --- Scene trimming ----------------------------------------------------------


def trim_sid_folder(sid_dir: Path, max_scenes: int) -> list[str]:
    """Delete clip dirs beyond the first max_scenes in a sid folder. Returns kept clip names."""
    clip_dirs = sorted(sid_dir.iterdir()) if sid_dir.exists() else []
    kept, removed = clip_dirs[:max_scenes], clip_dirs[max_scenes:]
    for clip_dir in removed:
        for f in clip_dir.glob("*"):
            f.unlink()
        clip_dir.rmdir()
    if removed:
        print(
            f"  trimmed {sid_dir.name}: removed {len(removed)} clips, kept {len(kept)}"
        )
    return [d.name for d in kept]


def trim_all_sids(frames_root: Path, max_scenes: int) -> dict[str, set[str]]:
    """Trim every sid_N folder. Returns {sid_name: {kept_clip_stems}}."""
    result = {}
    for sid_dir in sorted(frames_root.iterdir()):
        if sid_dir.is_dir():
            result[sid_dir.name] = set(trim_sid_folder(sid_dir, max_scenes))
    return result


def filter_entries_by_kept(
    entries: list[dict], kept: dict[str, set[str]]
) -> list[dict]:
    """Keep only entries whose clip is in the kept set for its sid."""

    def is_kept(entry: dict) -> bool:
        parts = Path(
            entry["video_file_path"]
        ).parts  # ('.', 'videos', 'sid_1', '000000.mpg')
        sid, clip_stem = parts[-2], Path(parts[-1]).stem
        return clip_stem in kept.get(sid, set())

    return [e for e in entries if is_kept(e)]


# --- Per-video orchestration -------------------------------------------------


def process_video(rel_path: str, idx: int, total: int) -> list[str]:
    out_dir = frame_out_dir(FRAMES_ROOT, rel_path)

    frames = cached_frames(out_dir)
    if frames is not None:
        print(f"  [{idx}/{total}] skip (cached) {rel_path}  ({len(frames)} frames)")
        return frames

    video_path = video_abs_path(CRAFT_ROOT, rel_path)
    if not video_path.exists():
        print(f"  [{idx}/{total}] MISSING {video_path}")
        return []

    frames = sample_video(video_path, out_dir)
    print(f"  [{idx}/{total}] {rel_path}  -> {len(frames)} frames")
    return frames


def build_frame_index(entries: list[dict]) -> dict[str, list[str]]:
    unique = sorted({e["video_file_path"] for e in entries})
    total = len(unique)
    print(f"Found {total} unique videos, {len(entries)} QA entries.")
    return {rel: process_video(rel, i, total) for i, rel in enumerate(unique, 1)}


# --- Dataset enrichment ------------------------------------------------------


def enrich_entries(
    entries: list[dict], frame_index: dict[str, list[str]]
) -> list[dict]:
    return [
        {**e, "frame_paths": frame_index.get(e["video_file_path"], [])} for e in entries
    ]


# --- Entry point -------------------------------------------------------------


def main():
    entries = load_dataset(CRAFT_ROOT / "dataset_minimal.json")
    frame_index = build_frame_index(entries)
    enriched = enrich_entries(entries, frame_index)

    kept = trim_all_sids(FRAMES_ROOT, MAX_SCENES_PER_SID)
    enriched = filter_entries_by_kept(enriched, kept)

    save_dataset(enriched, OUT_ROOT / "dataset.json")
    print(f"\nDone. {len(enriched)} entries written to {OUT_ROOT / 'dataset.json'}")


if __name__ == "__main__":
    main()
