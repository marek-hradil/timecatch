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
SPLIT_FILE = CRAFT_ROOT / "split_info_random.json"
FULL_DATASET = CRAFT_ROOT / "dataset.json"
SIM_FPS = 75.0


# --- I/O helpers -------------------------------------------------------------


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_test_entries(dataset: list[dict], split_path: Path) -> list[dict]:
    with open(split_path) as f:
        split = json.load(f)
    test_keys = {(e["video_index"], e["question_index"]) for e in split["test"]}
    filtered = [
        e for e in dataset
        if (e["video_index"], e["question_index"]) in test_keys
    ]
    print(f"Test split: {len(filtered)}/{len(dataset)} entries retained.")
    return filtered


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


# --- Scene text extraction ---------------------------------------------------


def _extract_object_labels(questions: list[dict]) -> dict[int, dict]:
    """Scan all filter steps to build {obj_idx: {size, color, shape}}."""
    labels: dict[int, dict] = {}
    for q in questions:
        for step in q["program"]:
            t = step["type"]
            vi = step.get("value_inputs", [])
            out = step.get("_output")
            if t in ("filter_color", "filter_shape", "filter_size") and vi and isinstance(out, list):
                attr = t[len("filter_"):]
                for obj in out:
                    if isinstance(obj, int):
                        labels.setdefault(obj, {})[attr] = vi[0]
    return labels


def _extract_dynamic_objects(questions: list[dict]) -> set[int]:
    dynamic: set[int] = set()
    for q in questions:
        for step in q["program"]:
            if step["type"] == "filter_dynamic_objects" and isinstance(step.get("_output"), list):
                dynamic.update(o for o in step["_output"] if isinstance(o, int))
    return dynamic


def _extract_events(questions: list[dict]) -> list[dict]:
    """Return the canonical physics event list from the first 'events' step found."""
    for q in questions:
        for step in q["program"]:
            if step["type"] == "events" and isinstance(step.get("_output"), list):
                return [e for e in step["_output"] if isinstance(e, dict)]
    return []


def _fmt_obj(idx: int, labels: dict[int, dict]) -> str:
    props = labels.get(idx, {})
    desc = " ".join(p for p in [props.get("size"), props.get("color"), props.get("shape")] if p)
    return desc or f"object {idx}"


def build_scene_text(questions: list[dict]) -> str:
    labels = _extract_object_labels(questions)
    dynamic = _extract_dynamic_objects(questions)
    events = _extract_events(questions)

    # Identify basket: static object appearing as objects[0] in ContainerEndUp events
    basket_ids = {
        e["objects"][0]
        for e in events
        if e["type"] == "ContainerEndUp" and len(e.get("objects", [])) >= 2
    }

    def fmt(idx: int) -> str:
        if idx in basket_ids:
            return "basket"
        if idx not in dynamic:
            return "wall/ramp"
        return _fmt_obj(idx, labels)

    obj_descs = [_fmt_obj(i, labels) for i in sorted(dynamic)]
    objects_line = "Objects: " + (", ".join(obj_descs) if obj_descs else "unknown")

    # Deduplicate: keep first occurrence of each (event_type, unordered_pair).
    # Collision events repeat dozens of times per pair — one mention is enough.
    seen: set[tuple] = set()
    lines: list[tuple[int, str]] = []

    for e in events:
        t = e["type"]
        if t in ("Start", "End"):
            continue
        objs = e.get("objects", [])
        step = e["step"]
        pair = tuple(sorted(objs[:2])) if len(objs) >= 2 else tuple(objs)
        key = (t, pair)
        if key in seen:
            continue
        seen.add(key)

        time_s = step / SIM_FPS
        a = fmt(objs[0]) if len(objs) > 0 else "?"
        b = fmt(objs[1]) if len(objs) > 1 else "?"

        if t == "Collision":
            line = f"  {time_s:.2f}s: {a} collides with {b}"
        elif t == "StartTouching":
            line = f"  {time_s:.2f}s: {a} starts touching {b}"
        elif t == "EndTouching":
            line = f"  {time_s:.2f}s: {a} stops touching {b}"
        elif t == "ContainerEndUp":
            # objects[0] is basket (static), objects[1] is the entering object
            entering = fmt(objs[1]) if len(objs) > 1 else "?"
            line = f"  {time_s:.2f}s: {entering} enters basket"
        else:
            line = f"  {time_s:.2f}s: {t} {a} {b}"

        lines.append((step, line))

    lines.sort()
    return objects_line + "\nEvents:\n" + "\n".join(l for _, l in lines)


def build_scene_index(full_entries: list[dict]) -> dict[int, str]:
    """Returns {video_index: scene_text} for every video in the full dataset."""
    return {
        entry["questions"]["info"]["video_index"]: build_scene_text(entry["questions"]["questions"])
        for entry in full_entries
    }


# --- Dataset enrichment ------------------------------------------------------


def enrich_entries(
    entries: list[dict],
    frame_index: dict[str, list[str]],
    scene_index: dict[int, str],
) -> list[dict]:
    return [
        {
            **e,
            "frame_paths": frame_index.get(e["video_file_path"], []),
            "scene_text": scene_index.get(e["video_index"], ""),
        }
        for e in entries
    ]


# --- Entry point -------------------------------------------------------------


def main():
    entries = load_dataset(CRAFT_ROOT / "dataset_minimal.json")
    entries = load_test_entries(entries, SPLIT_FILE)
    frame_index = build_frame_index(entries)

    print("Building scene text index...")
    scene_index = build_scene_index(load_dataset(FULL_DATASET))

    enriched = enrich_entries(entries, frame_index, scene_index)

    kept = trim_all_sids(FRAMES_ROOT, MAX_SCENES_PER_SID)
    enriched = filter_entries_by_kept(enriched, kept)

    save_dataset(enriched, OUT_ROOT / "dataset.json")
    print(f"\nDone. {len(enriched)} entries written to {OUT_ROOT / 'dataset.json'}")


if __name__ == "__main__":
    main()
