"""
Sample ~1.5 frames/second from EVERY CRAFT .mpg clip (train + validation +
test splits combined, 9,917 videos) and write frames into the CRAFT
dataset — not just the 1,983-video test split scripts/craft/prepare_frames.py
originally used.

dataset_minimal.json's entries are exactly the union of split_info_random's
train (5,951 unique videos) + validation (1,983) + test (1,983) splits, so
no split filtering is needed here to get "everything".

Frame extraction for the 1,983 test-split videos is already cached under
datasets/CRAFT/scenes/ from the original run and will be skipped.

Output layout:
  datasets/CRAFT/
    scenes/
      scene_1_000000/
        0.jpg
        1.jpg
          ...
    scenes_full_raw.json   -- ALL QA entries + "frame_paths" + "scene_text"
"""

import json
from pathlib import Path

import cv2

CRAFT_ROOT = Path(__file__).parent.parent.parent / "datasets" / "CRAFT"
OUT_ROOT = Path(__file__).parent.parent.parent / "datasets" / "CRAFT"
FRAMES_ROOT = OUT_ROOT / "scenes"
FULL_DATASET = CRAFT_ROOT / "dataset.json"
MINIMAL_DATASET = CRAFT_ROOT / "dataset_minimal.json"
SIM_FPS = 75.0


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
    """Map './videos/sid_1/000000.mpg' -> frames_root/scene_1_000000/"""
    parts = Path(rel_path).parts
    sid_num = parts[-2].split("_")[1]  # "sid_1" -> "1"
    clip = Path(parts[-1]).stem        # "000000"
    return frames_root / f"scene_{sid_num}_{clip}"


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
    """Write 1 frame/1.5s to out_dir, return relative paths from OUT_ROOT."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 75.0
    step = max(1, round(fps * 1.5))

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


def sample_video(video_path: Path, out_dir: Path) -> list[str]:
    cap = open_video(video_path)
    if cap is None:
        print(f"  WARNING: could not open {video_path}")
        return []
    frames = extract_frames(cap, out_dir)
    cap.release()
    return frames


# --- Per-video orchestration -------------------------------------------------


def process_video(rel_path: str, idx: int, total: int) -> list[str]:
    out_dir = frame_out_dir(FRAMES_ROOT, rel_path)

    frames = cached_frames(out_dir)
    if frames is not None:
        if idx % 500 == 0:
            print(f"  [{idx}/{total}] skip (cached) {rel_path}  ({len(frames)} frames)")
        return frames

    video_path = video_abs_path(CRAFT_ROOT, rel_path)
    if not video_path.exists():
        print(f"  [{idx}/{total}] MISSING {video_path}")
        return []

    frames = sample_video(video_path, out_dir)
    if idx % 200 == 0:
        print(f"  [{idx}/{total}] {rel_path}  -> {len(frames)} frames")
    return frames


def build_frame_index(entries: list[dict]) -> dict[str, list[str]]:
    unique = sorted({e["video_file_path"] for e in entries})
    total = len(unique)
    print(f"Found {total} unique videos, {len(entries)} QA entries.")
    return {rel: process_video(rel, i, total) for i, rel in enumerate(unique, 1)}


# --- Scene text extraction (identical to prepare_frames.py) -----------------


def _extract_object_labels(questions: list[dict]) -> dict[int, dict]:
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
    for q in questions:
        for step in q["program"]:
            if step["type"] == "events" and isinstance(step.get("_output"), list):
                return [e for e in step["_output"] if isinstance(e, dict)]
    return []


def _fmt_obj(idx: int, labels: dict[int, dict], dynamic: set[int], basket_ids: set[int]) -> str | None:
    if idx in basket_ids:
        return "basket"
    if idx not in dynamic:
        return None  # wall/ramp
    props = labels.get(idx, {})
    if not props.get("color") and not props.get("shape"):
        return None  # incomplete label — skip rather than emit "object N"
    parts = [p for p in [props.get("size"), props.get("color"), props.get("shape")] if p]
    return " ".join(parts)


def build_scene_text(questions: list[dict]) -> str:
    labels = _extract_object_labels(questions)
    dynamic = _extract_dynamic_objects(questions)
    events = _extract_events(questions)

    basket_ids = {
        e["objects"][0]
        for e in events
        if e["type"] == "ContainerEndUp" and len(e.get("objects", [])) >= 2
    }

    def fmt(idx: int) -> str | None:
        return _fmt_obj(idx, labels, dynamic, basket_ids)

    seen_sentences: set[str] = set()
    parts: list[tuple[int, str]] = []

    for e in events:
        t = e["type"]
        if t in ("Start", "End", "StartTouching", "EndTouching"):
            continue
        objs = e.get("objects", [])

        sentence = None
        if t == "Collision":
            a = fmt(objs[0]) if objs else None
            b = fmt(objs[1]) if len(objs) > 1 else None
            if a and b and "basket" not in (a, b):
                sentence = f"{a} will collide with {b}"
        elif t == "ContainerEndUp":
            entering = fmt(objs[1]) if len(objs) > 1 else None
            if entering and entering != "basket":
                sentence = f"{entering} will enter the basket"

        if sentence and sentence not in seen_sentences:
            seen_sentences.add(sentence)
            parts.append((e["step"], sentence))

    parts.sort()
    event_strs = [s for _, s in parts]

    if not event_strs:
        obj_descs = [fmt(i) for i in sorted(dynamic)]
        names = ", ".join(o for o in obj_descs if o and o != "basket") or "unknown objects"
        return f"In the scene, {names} are present."
    if len(event_strs) == 1:
        return f"In the scene, {event_strs[0]}."
    return "In the scene, " + ", and ".join([", ".join(event_strs[:-1]), event_strs[-1]]) + "."


def build_scene_index(full_entries: list[dict]) -> dict[int, str]:
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
    entries = load_dataset(MINIMAL_DATASET)
    print(f"Using ALL {len(entries)} entries from dataset_minimal.json (train+validation+test)")
    frame_index = build_frame_index(entries)

    print("Building scene text index...")
    scene_index = build_scene_index(load_dataset(FULL_DATASET))

    enriched = enrich_entries(entries, frame_index, scene_index)

    save_dataset(enriched, OUT_ROOT / "scenes_full_raw.json")
    print(f"\nDone. {len(enriched)} entries written to {OUT_ROOT / 'scenes_full_raw.json'}")


if __name__ == "__main__":
    main()
