"""
Firestore access layer — replaces SQLite.

Collections:
  participants  — doc ID: "{prolific_pid}__{dataset}__{task}"
  annotations   — doc ID: "{prolific_pid}__{dataset}__{task}__{sample_index}"

The Firestore client uses Application Default Credentials automatically on
Cloud Run. For local dev run: gcloud auth application-default login
"""
from datetime import datetime, timezone
from typing import Union

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore

_db = firestore.Client()
_participants = _db.collection("participants")
_annotations = _db.collection("annotations")


def _pid_key(prolific_pid: str, dataset: str, task: str, scene_id: str = "main") -> str:
    return f"{prolific_pid}__{dataset}__{task}__{scene_id}"


def _ann_key(prolific_pid: str, dataset: str, task: str, sample_index: int, scene_id: str = "main") -> str:
    return f"{prolific_pid}__{dataset}__{task}__{sample_index}__{scene_id}"


def init_db() -> None:
    """No-op — Firestore needs no schema initialisation."""
    pass


def upsert_participant(
    prolific_pid: str,
    dataset: str,
    task: str,
    seed: int,
    scene_id: str = "main",
    study_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Create participant record; silently ignore if it already exists (first visit wins)."""
    doc_ref = _participants.document(_pid_key(prolific_pid, dataset, task, scene_id))
    try:
        doc_ref.create({
            "prolific_pid": prolific_pid,
            "dataset": dataset,
            "task": task,
            "scene_id": scene_id,
            "seed": seed,
            "study_id": study_id,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        })
    except AlreadyExists:
        pass


def upsert_annotation(
    prolific_pid: str,
    dataset: str,
    task: str,
    sample_index: int,
    sample_name: str,
    n_frames: int,
    ground_truth: Union[bool, tuple[int, int]],
    human_answer: Union[bool, tuple[int, int]],
    scene_id: str = "main",
    is_attention_check: bool = False,
    view_duration_s: float | None = None,
    viewer_opens: int | None = None,
) -> None:
    """Insert or update an annotation (idempotent)."""
    doc_ref = _annotations.document(_ann_key(prolific_pid, dataset, task, sample_index, scene_id))
    data: dict = {
        "prolific_pid": prolific_pid,
        "dataset": dataset,
        "task": task,
        "scene_id": scene_id,
        "sample_index": sample_index,
        "sample_name": sample_name,
        "n_frames": n_frames,
        "ground_truth": list(ground_truth) if isinstance(ground_truth, tuple) else ground_truth,
        "human_answer": list(human_answer) if isinstance(human_answer, tuple) else human_answer,
        "is_attention_check": is_attention_check,
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }
    if view_duration_s is not None:
        data["view_duration_s"] = view_duration_s
    if viewer_opens is not None:
        data["viewer_opens"] = viewer_opens
    doc_ref.set(data)


def mark_completed(prolific_pid: str, dataset: str, task: str, scene_id: str = "main") -> None:
    doc_ref = _participants.document(_pid_key(prolific_pid, dataset, task, scene_id))
    doc_ref.update({"completed_at": datetime.now(timezone.utc).isoformat()})


def count_annotations(prolific_pid: str, dataset: str, task: str, scene_id: str = "main") -> int:
    docs = _annotations.where("prolific_pid", "==", prolific_pid) \
                        .where("dataset", "==", dataset) \
                        .where("task", "==", task) \
                        .where("scene_id", "==", scene_id) \
                        .get()
    return len(docs)


def update_instruction_metrics(
    prolific_pid: str,
    dataset: str,
    task: str,
    scene_id: str = "main",
    instructions_duration_s: float | None = None,
    instructions_lightbox_opens: int | None = None,
    video_played: bool | None = None,
) -> None:
    updates: dict = {}
    if instructions_duration_s is not None:
        updates["instructions_duration_s"] = instructions_duration_s
    if instructions_lightbox_opens is not None:
        updates["instructions_lightbox_opens"] = instructions_lightbox_opens
    if video_played is not None:
        updates["video_played"] = video_played
    if not updates:
        return
    doc_ref = _participants.document(_pid_key(prolific_pid, dataset, task, scene_id))
    doc_ref.set(updates, merge=True)


# ── Kept for compatibility with tests / export scripts ────────────────────────

def _encode(value: Union[bool, tuple[int, int]]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value[0]},{value[1]}"


def decode_answer(raw: str, task: str) -> Union[bool, tuple[int, int]]:
    if task == "detect":
        return raw.lower() == "true"
    parts = raw.split(",")
    return (int(parts[0]), int(parts[1]))
