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


def _pid_key(prolific_pid: str, dataset: str, task: str) -> str:
    return f"{prolific_pid}__{dataset}__{task}"


def _ann_key(prolific_pid: str, dataset: str, task: str, sample_index: int) -> str:
    return f"{prolific_pid}__{dataset}__{task}__{sample_index}"


def init_db() -> None:
    """No-op — Firestore needs no schema initialisation."""
    pass


def upsert_participant(prolific_pid: str, dataset: str, task: str, seed: int) -> None:
    """Create participant record; silently ignore if it already exists (first visit wins)."""
    doc_ref = _participants.document(_pid_key(prolific_pid, dataset, task))
    try:
        doc_ref.create({
            "prolific_pid": prolific_pid,
            "dataset": dataset,
            "task": task,
            "seed": seed,
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
) -> None:
    """Insert or update an annotation (idempotent)."""
    doc_ref = _annotations.document(_ann_key(prolific_pid, dataset, task, sample_index))
    doc_ref.set({
        "prolific_pid": prolific_pid,
        "dataset": dataset,
        "task": task,
        "sample_index": sample_index,
        "sample_name": sample_name,
        "n_frames": n_frames,
        "ground_truth": list(ground_truth) if isinstance(ground_truth, tuple) else ground_truth,
        "human_answer": list(human_answer) if isinstance(human_answer, tuple) else human_answer,
        "answered_at": datetime.now(timezone.utc).isoformat(),
    })


def mark_completed(prolific_pid: str, dataset: str, task: str) -> None:
    doc_ref = _participants.document(_pid_key(prolific_pid, dataset, task))
    doc_ref.update({"completed_at": datetime.now(timezone.utc).isoformat()})


def count_annotations(prolific_pid: str, dataset: str, task: str) -> int:
    docs = _annotations.where("prolific_pid", "==", prolific_pid) \
                        .where("dataset", "==", dataset) \
                        .where("task", "==", task) \
                        .get()
    return len(docs)


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
