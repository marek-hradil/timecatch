"""
Pure plan-builder — no FastAPI, no DB imports.

build_plan(prolific_pid, dataset, task, n) -> list of SamplePlan dicts
ground_truth_for(seed, dataset, task, n) -> list of ground-truth values

Seed derivation: hashlib so it is stable across process restarts
(Python's built-in hash() is randomized per process via PYTHONHASHSEED).
"""
import hashlib
from dataclasses import dataclass
from typing import Union

from .config import VALID_DATASETS, VALID_TASKS, SAMPLES_PER_PARTICIPANT
from .sampling import sample_binary_paths, sample_position_paths


def pid_to_seed(prolific_pid: str) -> int:
    return int(hashlib.sha256(prolific_pid.encode()).hexdigest(), 16) % (2**31)


@dataclass
class SamplePlan:
    sample_index: int
    dataset: str
    task: str
    sample_name: str
    n_frames: int
    frame_urls: list[str]
    scene_description: str | None


@dataclass
class SampleGroundTruth:
    sample_index: int
    ground_truth: Union[bool, tuple[int, int]]


def _validate(dataset: str, task: str) -> None:
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Valid: {sorted(VALID_DATASETS)}")
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task {task!r}. Valid: {sorted(VALID_TASKS)}")


def build_plan(
    prolific_pid: str,
    dataset: str,
    task: str,
    n: int = SAMPLES_PER_PARTICIPANT,
) -> list[SamplePlan]:
    """
    Returns the ordered sample plan for a participant.
    Ground truth is intentionally excluded — it stays server-side.
    """
    _validate(dataset, task)
    seed = pid_to_seed(prolific_pid)

    if task == "detect":
        samples = sample_binary_paths(dataset, seed, n)
    else:
        samples = sample_position_paths(dataset, seed, n)

    return [
        SamplePlan(
            sample_index=i,
            dataset=dataset,
            task=task,
            sample_name=s.sample_name,
            n_frames=s.n_frames,
            frame_urls=s.frame_paths,
            scene_description=s.scene_description,
        )
        for i, s in enumerate(samples)
    ]


def ground_truth_for(
    prolific_pid: str,
    dataset: str,
    task: str,
    n: int = SAMPLES_PER_PARTICIPANT,
) -> list[SampleGroundTruth]:
    """
    Server-side ground truth list (not sent to client).
    Recomputed on demand — cheap (no image loading).
    """
    _validate(dataset, task)
    seed = pid_to_seed(prolific_pid)

    if task == "detect":
        samples = sample_binary_paths(dataset, seed, n)
        return [SampleGroundTruth(i, s.ground_truth) for i, s in enumerate(samples)]
    else:
        samples = sample_position_paths(dataset, seed, n)
        return [SampleGroundTruth(i, s.ground_truth) for i, s in enumerate(samples)]
