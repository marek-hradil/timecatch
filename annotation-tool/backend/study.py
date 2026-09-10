"""
Pure plan-builder — no FastAPI, no DB imports.

build_plan(prolific_pid, dataset, task, n) -> list of SamplePlan dicts
ground_truth_for(seed, dataset, task, n) -> list of ground-truth values

Seed derivation: hashlib so it is stable across process restarts
(Python's built-in hash() is randomized per process via PYTHONHASHSEED).
"""
import hashlib
import random as _random
from dataclasses import dataclass
from typing import Union

from .config import VALID_DATASETS, VALID_TASKS, SAMPLES_PER_PARTICIPANT, N_ATTENTION_CHECKS
from .sampling import sample_binary_paths, sample_position_paths, load_attention_check_samples


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
    is_attention_check: bool = False


@dataclass
class SampleGroundTruth:
    sample_index: int
    ground_truth: Union[bool, tuple[int, int]]


_INTERLEAVE_MASK = 0x5EED1337


def _interleave(
    regular: list,
    checks: list,
    seed: int,
    n: int,
) -> list[tuple]:
    """Randomly insert attention checks into the regular sample list."""
    if not checks:
        return [(s, False) for s in regular]
    rng = _random.Random(seed ^ _INTERLEAVE_MASK)
    check_positions = set(rng.sample(range(n), len(checks)))
    result = []
    reg_iter = iter(regular)
    chk_iter = iter(checks)
    for i in range(n):
        if i in check_positions:
            result.append((next(chk_iter), True))
        else:
            result.append((next(reg_iter), False))
    return result


def _validate(dataset: str, task: str) -> None:
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Valid: {sorted(VALID_DATASETS)}")
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task {task!r}. Valid: {sorted(VALID_TASKS)}")


def _regular_samples(dataset: str, task: str, seed: int, n: int, exclude: set[str]) -> list:
    """
    Draw n regular samples, excluding any scene names in `exclude`.
    Requests n + len(exclude) candidates so filtering never leaves us short.
    """
    candidate_n = n + len(exclude)
    if task == "detect":
        candidates = sample_binary_paths(dataset, seed, candidate_n)
    else:
        candidates = sample_position_paths(dataset, seed, candidate_n)
    filtered = [s for s in candidates if s.sample_name not in exclude]
    return filtered[:n]


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

    checks = load_attention_check_samples(dataset, task, N_ATTENTION_CHECKS)
    n_regular = n - len(checks)
    exclude = {s.sample_name for s in checks}

    regular = _regular_samples(dataset, task, seed, n_regular, exclude)
    interleaved = _interleave(regular, checks, seed, n)

    return [
        SamplePlan(
            sample_index=i,
            dataset=dataset,
            task=task,
            sample_name=s.sample_name,
            n_frames=s.n_frames,
            frame_urls=s.frame_paths,
            scene_description=s.scene_description,
            is_attention_check=is_check,
        )
        for i, (s, is_check) in enumerate(interleaved)
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

    checks = load_attention_check_samples(dataset, task, N_ATTENTION_CHECKS)
    n_regular = n - len(checks)
    exclude = {s.sample_name for s in checks}

    regular = _regular_samples(dataset, task, seed, n_regular, exclude)
    interleaved = _interleave(regular, checks, seed, n)

    return [SampleGroundTruth(i, s.ground_truth) for i, (s, _) in enumerate(interleaved)]
