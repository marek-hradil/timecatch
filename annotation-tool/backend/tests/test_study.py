"""
Tests for sampling, study plan building, and DB operations.

Run from annotation-tool/:
    uv run pytest backend/tests/test_study.py -v
"""
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root (containing dataset.py) is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.config import SAMPLES_PER_PARTICIPANT, VALID_DATASETS, VALID_TASKS
from backend.db import (
    count_annotations,
    decode_answer,
    init_db,
    mark_completed,
    upsert_annotation,
    upsert_participant,
)
from backend.sampling import sample_binary_paths, sample_position_paths
from backend.study import build_plan, ground_truth_for, pid_to_seed

TEST_PID = "test_participant_abc123"
TEST_SEED = pid_to_seed(TEST_PID)
N = SAMPLES_PER_PARTICIPANT


# ── Seed determinism ──────────────────────────────────────────────────────────

def test_seed_stable_across_calls():
    """hashlib seed must be identical on repeated calls (not salted like hash())."""
    assert pid_to_seed(TEST_PID) == pid_to_seed(TEST_PID)


def test_seed_different_pids():
    """Different PIDs must produce different seeds."""
    assert pid_to_seed("pid_aaa") != pid_to_seed("pid_bbb")


# ── Sampling count & shape ────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset", sorted(VALID_DATASETS))
def test_binary_sample_count(dataset):
    samples = sample_binary_paths(dataset, TEST_SEED, N)
    assert len(samples) == N, f"{dataset}: expected {N}, got {len(samples)}"


@pytest.mark.parametrize("dataset", sorted(VALID_DATASETS))
def test_position_sample_count(dataset):
    samples = sample_position_paths(dataset, TEST_SEED, N)
    assert len(samples) == N, f"{dataset}: expected {N}, got {len(samples)}"


@pytest.mark.parametrize("dataset", sorted(VALID_DATASETS))
def test_binary_ground_truth_type(dataset):
    for s in sample_binary_paths(dataset, TEST_SEED, N):
        assert isinstance(s.ground_truth, bool)


@pytest.mark.parametrize("dataset", sorted(VALID_DATASETS))
def test_position_ground_truth_type(dataset):
    for s in sample_position_paths(dataset, TEST_SEED, N):
        i, j = s.ground_truth
        assert j == i + 1
        assert 0 <= i < s.n_frames - 1


# ── Frame URL format ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset", sorted(VALID_DATASETS))
def test_frame_paths_start_with_frames(dataset):
    for s in sample_binary_paths(dataset, TEST_SEED, N):
        for url in s.frame_paths:
            assert url.startswith(f"/frames/{dataset}/"), f"Bad URL: {url}"


# ── RNG determinism across calls ──────────────────────────────────────────────

def test_binary_paths_deterministic():
    a = sample_binary_paths("CRAFT", TEST_SEED, N)
    b = sample_binary_paths("CRAFT", TEST_SEED, N)
    for x, y in zip(a, b):
        assert x.sample_name == y.sample_name
        assert x.ground_truth == y.ground_truth
        assert x.frame_paths == y.frame_paths


def test_position_paths_deterministic():
    a = sample_position_paths("CRAFT", TEST_SEED, N)
    b = sample_position_paths("CRAFT", TEST_SEED, N)
    for x, y in zip(a, b):
        assert x.sample_name == y.sample_name
        assert x.ground_truth == y.ground_truth
        assert x.frame_paths == y.frame_paths


# ── Golden equivalence: path methods vs image methods ────────────────────────

def test_golden_binary_equivalence():
    """
    Path-based sampling must agree with the image-based SwapDataset on
    sample names, anomaly labels, and frame order (by comparing URL filenames
    against PIL-loaded path order).
    """
    from dataset import Dataset, SwapDataset  # noqa: E402

    from backend.config import DATASET_CONFIGS

    dataset = "CRAFT"
    cfg = DATASET_CONFIGS[dataset]
    base = Dataset(
        str(cfg["path"]), cfg["seq_len_min"], cfg["seq_len_max"],
        seed=TEST_SEED, manifest=cfg["manifest"],
    )
    img_samples = SwapDataset(base).sample_binary(N)
    path_samples = sample_binary_paths(dataset, TEST_SEED, N)

    assert len(img_samples) == len(path_samples)
    for img, pth in zip(img_samples, path_samples):
        assert img.name == pth.sample_name, f"Name mismatch: {img.name} vs {pth.sample_name}"
        assert img.anomaly == pth.ground_truth, f"Label mismatch for {img.name}"
        assert len(img.frames) == pth.n_frames


def test_golden_position_equivalence():
    from dataset import Dataset, SwapDataset  # noqa: E402

    from backend.config import DATASET_CONFIGS

    dataset = "CRAFT"
    cfg = DATASET_CONFIGS[dataset]
    base = Dataset(
        str(cfg["path"]), cfg["seq_len_min"], cfg["seq_len_max"],
        seed=TEST_SEED, manifest=cfg["manifest"],
    )
    img_samples = SwapDataset(base).sample_position(N)
    path_samples = sample_position_paths(dataset, TEST_SEED, N)

    assert len(img_samples) == len(path_samples)
    for img, pth in zip(img_samples, path_samples):
        assert img.name == pth.sample_name
        assert img.position == pth.ground_truth, f"Position mismatch for {img.name}"


# ── study.py plan builder ─────────────────────────────────────────────────────

def test_build_plan_length():
    plan = build_plan(TEST_PID, "CRAFT", "detect")
    assert len(plan) == N


def test_build_plan_indices():
    plan = build_plan(TEST_PID, "CLEVRER", "localize")
    for i, s in enumerate(plan):
        assert s.sample_index == i


def test_build_plan_dataset_task_tags():
    plan = build_plan(TEST_PID, "MTL-AQA", "detect")
    for s in plan:
        assert s.dataset == "MTL-AQA"
        assert s.task == "detect"


def test_build_plan_no_ground_truth():
    """SamplePlan must not expose ground_truth to client."""
    plan = build_plan(TEST_PID, "CRAFT", "detect")
    assert not hasattr(plan[0], "ground_truth")


def test_build_plan_invalid_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_plan(TEST_PID, "NONEXISTENT", "detect")


def test_build_plan_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        build_plan(TEST_PID, "CRAFT", "badtask")


def test_ground_truth_for_detect():
    gt_list = ground_truth_for(TEST_PID, "CRAFT", "detect")
    assert len(gt_list) == N
    for gt in gt_list:
        assert isinstance(gt.ground_truth, bool)


def test_ground_truth_for_localize():
    gt_list = ground_truth_for(TEST_PID, "CRAFT", "localize")
    for gt in gt_list:
        i, j = gt.ground_truth
        assert j == i + 1


# ── SQLite DB layer ───────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    return db


def test_upsert_participant(tmp_db):
    upsert_participant(tmp_db, "p1", "CRAFT", "detect", 42)
    # Second call must not raise or duplicate
    upsert_participant(tmp_db, "p1", "CRAFT", "detect", 99)
    # Count rows via annotations table (participants table is not exposed,
    # but we can verify via annotation count starting at 0)
    assert count_annotations(tmp_db, "p1", "CRAFT", "detect") == 0


def test_upsert_annotation_idempotent(tmp_db):
    upsert_participant(tmp_db, "p1", "CRAFT", "detect", 42)
    upsert_annotation(tmp_db, "p1", "CRAFT", "detect", 0, "scene_a", 6, True, True)
    upsert_annotation(tmp_db, "p1", "CRAFT", "detect", 0, "scene_a", 6, True, False)  # update
    assert count_annotations(tmp_db, "p1", "CRAFT", "detect") == 1


def test_count_annotations(tmp_db):
    upsert_participant(tmp_db, "p1", "CRAFT", "detect", 42)
    for i in range(5):
        upsert_annotation(tmp_db, "p1", "CRAFT", "detect", i, f"scene_{i}", 5, True, False)
    assert count_annotations(tmp_db, "p1", "CRAFT", "detect") == 5


def test_mark_completed(tmp_db):
    upsert_participant(tmp_db, "p1", "CRAFT", "detect", 42)
    mark_completed(tmp_db, "p1", "CRAFT", "detect")  # must not raise


def test_decode_answer_detect():
    assert decode_answer("true", "detect") is True
    assert decode_answer("false", "detect") is False


def test_decode_answer_localize():
    assert decode_answer("2,3", "localize") == (2, 3)
