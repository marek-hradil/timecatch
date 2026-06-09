"""
Path-based sampling — mirrors Dataset/SwapDataset from dataset.py exactly,
but operates on frame path lists instead of loaded PIL images.

Critical: the RNG draw sequence must match the image-based original precisely:
  - sample_binary_paths: per sample, draws rng.random() (coin flip);
    on swap draws rng.randrange(len-1); NON-swap draws NOTHING.
  - sample_position_paths: per sample, draws rng.randrange(len-1) unconditionally.

This asymmetry is load-bearing — any deviation shifts swap indices for all
subsequent samples and breaks determinism.
"""
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset import Dataset  # noqa: E402

from .config import DATASET_CONFIGS, DATASETS_ROOT


@dataclass
class BinaryPathSample:
    sample_name: str
    n_frames: int
    frame_paths: list[str]   # paths in swapped order
    ground_truth: bool
    scene_description: Optional[str]


@dataclass
class PositionPathSample:
    sample_name: str
    n_frames: int
    frame_paths: list[str]   # paths in swapped order
    ground_truth: tuple[int, int]
    scene_description: Optional[str]


def _swap_paths(paths: list[str], rng: random.Random) -> tuple[list[str], tuple[int, int]]:
    """Mirror of SwapDataset._swap but on path strings."""
    paths = list(paths)
    i = rng.randrange(len(paths) - 1)
    paths[i], paths[i + 1] = paths[i + 1], paths[i]
    return paths, (i, i + 1)


def _path_to_url(abs_path: str, dataset_name: str) -> str:
    """
    Convert an absolute frame path to a /frames/{dataset}/{relative} URL.
    Strips the DATASETS_ROOT/{dataset_name}/ prefix.
    """
    rel = Path(abs_path).relative_to(DATASETS_ROOT / dataset_name)
    return f"/frames/{dataset_name}/{rel.as_posix()}"


def sample_binary_paths(dataset_name: str, seed: int, n: int) -> list[BinaryPathSample]:
    """
    Returns n binary (detect) samples with frame paths in possibly-swapped order.
    Mirrors SwapDataset.sample_binary exactly (same RNG draw sequence).
    """
    cfg = DATASET_CONFIGS[dataset_name]
    base = Dataset(
        str(cfg["path"]),
        cfg["seq_len_min"],
        cfg["seq_len_max"],
        seed=seed,
        manifest=cfg["manifest"],
    )
    entries = base._select(n, random.Random(seed))  # same rng as Dataset.sample()

    rng = random.Random(seed)  # fresh generator — mirrors SwapDataset.__init__
    results = []
    for name, paths, desc in entries:
        if rng.random() < 0.5:
            swapped_paths, _ = _swap_paths(paths, rng)  # draws randrange
            anomaly = True
        else:
            swapped_paths = list(paths)
            anomaly = False  # NO randrange draw — load-bearing asymmetry
        urls = [_path_to_url(p, dataset_name) for p in swapped_paths]
        results.append(BinaryPathSample(
            sample_name=name,
            n_frames=len(paths),
            frame_paths=urls,
            ground_truth=anomaly,
            scene_description=desc,
        ))
    return results


def sample_position_paths(dataset_name: str, seed: int, n: int) -> list[PositionPathSample]:
    """
    Returns n localize samples with frame paths in swapped order.
    Mirrors SwapDataset.sample_position exactly (draws randrange unconditionally).
    """
    cfg = DATASET_CONFIGS[dataset_name]
    base = Dataset(
        str(cfg["path"]),
        cfg["seq_len_min"],
        cfg["seq_len_max"],
        seed=seed,
        manifest=cfg["manifest"],
    )
    entries = base._select(n, random.Random(seed))

    rng = random.Random(seed)
    results = []
    for name, paths, desc in entries:
        swapped_paths, position = _swap_paths(paths, rng)
        urls = [_path_to_url(p, dataset_name) for p in swapped_paths]
        results.append(PositionPathSample(
            sample_name=name,
            n_frames=len(paths),
            frame_paths=urls,
            ground_truth=position,
            scene_description=desc,
        ))
    return results
