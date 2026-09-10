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
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset import Dataset  # noqa: E402

from .config import DATASET_CONFIGS, DATASETS_ROOT, GCS_PUBLIC_BASE


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
    rel = Path(abs_path).relative_to(DATASETS_ROOT / dataset_name)
    if GCS_PUBLIC_BASE:
        return f"{GCS_PUBLIC_BASE}/datasets/{dataset_name}/{rel.as_posix()}"
    return f"/frames/{dataset_name}/{rel.as_posix()}"


_ATTENTION_CHECKS_PATH = Path(__file__).parent / "attention_checks.json"


def load_attention_check_samples(
    dataset_name: str,
    task: str,
    n: int,
) -> list[BinaryPathSample | PositionPathSample]:
    """Load fixed attention-check samples from attention_checks.json."""
    if not _ATTENTION_CHECKS_PATH.exists():
        return []

    with open(_ATTENTION_CHECKS_PATH) as f:
        all_checks = json.load(f)

    entries = all_checks.get(f"{dataset_name}_{task}", [])[:n]
    if not entries:
        return []

    cfg = DATASET_CONFIGS[dataset_name]
    root = cfg["path"]

    with open(root / cfg["manifest"]) as f:
        manifest_data = json.load(f)

    scene_index: dict[str, tuple[list[str], str | None]] = {}
    for entry in manifest_data:
        name = Path(entry["scene_paths"][0]).parent.name
        paths = [str(root / p) for p in entry["scene_paths"]]
        scene_index[name] = (paths, entry.get("scene_description"))

    results: list[BinaryPathSample | PositionPathSample] = []
    for check in entries:
        sample_name = check["sample_name"]
        swap_pair = check.get("swap_pair")

        if sample_name not in scene_index:
            raise ValueError(
                f"Attention check {sample_name!r} not found in {dataset_name} manifest"
            )

        paths, desc = scene_index[sample_name]

        if swap_pair is not None:
            i, j = int(swap_pair[0]), int(swap_pair[1])
            ordered = list(paths)
            ordered[i], ordered[j] = ordered[j], ordered[i]
            urls = [_path_to_url(p, dataset_name) for p in ordered]
        else:
            urls = [_path_to_url(p, dataset_name) for p in paths]

        if task == "detect":
            results.append(BinaryPathSample(
                sample_name=sample_name,
                n_frames=len(paths),
                frame_paths=urls,
                ground_truth=swap_pair is not None,
                scene_description=desc,
            ))
        else:
            if swap_pair is None:
                raise ValueError(
                    f"Localize attention check {sample_name!r} must have a swap_pair"
                )
            results.append(PositionPathSample(
                sample_name=sample_name,
                n_frames=len(paths),
                frame_paths=urls,
                ground_truth=(int(swap_pair[0]), int(swap_pair[1])),
                scene_description=desc,
            ))

    return results


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
