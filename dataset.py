import json
import os
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image as PILImage


@dataclass
class Scenario:
    name: str
    frames: list[PILImage.Image]
    scene_description: str | None = None


@dataclass
class BinaryScenario:
    name: str
    frames: list[PILImage.Image]
    anomaly: bool
    scene_description: str | None = None


@dataclass
class PositionScenario:
    name: str
    frames: list[PILImage.Image]
    position: tuple[int, int]
    scene_description: str | None = None


# Internal index entry: (name, frame_paths, scene_description)
_Entry = tuple[str, list[str], str | None]


class Dataset:
    def __init__(
        self,
        path: str,
        seq_len_min: int = 4,
        seq_len_max: int = 8,
        seed: int = 42,
        manifest: str | None = None,
    ):
        self.seed = seed
        root = Path(path)
        self._index: list[_Entry] = (
            self._load_manifest(root, manifest, seq_len_min, seq_len_max)
            if manifest
            else self._scan_dirs(root, seq_len_min, seq_len_max)
        )

    @staticmethod
    def _load_manifest(root: Path, manifest: str, seq_len_min: int, seq_len_max: int) -> list[_Entry]:
        with open(root / manifest) as f:
            entries = json.load(f)
        index = []
        for entry in entries:
            paths = [str(root / p) for p in entry["scene_paths"]]
            if seq_len_min <= len(paths) <= seq_len_max:
                name = Path(entry["scene_paths"][0]).parent.name
                index.append((name, paths, entry.get("scene_description")))
        return index

    @staticmethod
    def _scan_dirs(root: Path, seq_len_min: int, seq_len_max: int) -> list[_Entry]:
        index = []
        for name in sorted(os.listdir(root)):
            d = root / name
            if not d.is_dir():
                continue
            paths = Dataset._frame_paths(d)
            if seq_len_min <= len(paths) <= seq_len_max:
                index.append((name, paths, None))
        return index

    @staticmethod
    def _frame_paths(d: Path) -> list[str]:
        exts = {".png", ".jpg", ".jpeg"}
        paths = [
            str(d / name)
            for name in os.listdir(d)
            if not name.startswith(".") and Path(name).suffix.lower() in exts
        ]
        paths.sort(key=lambda p: int(Path(p).stem))
        return paths

    def _group_by_length(self) -> dict[int, list[_Entry]]:
        groups: dict[int, list[_Entry]] = {}
        for entry in self._index:
            groups.setdefault(len(entry[1]), []).append(entry)
        return groups

    def _select(self, n: int | None, rng: random.Random) -> list[_Entry]:
        if n is None:
            selected = list(self._index)
            rng.shuffle(selected)
            return selected

        groups = self._group_by_length()
        lengths = sorted(groups.keys())
        per_length, remainder = divmod(n, len(lengths))
        selected = []
        for i, length in enumerate(lengths):
            k = min(per_length + (1 if i < remainder else 0), len(groups[length]))
            selected.extend(rng.sample(groups[length], k))
        rng.shuffle(selected)
        return selected

    @staticmethod
    def _load(name: str, paths: list[str], scene_description: str | None) -> Scenario:
        return Scenario(
            name=name,
            frames=[PILImage.open(p).copy() for p in paths],
            scene_description=scene_description,
        )

    def sample(self, n: int | None = None) -> list[Scenario]:
        rng = random.Random(self.seed)
        return [self._load(name, paths, desc) for name, paths, desc in self._select(n, rng)]


class SwapDataset:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._seed = dataset.seed

    def _swap(self, frames: list[PILImage.Image], rng: random.Random) -> tuple[list[PILImage.Image], tuple[int, int]]:
        frames = list(frames)
        i = rng.randrange(len(frames) - 1)
        frames[i], frames[i + 1] = frames[i + 1], frames[i]
        return frames, (i, i + 1)

    def sample_binary(self, n: int | None = None) -> list[BinaryScenario]:
        rng = random.Random(self._seed)
        result = []
        for s in self._dataset.sample(n):
            if rng.random() < 0.5:
                frames, _ = self._swap(s.frames, rng)
                result.append(BinaryScenario(name=s.name, frames=frames, anomaly=True, scene_description=s.scene_description))
            else:
                result.append(BinaryScenario(name=s.name, frames=list(s.frames), anomaly=False, scene_description=s.scene_description))
        return result

    def sample_position(self, n: int | None = None) -> list[PositionScenario]:
        rng = random.Random(self._seed)
        result = []
        for s in self._dataset.sample(n):
            frames, position = self._swap(s.frames, rng)
            result.append(PositionScenario(name=s.name, frames=frames, position=position, scene_description=s.scene_description))
        return result


class ShuffleDataset:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._seed = dataset.seed

    def _derange(self, frames: list, rng: random.Random) -> list:
        indices = list(range(len(frames)))
        while True:
            rng.shuffle(indices)
            if all(i != j for i, j in enumerate(indices)):
                break
        return [frames[i] for i in indices]

    def sample_binary(self, n: int | None = None) -> list[BinaryScenario]:
        rng = random.Random(self._seed)
        result = []
        for s in self._dataset.sample(n):
            if rng.random() < 0.5:
                frames = self._derange(s.frames, rng)
                result.append(BinaryScenario(name=s.name, frames=frames, anomaly=True, scene_description=s.scene_description))
            else:
                result.append(BinaryScenario(name=s.name, frames=list(s.frames), anomaly=False, scene_description=s.scene_description))
        return result


class CorruptDataset:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._seed = dataset.seed

    def _corrupt(self, frames: list[PILImage.Image], rng: random.Random) -> tuple[list[PILImage.Image], tuple[int, int]]:
        frames = list(frames)
        i = rng.randrange(len(frames))
        w, h = frames[i].size
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        noise = np_rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        frames[i] = PILImage.fromarray(noise)
        return frames, (i, i)

    def sample_binary(self, n: int | None = None) -> list[BinaryScenario]:
        rng = random.Random(self._seed)
        result = []
        for s in self._dataset.sample(n):
            if rng.random() < 0.5:
                frames, _ = self._corrupt(s.frames, rng)
                result.append(BinaryScenario(name=s.name, frames=frames, anomaly=True, scene_description=s.scene_description))
            else:
                result.append(BinaryScenario(name=s.name, frames=list(s.frames), anomaly=False, scene_description=s.scene_description))
        return result

    def sample_position(self, n: int | None = None) -> list[PositionScenario]:
        rng = random.Random(self._seed)
        result = []
        for s in self._dataset.sample(n):
            frames, position = self._corrupt(s.frames, rng)
            result.append(PositionScenario(name=s.name, frames=frames, position=position, scene_description=s.scene_description))
        return result
