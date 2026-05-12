"""Shared utilities for per-dataset describe_dataset scripts."""

import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model() -> tuple:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval().to(device)
    return model, device


def embed(path: str, model, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = _TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x)


def embed_fg(path: str, model, device, bg_emb: torch.Tensor) -> torch.Tensor:
    """Embed image and subtract background embedding, then re-normalise."""
    emb = embed(path, model, device)
    residual = emb - bg_emb
    return F.normalize(residual, dim=-1)


def load_scenes(dataset_dir: str) -> list:
    with open(os.path.join(dataset_dir, "scenes.json")) as f:
        return json.load(f)


def consecutive_pairs(scenes: list, dataset_dir: str) -> list[tuple[str, str, str]]:
    pairs = []
    for scene in scenes:
        scene_id = str(scene["video_index"])
        for a, b in zip(scene["scene_paths"], scene["scene_paths"][1:]):
            path_a = os.path.join(dataset_dir, a)
            path_b = os.path.join(dataset_dir, b)
            if os.path.exists(path_a) and os.path.exists(path_b):
                pairs.append((path_a, path_b, scene_id))
    return pairs


def compute_scores(pairs: list, model, device, bg_emb: torch.Tensor | None = None) -> list[float]:
    scores = []
    for a, b, _ in tqdm(pairs, desc="Computing similarity"):
        if bg_emb is not None:
            ea = embed_fg(a, model, device, bg_emb)
            eb = embed_fg(b, model, device, bg_emb)
        else:
            ea = embed(a, model, device)
            eb = embed(b, model, device)
        scores.append(F.cosine_similarity(ea, eb).item())
    return scores


def print_overview(scenes: list, name: str):
    total = sum(len(s["scene_paths"]) for s in scenes)
    print(f"Dataset: {name}  |  Scenes: {len(scenes)}  |  Images: {total}")


def print_frame_histogram(scenes: list):
    counts = Counter(len(s["scene_paths"]) for s in scenes)
    print("\nFrames per scene:")
    for n in sorted(counts):
        print(f"  {n}: {'#' * (counts[n] // 10)} ({counts[n]})")


def print_histogram(scores: list[float], label: str, bins: int = 20):
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    print(f"\n{label} ({len(scores)} pairs):")
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        print(f"  {lo:.2f}-{hi:.2f}: {'#' * (count // max(1, len(scores) // 500))} ({count})")
    print(f"\n  mean={np.mean(scores):.3f}  median={np.median(scores):.3f}  "
          f"min={np.min(scores):.3f}  max={np.max(scores):.3f}")
    for threshold in (0.95, 0.98):
        above = sum(1 for s in scores if s >= threshold)
        print(f"  >={threshold}: {above} ({100*above/len(scores):.1f}%)")


def print_pairs_in_range(pairs: list, scores: list[float], lo: float, hi: float):
    matches = [(s, a, b, sid) for (a, b, sid), s in zip(pairs, scores) if lo <= s < hi]
    print(f"\nPairs with similarity {lo:.2f}-{hi:.2f} ({len(matches)}):")
    for score, a, b, sid in sorted(matches, reverse=True)[:20]:
        print(f"  {score:.3f}  scene {sid}  {os.path.basename(a)} -> {os.path.basename(b)}")
