"""Add avg_seq_lpips column to swap_detect CSVs.

For each scene, computes the average LPIPS across all C(n,2) frame pairs
and writes it as `seq_lpips`. All pairs for a scene are batched into a
single forward pass for speed.

Usage:
  python scripts/add_seq_lpips.py results/qwen3-vl-8b
  python scripts/add_seq_lpips.py results/*/
"""
import csv
import itertools
import sys
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision import transforms

BASE = Path(__file__).resolve().parent.parent

DATASET_ROOTS = {
    "clevrer":  BASE / "datasets/CLEVRER/scenes",
    "craft":    BASE / "datasets/CRAFT/scenes",
    "drive-lm": BASE / "datasets/drive_lm/scenes",
    "mtl-aqa":  BASE / "datasets/MTL-AQA/scenes",
}

_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_frame(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _to_tensor(img)


def load_scene_frames(scenes_root: Path, scenario: str, n: int) -> list[torch.Tensor] | None:
    frames = []
    for i in range(n):
        for ext in (".jpg", ".png"):
            p = scenes_root / scenario / f"{i}{ext}"
            if p.exists():
                frames.append(load_frame(p))
                break
        else:
            return None
    return frames


def avg_seq_lpips(loss_fn: lpips.LPIPS, frames: list[torch.Tensor]) -> float:
    pairs = list(itertools.combinations(range(len(frames)), 2))
    a = torch.stack([frames[i] for i, _ in pairs])  # (P, 3, H, W)
    b = torch.stack([frames[j] for _, j in pairs])
    with torch.no_grad():
        dists = loss_fn(a, b)                        # (P, 1, 1, 1)
    return float(dists.mean())


def process(results_dir: Path, loss_fn: lpips.LPIPS) -> None:
    print(f"\n=== {results_dir.name} ===")
    for ds, scenes_root in DATASET_ROOTS.items():
        csv_path = results_dir / f"swap_detect_{ds}.csv"
        if not csv_path.exists():
            print(f"  skip {ds}: no swap_detect CSV")
            continue

        rows = list(csv.DictReader(open(csv_path, newline="")))
        fieldnames = list(rows[0].keys())
        if "seq_lpips" not in fieldnames:
            fieldnames.append("seq_lpips")

        # build scenario → seq_len map (take first occurrence)
        scene_lens: dict[str, int] = {}
        for r in rows:
            sc = r["scenario"]
            if sc not in scene_lens:
                scene_lens[sc] = int(r["seq_len"])

        print(f"  [{ds}] computing seq LPIPS for {len(scene_lens)} scenes…")
        seq_lpips_map: dict[str, float] = {}
        for idx, (scenario, n) in enumerate(scene_lens.items(), 1):
            frames = load_scene_frames(scenes_root, scenario, n)
            if frames is None:
                print(f"    WARNING: missing frames for {scenario}")
                continue
            seq_lpips_map[scenario] = avg_seq_lpips(loss_fn, frames)
            if idx % 500 == 0 or idx == len(scene_lens):
                print(f"    {idx}/{len(scene_lens)}")

        patched = 0
        for r in rows:
            sc = r["scenario"]
            if sc in seq_lpips_map:
                r["seq_lpips"] = f"{seq_lpips_map[sc]:.6f}"
                patched += 1
            elif "seq_lpips" not in r:
                r["seq_lpips"] = ""

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        print(f"  patched {patched}/{len(rows)} rows → {csv_path.name}")


def main(dirs: list[str]) -> None:
    print("Loading LPIPS (alexnet)…")
    loss_fn = lpips.LPIPS(net="alex")
    loss_fn.eval()

    result_dirs = [Path(d) for d in dirs] if dirs else sorted((BASE / "results").glob("*/"))
    for d in result_dirs:
        if d.is_dir():
            process(d, loss_fn)
    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
