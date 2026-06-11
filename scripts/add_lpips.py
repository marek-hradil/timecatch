"""Patch swap_localize and swap_detect result CSVs with an `lpips` column.

For each sample where a swap occurred, computes the LPIPS perceptual distance
between the two swapped frames and appends it as a new column. Samples with no
swap (swap_detect ground_truth=False) get an empty cell.

The swap pair is read from the swap_localize CSV (ground_truth column). For
swap_detect CSVs, the pair is looked up from the corresponding swap_localize
CSV in the same results directory.

Usage:
    # patch one results dir
    python scripts/add_lpips.py results/qwen3-vl-8b

    # patch all results dirs
    python scripts/add_lpips.py results/*/
"""
import ast
import csv
import os
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

# LPIPS expects [-1, 1] tensors of shape (1, 3, H, W)
_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                        # [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),        # [-1, 1]
])


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _to_tensor(img).unsqueeze(0)           # (1, 3, H, W)


def compute_lpips_for_dataset(
    loss_fn: lpips.LPIPS,
    scenes_root: Path,
    pairs: dict[str, tuple[int, int]],            # scenario → (i, j)
) -> dict[str, float]:
    """Return scenario → lpips distance for all pairs."""
    results = {}
    n = len(pairs)
    for idx, (scenario, (i, j)) in enumerate(pairs.items(), 1):
        frame_i = scenes_root / scenario / f"{i}.jpg"
        frame_j = scenes_root / scenario / f"{j}.jpg"
        if not frame_i.exists() or not frame_j.exists():
            # try .png fallback
            frame_i = frame_i.with_suffix(".png")
            frame_j = frame_j.with_suffix(".png")
        if not frame_i.exists() or not frame_j.exists():
            print(f"  WARNING: frames not found for {scenario} ({i}, {j})")
            continue
        with torch.no_grad():
            dist = loss_fn(load_image(frame_i), load_image(frame_j))
        results[scenario] = float(dist)
        if idx % 500 == 0 or idx == n:
            print(f"    {idx}/{n}  last={results[scenario]:.4f}")
    return results


def read_swap_pairs(localize_csv: Path) -> dict[str, tuple[int, int]]:
    """Read scenario → (i, j) from a swap_localize CSV."""
    pairs = {}
    with open(localize_csv, newline="") as f:
        for row in csv.DictReader(f):
            gt = row["ground_truth"].strip()
            if gt.startswith("("):
                try:
                    a, b = ast.literal_eval(gt)
                    pairs[row["scenario"]] = (int(a), int(b))
                except Exception:
                    pass
    return pairs


def patch_csv(csv_path: Path, lpips_map: dict[str, float]) -> None:
    """Add/overwrite the `lpips` column in a CSV file in-place."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "lpips" not in fieldnames:
        fieldnames = list(fieldnames) + ["lpips"]

    patched = 0
    for row in rows:
        scenario = row["scenario"]
        if scenario in lpips_map:
            row["lpips"] = f"{lpips_map[scenario]:.6f}"
            patched += 1
        elif "lpips" not in row:
            row["lpips"] = ""

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  patched {patched}/{len(rows)} rows → {csv_path.name}")


def process_results_dir(results_dir: Path, loss_fn: lpips.LPIPS) -> None:
    print(f"\n=== {results_dir.name} ===")
    datasets = list(DATASET_ROOTS.keys())

    for ds in datasets:
        scenes_root = DATASET_ROOTS[ds]
        localize_csv = results_dir / f"swap_localize_{ds}.csv"
        detect_csv   = results_dir / f"swap_detect_{ds}.csv"

        if not localize_csv.exists():
            print(f"  skip {ds}: no swap_localize CSV")
            continue

        print(f"\n  [{ds}] computing LPIPS...")
        pairs = read_swap_pairs(localize_csv)
        lpips_map = compute_lpips_for_dataset(loss_fn, scenes_root, pairs)

        patch_csv(localize_csv, lpips_map)
        if detect_csv.exists():
            patch_csv(detect_csv, lpips_map)


def main(dirs: list[str]) -> None:
    print("Loading LPIPS (alex net)...")
    loss_fn = lpips.LPIPS(net="alex")
    loss_fn.eval()

    result_dirs = [Path(d) for d in dirs]
    if not result_dirs:
        result_dirs = sorted((BASE / "results").glob("*/"))

    for d in result_dirs:
        if d.is_dir():
            process_results_dir(d, loss_fn)

    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
