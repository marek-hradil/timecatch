"""Fresh 80/20 split of CRAFT's FULL filtered pool (train+validation+test
splits combined, recovered via scripts/craft/prepare_frames_full.py — the
original CRAFT fine-tune only ever used the 1,983-video official test
split) for the LoRA fine-tuning pilot.

Mirrors scripts/split_finetune_mtlaqa_full.py's approach: draws a fresh
80/20 split from the full pool rather than keeping the old (test-split-only)
172-scene held-out set fixed, since that held-out set is itself a small
subset of what's now available and a fresh split gives a more precise
standalone read.

Reads:  datasets/CRAFT/scenes_filtered_full.json
Writes: datasets/CRAFT/scenes_finetune_train_full80.json
        datasets/CRAFT/scenes_finetune_heldout_full80.json

Usage:
    python3 scripts/split_finetune_craft_full.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "datasets" / "CRAFT" / "scenes_filtered_full.json"
TRAIN_OUT = ROOT / "datasets" / "CRAFT" / "scenes_finetune_train_full80.json"
HELDOUT_OUT = ROOT / "datasets" / "CRAFT" / "scenes_finetune_heldout_full80.json"

SEED = 42
TRAIN_FRACTION = 0.8


def main():
    scenes = json.loads(SRC.read_text())

    buckets = defaultdict(list)
    for scene in scenes:
        buckets[len(scene["scene_paths"])].append(scene)

    rng = random.Random(SEED)
    train, heldout = [], []
    for length in sorted(buckets):
        group = list(buckets[length])
        rng.shuffle(group)
        n_train = round(len(group) * TRAIN_FRACTION)
        train.extend(group[:n_train])
        heldout.extend(group[n_train:])
        print(f"  len={length}: {len(group)} scenes -> {n_train} train / {len(group) - n_train} held-out")

    rng.shuffle(train)
    rng.shuffle(heldout)

    train_ids = {s["video_index"] for s in train}
    heldout_ids = {s["video_index"] for s in heldout}
    assert not (train_ids & heldout_ids), "leakage between splits"

    TRAIN_OUT.write_text(json.dumps(train, indent=2))
    HELDOUT_OUT.write_text(json.dumps(heldout, indent=2))

    print(f"\nTotal: {len(scenes)} scenes -> {len(train)} train / {len(heldout)} held-out")
    print(f"Written to {TRAIN_OUT.name} and {HELDOUT_OUT.name}")


if __name__ == "__main__":
    main()
