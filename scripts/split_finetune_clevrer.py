"""Split CLEVRER's filtered scenes 80/20 for the LoRA fine-tuning pilot.

Split is stratified by frame-length bucket so both halves have matching
length distributions. The train half is the SFT fine-tuning source; the
held-out half is disjoint (by scene) and used only for post-training eval.

Reads:  datasets/CLEVRER/scenes_filtered.json
Writes: datasets/CLEVRER/scenes_finetune_train.json
        datasets/CLEVRER/scenes_finetune_heldout.json

Usage:
    python3 scripts/split_finetune_clevrer.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "datasets" / "CLEVRER" / "scenes_filtered.json"
TRAIN_OUT = ROOT / "datasets" / "CLEVRER" / "scenes_finetune_train.json"
HELDOUT_OUT = ROOT / "datasets" / "CLEVRER" / "scenes_finetune_heldout.json"

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
