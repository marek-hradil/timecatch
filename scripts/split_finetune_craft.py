"""Split CRAFT's 4-8 frame scenes 80/20 for the LoRA fine-tuning pilot.

Unlike CLEVRER/DriveLM, CRAFT's scenes_filtered.json mixes the 4-8 frame
"main results" subset with the 9-16 frame CRAFT-long subset (used only for
the sequence-length ablation, see AGENTS.md). Filter to 4-8 frames FIRST,
before splitting, so CRAFT-long sequences never leak into fine-tuning data.

Mirrors scripts/split_finetune_clevrer.py otherwise — see that file for
rationale on the stratified 80/20 split.

Reads:  datasets/CRAFT/scenes_filtered.json
Writes: datasets/CRAFT/scenes_finetune_train.json
        datasets/CRAFT/scenes_finetune_heldout.json

Usage:
    python3 scripts/split_finetune_craft.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "datasets" / "CRAFT" / "scenes_filtered.json"
TRAIN_OUT = ROOT / "datasets" / "CRAFT" / "scenes_finetune_train.json"
HELDOUT_OUT = ROOT / "datasets" / "CRAFT" / "scenes_finetune_heldout.json"

SEED = 42
TRAIN_FRACTION = 0.8
SEQ_LEN_MIN = 4
SEQ_LEN_MAX = 8


def main():
    all_scenes = json.loads(SRC.read_text())
    scenes = [s for s in all_scenes if SEQ_LEN_MIN <= len(s["scene_paths"]) <= SEQ_LEN_MAX]
    print(f"Filtered to {SEQ_LEN_MIN}-{SEQ_LEN_MAX} frames: {len(all_scenes)} -> {len(scenes)} scenes "
          f"({len(all_scenes) - len(scenes)} CRAFT-long scenes excluded)")

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
